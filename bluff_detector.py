"""
bluff_detector.py — Détection de patterns de bluff et tendances adverses via Claude

Ce module analyse les séquences d'actions adverses pour détecter :
  - Les patterns de bluff (mise systématique sur certaines textures de board)
  - Les tendances exploitables (always cbet, always fold to 3bet, etc.)
  - Les tells comportementaux (timing, sizing, séquences répétitives)
  - Les lines suspects (check-raise river sans value, overbet bluff, etc.)

Architecture :
  1. PatternAccumulator  — collecte les séquences d'actions par adversaire
  2. BluffDetector       — analyse locale rapide (règles heuristiques)
  3. ClaudePatternAnalyser — analyse IA approfondie (batch, pas temps réel)
  4. TendencyReport      — rapport exploitable par main

Dépendances : profil_builder.py, action_detector.py, claude_client.py
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Seuils d'exploitation (% d'occurrences pour considérer un pattern fiable)
PATTERN_THRESHOLD     = 0.65   # ≥ 65% d'occurrences → pattern exploitable
PATTERN_MIN_SAMPLES   = 5      # minimum d'échantillons avant de tirer des conclusions

# Catégories de patterns détectés
class PatternType(str, Enum):
    ALWAYS_CBET         = "always_cbet"          # cbet systématique
    NEVER_CBET          = "never_cbet"            # ne cbet jamais
    ALWAYS_FOLD_3BET    = "always_fold_3bet"      # fold toujours au 3-bet
    ALWAYS_CALL_3BET    = "always_call_3bet"      # call toujours le 3-bet
    OVERBET_BLUFF       = "overbet_bluff"          # overbet = bluff probable
    SMALL_BET_VALUE     = "small_bet_value"        # petite mise = value
    CHECK_RAISE_BLUFF   = "check_raise_bluff"      # check-raise = bluff freq.
    RIVER_BARREL        = "river_barrel"           # triple barrel fréquent
    FLOAT_CALL          = "float_call"             # call fréquent pour steal turn
    DONK_BET            = "donk_bet"              # mise hors de position fréquente
    SIZING_TELL         = "sizing_tell"            # taille de mise = force de main
    TILT_PATTERN        = "tilt_pattern"           # pattern de tilt (après bad beat)
    SHOWDOWN_BLUFF      = "showdown_bluff"         # montré un bluff au SD


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------

@dataclass
class ActionSequence:
    """Séquence d'actions sur une main complète pour un adversaire."""
    player:       str
    hand_id:      str
    stage:        str          # street finale
    board:        list[str]    # cartes du board
    actions:      list[dict]   # [{street, action, amount, pot, pot_ratio}]
    result:       float        # gain/perte
    showed_cards: list[str]    # cartes montrées au showdown
    had_value:    bool         # avait une main forte au showdown

    @property
    def was_bluff(self) -> bool:
        """True si montré au showdown avec une main faible."""
        if not self.showed_cards or not self.had_value:
            return False
        return not self.had_value and self.result > 0  # a gagné sans value

    @property
    def final_action(self) -> str:
        if not self.actions:
            return ""
        return self.actions[-1].get("action", "")

    @property
    def bet_sizes_relative(self) -> list[float]:
        """Retourne les tailles de mise en % du pot."""
        return [
            a.get("pot_ratio", 0.0)
            for a in self.actions
            if a.get("action") in ("BET", "RAISE")
        ]


@dataclass
class DetectedPattern:
    """Pattern comportemental détecté chez un adversaire."""
    player:         str
    pattern_type:   PatternType
    frequency:      float       # 0.0 – 1.0
    sample_count:   int
    description:    str
    exploitation:   str         # conseil d'exploitation
    confidence:     float       # fiabilité (dépend du nb d'échantillons)
    detected_at:    float       = field(default_factory=time.time)

    @property
    def is_reliable(self) -> bool:
        return self.sample_count >= PATTERN_MIN_SAMPLES and self.frequency >= PATTERN_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "player":       self.player,
            "pattern":      self.pattern_type.value,
            "frequency":    round(self.frequency, 2),
            "samples":      self.sample_count,
            "description":  self.description,
            "exploitation": self.exploitation,
            "confidence":   round(self.confidence, 2),
            "reliable":     self.is_reliable,
        }

    def to_hud_line(self) -> str:
        pct = f"{self.frequency:.0%}"
        rel = "✓" if self.is_reliable else "~"
        return f"{self.pattern_type.value:<22} {pct:>5} [{self.sample_count}]{rel}"


@dataclass
class TendencyReport:
    """Rapport complet des tendances d'un adversaire pour une main."""
    player:         str
    patterns:       list[DetectedPattern]
    claude_analysis:str   = ""   # analyse Claude en texte libre
    top_exploitation:str  = ""   # conseil principal
    timestamp:      float = field(default_factory=time.time)

    def to_prompt_context(self) -> str:
        """Format pour injection dans le prompt Claude."""
        if not self.patterns:
            return ""
        lines = [f"Tendances de {self.player} :"]
        for p in self.patterns[:5]:
            if p.is_reliable:
                lines.append(f"  • {p.description} ({p.frequency:.0%}) → {p.exploitation}")
        if self.claude_analysis:
            lines.append(f"  Analyse IA : {self.claude_analysis[:200]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Accumulateur de patterns
# ---------------------------------------------------------------------------

class PatternAccumulator:
    """
    Collecte et stocke les séquences d'actions de chaque adversaire.
    Maintient un historique glissant par joueur.
    """

    def __init__(self, max_history: int = 200):
        self.max_history = max_history
        # {player_name: deque[ActionSequence]}
        self._sequences: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        # Contexte de la main en cours par joueur
        self._current_hand: dict[str, dict] = {}

    def start_hand(self, player: str, hand_id: str, board: list[str]) -> None:
        """Démarre l'enregistrement d'une nouvelle main pour un adversaire."""
        self._current_hand[player] = {
            "hand_id":  hand_id,
            "board":    board,
            "actions":  [],
            "result":   0.0,
            "showed":   [],
            "had_value":False,
        }

    def record_action(
        self,
        player:    str,
        action:    str,
        amount:    float = 0.0,
        pot:       float = 0.0,
        street:    str   = "flop",
    ) -> None:
        """Enregistre une action adverse pendant la main."""
        if player not in self._current_hand:
            self._current_hand[player] = {
                "hand_id": "", "board": [], "actions": [],
                "result": 0.0, "showed": [], "had_value": False,
            }
        pot_ratio = amount / max(pot, 1)
        self._current_hand[player]["actions"].append({
            "street":    street,
            "action":    action,
            "amount":    amount,
            "pot":       pot,
            "pot_ratio": round(pot_ratio, 2),
        })

    def end_hand(
        self,
        player:    str,
        result:    float = 0.0,
        showed:    list  = None,
        had_value: bool  = False,
        stage:     str   = "river",
    ) -> Optional[ActionSequence]:
        """Finalise la main et ajoute la séquence à l'historique."""
        if player not in self._current_hand:
            return None
        ctx = self._current_hand.pop(player)
        seq = ActionSequence(
            player       = player,
            hand_id      = ctx["hand_id"],
            stage        = stage,
            board        = ctx["board"],
            actions      = ctx["actions"],
            result       = result,
            showed_cards = showed or [],
            had_value    = had_value,
        )
        self._sequences[player].append(seq)
        return seq

    def get_sequences(self, player: str, last_n: int = 50) -> list[ActionSequence]:
        """Retourne les N dernières séquences d'un adversaire."""
        seqs = list(self._sequences.get(player, []))
        return seqs[-last_n:]

    def get_all_players(self) -> list[str]:
        return list(self._sequences.keys())

    def total_hands(self, player: str) -> int:
        return len(self._sequences.get(player, []))


# ---------------------------------------------------------------------------
# Détecteur de patterns (heuristiques locales)
# ---------------------------------------------------------------------------

class BluffDetector:
    """
    Détecte les patterns comportementaux par analyse heuristique.
    Rapide, ne nécessite pas l'API Claude.
    """

    def __init__(self, accumulator: PatternAccumulator):
        self.acc = accumulator

    def analyse_player(self, player: str) -> list[DetectedPattern]:
        """Retourne tous les patterns détectés pour un adversaire."""
        sequences = self.acc.get_sequences(player)
        if len(sequences) < PATTERN_MIN_SAMPLES:
            return []

        patterns = []

        patterns += self._detect_cbet_patterns(player, sequences)
        patterns += self._detect_3bet_response(player, sequences)
        patterns += self._detect_sizing_tells(player, sequences)
        patterns += self._detect_barrel_patterns(player, sequences)
        patterns += self._detect_check_raise(player, sequences)
        patterns += self._detect_showdown_bluffs(player, sequences)

        # Trier par fiabilité décroissante
        patterns.sort(key=lambda p: (p.is_reliable, p.frequency), reverse=True)
        return patterns

    # ── CBet ──────────────────────────────────────────────────────────────

    def _detect_cbet_patterns(
        self, player: str, seqs: list[ActionSequence]
    ) -> list[DetectedPattern]:
        patterns = []
        cbet_opps = 0   # mains où il pouvait cbet (pré-flop raiser + voit le flop)
        cbet_done = 0

        for seq in seqs:
            preflop_raised = any(
                a["action"] in ("RAISE", "ALL-IN") and a["street"] == "preflop"
                for a in seq.actions
            )
            saw_flop = any(a["street"] == "flop" for a in seq.actions)

            if preflop_raised and saw_flop:
                cbet_opps += 1
                flop_bet = any(
                    a["action"] in ("BET", "RAISE") and a["street"] == "flop"
                    for a in seq.actions
                )
                if flop_bet:
                    cbet_done += 1

        if cbet_opps >= PATTERN_MIN_SAMPLES:
            freq = cbet_done / cbet_opps
            conf = min(cbet_opps / 20, 1.0)

            if freq >= PATTERN_THRESHOLD:
                patterns.append(DetectedPattern(
                    player       = player,
                    pattern_type = PatternType.ALWAYS_CBET,
                    frequency    = freq,
                    sample_count = cbet_opps,
                    description  = f"CBet systématique ({freq:.0%} des fois)",
                    exploitation = "Float le flop et prends le pot au turn quand il check.",
                    confidence   = conf,
                ))
            elif freq <= (1 - PATTERN_THRESHOLD):
                patterns.append(DetectedPattern(
                    player       = player,
                    pattern_type = PatternType.NEVER_CBET,
                    frequency    = 1 - freq,
                    sample_count = cbet_opps,
                    description  = f"Ne cbet presque jamais ({1-freq:.0%} des checks flop)",
                    exploitation = "Steal le pot au flop quand il check.",
                    confidence   = conf,
                ))
        return patterns

    # ── Réponse au 3-bet ──────────────────────────────────────────────────

    def _detect_3bet_response(
        self, player: str, seqs: list[ActionSequence]
    ) -> list[DetectedPattern]:
        patterns = []
        faced_3bet = 0
        folded_3bet = 0
        called_3bet = 0

        for seq in seqs:
            preflop_actions = [a for a in seq.actions if a["street"] == "preflop"]
            # Détect si le joueur a fait face à un 3-bet (a relancé puis a vu une re-relance)
            raised = False
            for a in preflop_actions:
                if a["action"] in ("RAISE",) and not raised:
                    raised = True
                elif raised and a["action"] == "FOLD":
                    faced_3bet += 1
                    folded_3bet += 1
                elif raised and a["action"] in ("CALL",):
                    faced_3bet += 1
                    called_3bet += 1

        if faced_3bet >= PATTERN_MIN_SAMPLES:
            fold_freq = folded_3bet / faced_3bet
            conf = min(faced_3bet / 15, 1.0)

            if fold_freq >= PATTERN_THRESHOLD:
                patterns.append(DetectedPattern(
                    player       = player,
                    pattern_type = PatternType.ALWAYS_FOLD_3BET,
                    frequency    = fold_freq,
                    sample_count = faced_3bet,
                    description  = f"Fold au 3-bet {fold_freq:.0%} du temps",
                    exploitation = "3-bet light en position — il fold souvent.",
                    confidence   = conf,
                ))
            elif fold_freq <= (1 - PATTERN_THRESHOLD):
                call_freq = called_3bet / faced_3bet
                patterns.append(DetectedPattern(
                    player       = player,
                    pattern_type = PatternType.ALWAYS_CALL_3BET,
                    frequency    = call_freq,
                    sample_count = faced_3bet,
                    description  = f"Call le 3-bet {call_freq:.0%} du temps",
                    exploitation = "3-bet uniquement pour valeur, pas de bluff.",
                    confidence   = conf,
                ))
        return patterns

    # ── Sizing tells ──────────────────────────────────────────────────────

    def _detect_sizing_tells(
        self, player: str, seqs: list[ActionSequence]
    ) -> list[DetectedPattern]:
        """
        Détecte si la taille de mise corrèle avec la force de la main.
        Grosse mise = bluff ou value ? Petite mise = value ou bluff ?
        """
        patterns = []
        large_bets_results = []   # (won: bool) pour les grosses mises
        small_bets_results = []   # (won: bool) pour les petites mises

        LARGE_RATIO = 0.80   # ≥ 80% du pot = grosse mise
        SMALL_RATIO = 0.35   # ≤ 35% du pot = petite mise

        for seq in seqs:
            for act in seq.actions:
                if act["action"] not in ("BET", "RAISE"):
                    continue
                r = act.get("pot_ratio", 0.0)
                won = seq.result > 0
                if r >= LARGE_RATIO:
                    large_bets_results.append(won)
                elif r <= SMALL_RATIO:
                    small_bets_results.append(won)

        # Grosse mise = bluff si perd souvent au showdown après
        if len(large_bets_results) >= PATTERN_MIN_SAMPLES:
            lose_rate = 1 - (sum(large_bets_results) / len(large_bets_results))
            if lose_rate >= PATTERN_THRESHOLD:
                patterns.append(DetectedPattern(
                    player       = player,
                    pattern_type = PatternType.OVERBET_BLUFF,
                    frequency    = lose_rate,
                    sample_count = len(large_bets_results),
                    description  = f"Overbet (>80% pot) = bluff probable ({lose_rate:.0%} perdu)",
                    exploitation = "Call (ou raise) ses overbets avec range médiane.",
                    confidence   = min(len(large_bets_results) / 15, 1.0),
                ))

        # Petite mise = value si gagne souvent
        if len(small_bets_results) >= PATTERN_MIN_SAMPLES:
            win_rate = sum(small_bets_results) / len(small_bets_results)
            if win_rate >= PATTERN_THRESHOLD:
                patterns.append(DetectedPattern(
                    player       = player,
                    pattern_type = PatternType.SMALL_BET_VALUE,
                    frequency    = win_rate,
                    sample_count = len(small_bets_results),
                    description  = f"Petite mise (<35% pot) = value ({win_rate:.0%} gagné)",
                    exploitation = "Fold face aux petites mises sans top pair.",
                    confidence   = min(len(small_bets_results) / 15, 1.0),
                ))
        return patterns

    # ── Triple barrel ─────────────────────────────────────────────────────

    def _detect_barrel_patterns(
        self, player: str, seqs: list[ActionSequence]
    ) -> list[DetectedPattern]:
        patterns = []
        multi_street_bets = 0
        triple_barrels    = 0

        for seq in seqs:
            streets_bet = set()
            for a in seq.actions:
                if a["action"] in ("BET", "RAISE"):
                    streets_bet.add(a["street"])

            if len(streets_bet) >= 2:
                multi_street_bets += 1
            if "flop" in streets_bet and "turn" in streets_bet and "river" in streets_bet:
                triple_barrels += 1

        total = len(seqs)
        if multi_street_bets >= PATTERN_MIN_SAMPLES:
            barrel_freq = triple_barrels / max(multi_street_bets, 1)
            if barrel_freq >= PATTERN_THRESHOLD:
                patterns.append(DetectedPattern(
                    player       = player,
                    pattern_type = PatternType.RIVER_BARREL,
                    frequency    = barrel_freq,
                    sample_count = multi_street_bets,
                    description  = f"Triple barrel fréquent ({barrel_freq:.0%} des mains multi-streets)",
                    exploitation = "Call-down avec top pair ou mieux sur toutes les streets.",
                    confidence   = min(multi_street_bets / 10, 1.0),
                ))
        return patterns

    # ── Check-raise ───────────────────────────────────────────────────────

    def _detect_check_raise(
        self, player: str, seqs: list[ActionSequence]
    ) -> list[DetectedPattern]:
        patterns = []
        check_raises   = 0
        bluff_cr       = 0   # check-raise sans value (perdu)

        for seq in seqs:
            actions = seq.actions
            for i, a in enumerate(actions[:-1]):
                next_a = actions[i+1]
                if (a["action"] == "CHECK" and
                        next_a["action"] == "RAISE" and
                        a["street"] == next_a["street"]):
                    check_raises += 1
                    if seq.result <= 0:
                        bluff_cr += 1

        if check_raises >= PATTERN_MIN_SAMPLES:
            bluff_freq = bluff_cr / check_raises
            if bluff_freq >= 0.50:
                patterns.append(DetectedPattern(
                    player       = player,
                    pattern_type = PatternType.CHECK_RAISE_BLUFF,
                    frequency    = bluff_freq,
                    sample_count = check_raises,
                    description  = f"Check-raise = bluff fréquent ({bluff_freq:.0%} sans value)",
                    exploitation = "Call (ou re-raise) ses check-raises avec TP+.",
                    confidence   = min(check_raises / 10, 1.0),
                ))
        return patterns

    # ── Showdown bluffs ───────────────────────────────────────────────────

    def _detect_showdown_bluffs(
        self, player: str, seqs: list[ActionSequence]
    ) -> list[DetectedPattern]:
        patterns = []
        showdowns     = [s for s in seqs if s.showed_cards]
        bluffs_shown  = [s for s in showdowns if s.was_bluff]

        if len(showdowns) >= 3 and len(bluffs_shown) >= 2:
            bluff_freq = len(bluffs_shown) / len(showdowns)
            patterns.append(DetectedPattern(
                player       = player,
                pattern_type = PatternType.SHOWDOWN_BLUFF,
                frequency    = bluff_freq,
                sample_count = len(showdowns),
                description  = f"A montré des bluffs au showdown ({len(bluffs_shown)}/{len(showdowns)})",
                exploitation = "Call plus large sur la rivière — il bluff réellement.",
                confidence   = min(len(showdowns) / 10, 1.0),
            ))
        return patterns


# ---------------------------------------------------------------------------
# Analyseur Claude (analyse approfondie)
# ---------------------------------------------------------------------------

class ClaudePatternAnalyser:
    """
    Utilise Claude pour une analyse approfondie des patterns adverses.

    Contrairement au BluffDetector (heuristiques locales), Claude peut :
    - Identifier des patterns complexes multi-variables
    - Contextualiser selon le board, la position, le stack
    - Détecter le tilt et les états émotionnels
    - Suggérer des adaptations stratégiques précises
    """

    ANALYSIS_SYSTEM_PROMPT = """Tu es un expert en analyse comportementale au poker Texas Hold'em.

On te fournit des séquences d'actions d'un adversaire sur plusieurs mains.
Tu dois identifier :
1. Ses patterns de bluff (quand bluffe-t-il ? Sur quelles textures ? Avec quel sizing ?)
2. Ses tendances exploitables (fold trop souvent à tel endroit, call trop souvent ailleurs)
3. Ses tells de sizing (grosse mise = bluff ou value ?)
4. Son profil psychologique (agressif, passif, tilté, solide)
5. Le conseil d'exploitation principal pour la prochaine main

Réponds UNIQUEMENT en JSON strict avec cette structure :
{
  "bluff_patterns": ["description courte", ...],
  "exploitable_tendencies": ["description courte", ...],
  "sizing_tells": "description ou ''",
  "psychological_profile": "description 1 phrase",
  "top_exploitation": "conseil d'action principal en 1 phrase",
  "confidence": 0.0,
  "hand_count_used": 0
}

Sois concis et actionnable. Pas de texte hors du JSON."""

    def __init__(self, api_key: Optional[str] = None):
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
            self._available = True
        except ImportError:
            self._available = False
            log.warning("anthropic non installé — ClaudePatternAnalyser désactivé.")

    def analyse_player(
        self,
        player:    str,
        sequences: list[ActionSequence],
        profile:   Optional[object] = None,   # OpponentProfile
    ) -> Optional[dict]:
        """
        Envoie les séquences à Claude pour analyse.
        Retourne un dict avec les patterns détectés, ou None si erreur.
        """
        if not self._available or len(sequences) < PATTERN_MIN_SAMPLES:
            return None

        # Construire le payload des séquences
        seq_summaries = []
        for seq in sequences[-30:]:   # max 30 mains pour limiter les tokens
            summary = {
                "board":   seq.board,
                "actions": [
                    {"s": a["street"][:2], "a": a["action"], "r": a.get("pot_ratio", 0)}
                    for a in seq.actions
                ],
                "result":  round(seq.result, 1),
            }
            if seq.showed_cards:
                summary["showed"] = seq.showed_cards
            seq_summaries.append(summary)

        profile_ctx = ""
        if profile:
            profile_ctx = (
                f"Profil existant : VPIP={profile.vpip:.0f}% "
                f"PFR={profile.pfr:.0f}% AF={profile.af:.1f} "
                f"Tendance={profile.tendency}"
            )

        user_msg = (
            f"Adversaire : {player}\n"
            f"{profile_ctx}\n"
            f"Séquences ({len(seq_summaries)} mains) :\n"
            + json.dumps(seq_summaries, ensure_ascii=False)
        )

        try:
            import anthropic
            response = self._client.messages.create(
                model      = "claude-sonnet-4-20250514",
                max_tokens = 600,
                system     = self.ANALYSIS_SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()
            # Nettoyer les backticks éventuels
            if raw.startswith("```"):
                raw = "\n".join(
                    l for l in raw.splitlines()
                    if not l.strip().startswith("```")
                ).strip()
            return json.loads(raw)

        except Exception as e:
            log.error(f"ClaudePatternAnalyser erreur pour {player} : {e}")
            return None

    def batch_analyse(
        self,
        players:      list[str],
        accumulator:  PatternAccumulator,
        profil_builder = None,
    ) -> dict[str, dict]:
        """
        Analyse plusieurs adversaires en batch (entre les mains).
        Retourne {player_name: analyse_dict}.
        """
        results = {}
        for player in players:
            seqs    = accumulator.get_sequences(player, last_n=30)
            profile = profil_builder.get_profile(player) if profil_builder else None
            result  = self.analyse_player(player, seqs, profile)
            if result:
                results[player] = result
                log.info(f"Analyse Claude terminée pour {player}")
        return results


# ---------------------------------------------------------------------------
# Moteur principal
# ---------------------------------------------------------------------------

class PatternEngine:
    """
    Orchestre la détection de patterns en temps réel et l'analyse Claude.

    Usage dans main.py :
        engine = PatternEngine()

        # À chaque action adverse :
        engine.record_action("Villain42", "RAISE", 80, pot=120, stage="flop")

        # En fin de main :
        engine.end_hand("Villain42", result=-80, stage="river")

        # Obtenir les patterns (locaux, instantanés) :
        patterns = engine.get_patterns("Villain42")

        # Analyse Claude approfondie (entre les mains) :
        analysis = engine.analyse_with_claude("Villain42")

        # Contexte pour le prompt Claude :
        ctx = engine.get_prompt_context(["Villain42", "Player3"])
    """

    def __init__(self, api_key: Optional[str] = None, profil_builder=None):
        self.accumulator    = PatternAccumulator()
        self.detector       = BluffDetector(self.accumulator)
        self.claude_analyser= ClaudePatternAnalyser(api_key=api_key)
        self.profil_builder = profil_builder

        # Cache des analyses Claude (valide 30 minutes)
        self._claude_cache: dict[str, dict]  = {}
        self._cache_ts:     dict[str, float] = {}
        self._cache_ttl     = 1800   # 30 minutes

        # Cache des patterns locaux (valide 2 minutes)
        self._pattern_cache: dict[str, list] = {}
        self._pattern_ts:    dict[str, float] = {}
        self._pattern_ttl    = 120

        # Patterns détectés persistants
        self._all_patterns: dict[str, list[DetectedPattern]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Enregistrement temps réel
    # ------------------------------------------------------------------

    def start_hand(self, players: list[str], hand_id: str = "", board: list[str] = None) -> None:
        for player in players:
            self.accumulator.start_hand(player, hand_id, board or [])

    def record_action(
        self,
        player: str,
        action: str,
        amount: float = 0.0,
        pot:    float = 0.0,
        stage:  str   = "flop",
    ) -> None:
        self.accumulator.record_action(player, action, amount, pot, stage)

    def end_hand(
        self,
        player:    str,
        result:    float = 0.0,
        showed:    list  = None,
        had_value: bool  = False,
        stage:     str   = "river",
    ) -> None:
        seq = self.accumulator.end_hand(player, result, showed, had_value, stage)
        if seq:
            # Invalider le cache des patterns pour ce joueur
            self._pattern_cache.pop(player, None)

    # ------------------------------------------------------------------
    # Récupération des patterns
    # ------------------------------------------------------------------

    def get_patterns(self, player: str) -> list[DetectedPattern]:
        """Retourne les patterns locaux (heuristiques, instantané)."""
        now = time.time()
        if (player in self._pattern_cache and
                now - self._pattern_ts.get(player, 0) < self._pattern_ttl):
            return self._pattern_cache[player]

        patterns = self.detector.analyse_player(player)
        self._pattern_cache[player] = patterns
        self._pattern_ts[player]    = now
        return patterns

    def analyse_with_claude(
        self,
        player:    str,
        force:     bool = False,
    ) -> Optional[dict]:
        """
        Lance l'analyse Claude pour un adversaire.
        Utilise le cache si disponible (TTL 30 min).
        """
        now = time.time()
        if (not force and
                player in self._claude_cache and
                now - self._cache_ts.get(player, 0) < self._cache_ttl):
            return self._claude_cache[player]

        seqs    = self.accumulator.get_sequences(player)
        profile = self.profil_builder.get_profile(player) if self.profil_builder else None
        result  = self.claude_analyser.analyse_player(player, seqs, profile)

        if result:
            self._claude_cache[player] = result
            self._cache_ts[player]     = now

        return result

    def get_tendency_report(self, player: str) -> TendencyReport:
        """Rapport complet pour un adversaire (patterns + analyse Claude si dispo)."""
        patterns       = self.get_patterns(player)
        claude_result  = self._claude_cache.get(player, {})

        top_exploitation = ""
        if patterns:
            top = next((p for p in patterns if p.is_reliable), None)
            if top:
                top_exploitation = top.exploitation

        if claude_result:
            top_exploitation = claude_result.get("top_exploitation", top_exploitation)

        return TendencyReport(
            player           = player,
            patterns         = patterns,
            claude_analysis  = claude_result.get("psychological_profile", ""),
            top_exploitation = top_exploitation,
        )

    # ------------------------------------------------------------------
    # Contexte pour le prompt Claude (poker conseiller)
    # ------------------------------------------------------------------

    def get_prompt_context(self, players: list[str]) -> str:
        """
        Génère le bloc de contexte patterns à injecter dans build_user_prompt().
        Format compact pour ne pas saturer le contexte.
        """
        lines = []
        for player in players:
            patterns = self.get_patterns(player)
            reliable = [p for p in patterns if p.is_reliable]
            if not reliable:
                continue

            lines.append(f"\n{player} :")
            for p in reliable[:3]:   # max 3 patterns par joueur
                lines.append(f"  [{p.pattern_type.value}] {p.description}")
                lines.append(f"  → Exploitation : {p.exploitation}")

            # Ajouter l'analyse Claude si disponible
            claude = self._claude_cache.get(player, {})
            if claude.get("top_exploitation"):
                lines.append(f"  [IA] {claude['top_exploitation']}")

        return "\n".join(lines) if lines else ""

    def get_structured_profiles(self, players: list[str]) -> dict:
        """
        Retourne les profils enrichis (VPIP/PFR/AF + patterns) pour claude_client.
        """
        profiles = {}
        for player in players:
            base = {}
            if self.profil_builder:
                p = self.profil_builder.get_profile(player)
                base = p.to_claude_context()

            patterns  = self.get_patterns(player)
            reliable  = [p.to_dict() for p in patterns if p.is_reliable]
            claude    = self._claude_cache.get(player, {})

            profiles[player] = {
                **base,
                "detected_patterns":   reliable,
                "bluff_patterns":      claude.get("bluff_patterns", []),
                "exploitable_tendencies": claude.get("exploitable_tendencies", []),
                "sizing_tell":         claude.get("sizing_tell", ""),
                "top_exploitation":    claude.get("top_exploitation", base.get("advice", "")),
            }
        return profiles


# ---------------------------------------------------------------------------
# Intégration dans capture.py / main.py
# ---------------------------------------------------------------------------

def update_engine_from_game_state(
    engine:     PatternEngine,
    game_state,                  # GameState de capture.py
    hero_name:  str = "",
) -> None:
    """
    Met à jour le PatternEngine depuis un GameState (action_detector intégré).
    À appeler dans main.py après chaque detect().
    """
    for action in getattr(game_state, "opponent_actions", []):
        seat_name = f"Seat_{action.seat}"
        act_str   = action.action.value if hasattr(action.action, "value") else str(action.action)
        engine.record_action(
            player = seat_name,
            action = act_str,
            amount = action.amount,
            pot    = game_state.pot,
            stage  = game_state.stage,
        )


# ---------------------------------------------------------------------------
# CLI de test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Détecteur de patterns de bluff et tendances adverses"
    )
    parser.add_argument("--analyse",  type=str, metavar="JOUEUR",
                        help="Analyser un joueur depuis la base de profils")
    parser.add_argument("--hh",       type=str, metavar="DOSSIER",
                        help="Construire les patterns depuis un dossier HH")
    parser.add_argument("--hero",     type=str, default="",
                        help="Pseudo du héros")
    parser.add_argument("--demo",     action="store_true",
                        help="Démonstration avec données simulées")
    args = parser.parse_args()

    engine = PatternEngine()

    if args.demo:
        print("\n=== DÉMO PatternEngine ===\n")

        # Simuler des séquences pour "Villain"
        import random
        villain = "Villain42"

        for i in range(25):
            engine.start_hand([villain], f"hand_{i}", ["Ah", "Kd", "3c"])

            # Villain relance souvent préflop
            if random.random() < 0.70:
                engine.record_action(villain, "RAISE", 6, pot=3, stage="preflop")
                # CBet systématique au flop
                if random.random() < 0.80:
                    engine.record_action(villain, "BET", 15, pot=18, stage="flop")
                    # Overbet river = bluff (perd souvent)
                    if random.random() < 0.60:
                        engine.record_action(villain, "BET", 80, pot=50, stage="river")
                        engine.end_hand(villain, result=random.choice([-80, -80, 30]), stage="river")
                    else:
                        engine.end_hand(villain, result=random.choice([20, -15]), stage="flop")
                else:
                    engine.end_hand(villain, result=random.choice([-6, 8]), stage="preflop")
            else:
                engine.record_action(villain, "CALL", 3, pot=3, stage="preflop")
                engine.end_hand(villain, result=random.choice([-3, 12, -3]), stage="flop")

        # Afficher les patterns détectés
        patterns = engine.get_patterns(villain)
        print(f"Patterns détectés pour {villain} ({engine.accumulator.total_hands(villain)} mains) :\n")
        if patterns:
            for p in patterns:
                status = "✓ FIABLE" if p.is_reliable else "~ Estimé"
                print(f"  [{status}] {p.to_hud_line()}")
                print(f"          → {p.exploitation}\n")
        else:
            print("  Aucun pattern détecté (manque de données).")

        # Contexte prompt
        ctx = engine.get_prompt_context([villain])
        if ctx:
            print(f"\nContexte prompt Claude :\n{ctx}")

    elif args.hh:
        print(f"\nConstruction patterns depuis {args.hh}…")
        from hh_parser import HandHistoryParser
        from profil_builder import ProfilBuilder, ProfileDatabase

        db      = ProfileDatabase()
        builder = ProfilBuilder(db)
        engine  = PatternEngine(profil_builder=builder)

        hh_parser = HandHistoryParser(hero_name=args.hero)
        folder    = Path(args.hh)

        for path in sorted(folder.rglob("*.txt")):
            hands = hh_parser.parse_file(path)
            for hand in hands:
                players = [p.name for p in hand.players if p.name != args.hero]
                engine.start_hand(players, hand.hand_id, hand.board)
                for action in hand.actions:
                    if action.player != args.hero:
                        engine.record_action(
                            action.player, action.action.upper(),
                            action.amount, 0, action.street
                        )
                for p in hand.players:
                    if p.name != args.hero:
                        engine.end_hand(p.name, p.result)

        # Afficher le top des patterns
        all_players = engine.accumulator.get_all_players()
        print(f"\n{len(all_players)} adversaires analysés :\n")
        for player in all_players[:10]:
            patterns = engine.get_patterns(player)
            reliable = [p for p in patterns if p.is_reliable]
            if reliable:
                print(f"  {player} ({engine.accumulator.total_hands(player)} mains)")
                for p in reliable[:2]:
                    print(f"    • {p.description}")
                    print(f"      → {p.exploitation}")
                print()
