"""
claude_client.py — Intégration Claude API pour conseils poker temps réel
Dépendances : anthropic
Installation : pip install anthropic

Ce module envoie l'état du jeu (GameState + EquityResult) à Claude
et retourne une recommandation stratégique structurée (résumé + JSON).
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL         = "claude-sonnet-4-20250514"
MAX_TOKENS    = 1024
RETRY_LIMIT   = 3
RETRY_DELAY   = 2.0    # secondes entre les tentatives

# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------

@dataclass
class PokerAdvice:
    """Conseil retourné par Claude pour une situation donnée."""
    summary:              str   = ""      # Texte lisible court
    recommended_action:   str   = ""      # FOLD / CALL / RAISE / etc.
    recommended_sizing:   str   = ""      # Ex. "60% du pot"
    win_probability:      float = 0.0
    hands_beating_us:     int   = 0
    hand_class:           str   = ""
    position:             str   = ""      # BTN / CO / HJ / MP / UTG / SB / BB
    position_advantage:   int   = 99      # 1=meilleur, plus grand=pire
    spr:                  float = 0.0     # Stack-to-Pot Ratio
    spr_label:            str   = ""      # ex. "court (TP+ requis)"
    spr_comment:          str   = ""      # conseil SPR
    pot_odds:             float = 0.0     # % pot à payer pour caller
    mdf:                  float = 0.0     # Minimum Defense Frequency
    action_explanation:   str   = ""
    raw_json:             dict  = field(default_factory=dict)
    latency_ms:           float = 0.0
    error:                str   = ""

    def to_overlay_text(self) -> str:
        """Texte compact pour l'overlay HUD."""
        if self.error:
            return f"⚠ Erreur : {self.error}"
        action_icon = {
            "FOLD":   "✗ FOLD",
            "CHECK":  "— CHECK",
            "CALL":   "✓ CALL",
            "BET":    "↑ BET",
            "RAISE":  "↑↑ RAISE",
            "ALL-IN": "⚡ ALL-IN",
        }.get(self.recommended_action, self.recommended_action)

        pos_str = f"  [{self.position}]" if self.position else ""
        lines = [
            f"{action_icon}{pos_str}",
            f"Equity : {self.win_probability:.0%}",
            f"Main   : {self.hand_class}",
        ]
        if self.recommended_sizing:
            lines.append(f"Sizing : {self.recommended_sizing}")
        if self.action_explanation:
            lines.append(f"→ {self.action_explanation[:120]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt système (le système prompt de votre document)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Tu es un agent poker professionnel temps réel spécialisé en Texas Hold'em No-Limit.

À chaque appel, tu reçois un état de jeu JSON. Tu dois retourner UNIQUEMENT un objet JSON valide (aucun texte avant ou après) avec cette structure exacte :

{
  "stage": "",
  "player_hand": ["", ""],
  "board": [],
  "num_opponents": 0,
  "hands_that_beat_us": 0,
  "estimated_win_probability": 0.0,
  "recommended_action": "",
  "recommended_bet_size": "",
  "hand_class": "",
  "action_explanations": "",
  "summary": "",
  "position": "",
  "position_advantage": 0,
  "position_comment": "",
  "spr": 0.0,
  "spr_label": "",
  "spr_comment": "",
  "mdf": 0.0,
  "pot_odds": 0.0
}

Règles :
- recommended_action : FOLD | CHECK | CALL | BET | RAISE | ALL-IN
- recommended_bet_size : sizing en % du pot ou montant si fourni (vide si FOLD/CHECK/CALL)
- summary : 1-2 phrases max, clair et stratégique
- action_explanations : justification courte basée sur equity, texture du board, profil adverse
- Ne jamais modifier la structure JSON
- Ne jamais ajouter de texte hors du JSON
- Intègre les données d'équité fournies pour affiner ton analyse exploitante
- Intègre TOUJOURS la position dans tes recommandations :
  * BTN/CO/HJ (position tardive) : range plus large, steal/squeeze possibles
  * UTG/UTG+1/UTG+2 (position précoce) : range premium uniquement
  * SB/BB (blindes) : pot odds et désavantage post-flop à considérer
- Remplis les champs position, position_advantage (1=BTN meilleur, 11=BB pire)
  et position_comment avec un conseil spécifique à la position
- Intègre le SPR dans ta recommandation :
  * SPR ≤ 1 : shove/fold uniquement, pas de call partiel
  * SPR 1–4 : TP+ minimum, évite les bluffs coûteux
  * SPR 4–13 : jeu post-flop équilibré, semi-bluff rentable
  * SPR > 13 : spéculation et implied odds maximaux
- Remplis spr_comment avec une phrase stratégique basée sur le SPR
- Si des patterns de bluff ou tendances sont fournis, adapte IMMÉDIATEMENT ta recommandation :
  * always_cbet → float le flop et steal au turn
  * always_fold_3bet → 3-bet light en position
  * overbet_bluff → call ses overbets plus large
  * small_bet_value → fold sans TP+ face à petite mise
  * check_raise_bluff → call ou re-raise ses check-raises
- Le champ top_exploitation doit refléter le pattern le plus exploitable de la main"""


# ---------------------------------------------------------------------------
# Construction du prompt utilisateur
# ---------------------------------------------------------------------------

def build_user_prompt(
    game_state_dict: dict,
    equity_result_dict: dict,
    opponent_profiles: Optional[dict] = None,
) -> str:
    """
    Assemble le prompt utilisateur à partir de l'état du jeu et du résultat d'équité.
    Inclut un résumé de position pour guider Claude directement.
    """
    payload = {
        "game_state":     game_state_dict,
        "equity_result":  equity_result_dict,
    }
    if opponent_profiles:
        payload["opponent_profiles"] = opponent_profiles

    # Résumé de position en tête de prompt pour guider Claude
    position    = game_state_dict.get("position", "")
    pos_adv     = game_state_dict.get("position_advantage", 99)
    is_dealer   = game_state_dict.get("is_dealer", False)
    num_players = game_state_dict.get("num_players", 0)
    stage       = game_state_dict.get("stage", "preflop")

    pos_context = ""
    if position:
        pos_quality = (
            "LATE (avantage maximal)"   if pos_adv <= 3 else
            "MIDDLE (avantage moyen)"   if pos_adv <= 6 else
            "EARLY (désavantage)"       if pos_adv <= 9 else
            "BLIND (désavantage post-flop)"
        )
        dealer_str  = " [DEALER/BTN]" if is_dealer else ""
        pos_context = (
            f"\nCONTEXTE POSITION : {position}{dealer_str} — {pos_quality} "
            f"| {num_players} joueurs | {stage.upper()}"
            f"\nAdapte ta recommandation à cette position.\n"
        )

    # Résumé des actions adverses récentes
    opp_actions = game_state_dict.get("opponent_actions", [])
    action_context = ""
    if opp_actions:
        lines = []
        for a in opp_actions[-6:]:   # 6 dernières actions max
            amt = f" {a['amount']:.0f}$" if a.get("amount", 0) > 0 else ""
            lines.append(f"Siège {a['seat']}: {a['action']}{amt}")
        action_context = (
            f"\nACTIONS ADVERSES RÉCENTES : " + " | ".join(lines) +
            "\nAdapte ta recommandation en tenant compte de ces actions.\n"
        )

    # Résumé profil d'agressivité
    agg_profile = game_state_dict.get("aggression_profile", {})
    profile_context = ""
    if agg_profile:
        plines = []
        for seat, p in list(agg_profile.items())[:4]:
            plines.append(
                f"S{seat}:AF={p.get('aggression_factor',0):.1f} "
                f"VPIP≈{p.get('vpip_est',0):.0f}% PFR≈{p.get('pfr_est',0):.0f}%"
            )
        profile_context = "\nPROFILS ADVERSES : " + " | ".join(plines) + "\n"

    # Contexte SPR
    spr_val     = equity_result_dict.get("spr", 0.0)
    spr_lbl     = equity_result_dict.get("spr_label", "")
    spr_comment = equity_result_dict.get("spr_comment", "")
    pot_odds    = equity_result_dict.get("pot_odds", 0.0)
    mdf         = equity_result_dict.get("mdf", 0.0)
    spr_context = ""
    if spr_val > 0:
        spr_context = (
            f"\nSPR : {spr_val:.1f} — {spr_lbl}"
            + (f" | Pot odds : {pot_odds:.0%}" if pot_odds > 0 else "")
            + (f" | MDF : {mdf:.0%}" if mdf > 0 else "")
            + (f"\n→ {spr_comment}" if spr_comment else "")
            + "\n"
        )

    # Contexte patterns de bluff
    patterns_context = ""
    if opponent_profiles:
        plines = []
        for pname, prof in list(opponent_profiles.items())[:4]:
            det_patterns = prof.get("detected_patterns", [])
            bluff_pats   = prof.get("bluff_patterns", [])
            top_exploit  = prof.get("top_exploitation", "")
            if det_patterns or bluff_pats:
                plines.append(f"\n{pname} :")
                for p in det_patterns[:2]:
                    plines.append(f"  [{p.get('pattern','')}] {p.get('description','')} ({p.get('frequency',0):.0%})")
                    plines.append(f"  → {p.get('exploitation','')}")
                for bp in bluff_pats[:2]:
                    plines.append(f"  [bluff] {bp}")
                if top_exploit:
                    plines.append(f"  [ACTION] {top_exploit}")
        if plines:
            patterns_context = "\nPATTERNS ADVERSES DÉTECTÉS :" + "\n".join(plines) + "\n"

    return (
        pos_context
        + spr_context
        + patterns_context
        + action_context
        + profile_context
        + "\nVoici l'état actuel de la main. Analyse et retourne ta recommandation en JSON :\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


# ---------------------------------------------------------------------------
# Client Claude
# ---------------------------------------------------------------------------

class ClaudePokerClient:
    """
    Envoie les situations de jeu à Claude et parse les recommandations.

    Usage :
        client = ClaudePokerClient(api_key="sk-ant-...")
        advice = client.get_advice(state.to_dict(), result.to_dict())
        print(advice.to_overlay_text())
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        api_key : clé Anthropic. Si None, utilise la variable d'environnement
                  ANTHROPIC_API_KEY automatiquement.
        """
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self._history: list[dict]   = []    # historique de la main en cours
        self._last_advice: Optional[PokerAdvice] = None

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def get_advice(
        self,
        game_state:        dict,
        equity_result:     dict,
        opponent_profiles: Optional[dict] = None,
        use_history:       bool = True,
    ) -> PokerAdvice:
        """
        Envoie la situation à Claude et retourne un PokerAdvice.

        Paramètres :
          game_state        : dict issu de GameState.to_dict()
          equity_result     : dict issu de EquityResult.to_dict()
          opponent_profiles : stats HUD optionnelles {player_id: {vpip, pfr, af}}
          use_history       : inclure l'historique de la main pour le contexte
        """
        user_msg = build_user_prompt(game_state, equity_result, opponent_profiles)

        if use_history:
            self._history.append({"role": "user", "content": user_msg})
            messages = self._history.copy()
        else:
            messages = [{"role": "user", "content": user_msg}]

        advice = self._call_with_retry(messages)

        if use_history and advice.raw_json:
            # Ajouter la réponse de Claude à l'historique
            self._history.append({
                "role":    "assistant",
                "content": json.dumps(advice.raw_json, ensure_ascii=False),
            })

        self._last_advice = advice
        return advice

    def new_hand(self) -> None:
        """Réinitialise le contexte pour une nouvelle main."""
        self._history    = []
        self._last_advice = None
        log.info("Contexte réinitialisé — nouvelle main.")

    @property
    def last_advice(self) -> Optional[PokerAdvice]:
        return self._last_advice

    # ------------------------------------------------------------------
    # Appel API avec retry
    # ------------------------------------------------------------------

    def _call_with_retry(self, messages: list[dict]) -> PokerAdvice:
        """Appelle l'API Claude avec jusqu'à RETRY_LIMIT tentatives."""
        last_error = ""

        for attempt in range(1, RETRY_LIMIT + 1):
            t0 = time.monotonic()
            try:
                response = self._client.messages.create(
                    model      = MODEL,
                    max_tokens = MAX_TOKENS,
                    system     = SYSTEM_PROMPT,
                    messages   = messages,
                )
                latency_ms = (time.monotonic() - t0) * 1000
                log.info(f"Claude API répondu en {latency_ms:.0f}ms (tentative {attempt})")

                raw_text = self._extract_text(response)
                return self._parse_response(raw_text, latency_ms)

            except anthropic.RateLimitError as e:
                last_error = f"Rate limit : {e}"
                log.warning(f"Rate limit atteint, attente {RETRY_DELAY}s…")
                time.sleep(RETRY_DELAY * attempt)

            except anthropic.APIConnectionError as e:
                last_error = f"Connexion : {e}"
                log.error(f"Erreur réseau tentative {attempt} : {e}")
                time.sleep(RETRY_DELAY)

            except anthropic.APIStatusError as e:
                last_error = f"API status {e.status_code} : {e.message}"
                log.error(last_error)
                if e.status_code in (400, 401, 403):
                    break   # erreurs non retriables
                time.sleep(RETRY_DELAY)

            except Exception as e:
                last_error = str(e)
                log.error(f"Erreur inattendue tentative {attempt} : {e}")
                time.sleep(RETRY_DELAY)

        return PokerAdvice(error=last_error)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(response) -> str:
        """Extrait le texte brut de la réponse Anthropic."""
        for block in response.content:
            if block.type == "text":
                return block.text.strip()
        return ""

    @staticmethod
    def _parse_response(raw_text: str, latency_ms: float) -> PokerAdvice:
        """
        Parse le JSON retourné par Claude en PokerAdvice.
        Gère les cas où Claude entoure le JSON de backticks.
        """
        # Nettoyage des balises markdown éventuelles
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text  = "\n".join(
                l for l in lines
                if not l.strip().startswith("```")
            ).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            log.error(f"JSON invalide : {e}\nTexte brut : {raw_text[:300]}")
            return PokerAdvice(
                summary = raw_text[:200],
                error   = f"JSON parse error : {e}",
                latency_ms = latency_ms,
            )

        advice = PokerAdvice(
            summary            = data.get("summary", ""),
            recommended_action = data.get("recommended_action", ""),
            recommended_sizing = data.get("recommended_bet_size", ""),
            win_probability    = float(data.get("estimated_win_probability", 0.0)),
            hands_beating_us   = int(data.get("hands_that_beat_us", 0)),
            hand_class         = data.get("hand_class", ""),
            position           = data.get("position", ""),
            position_advantage = int(data.get("position_advantage", 99)),
            spr                = float(data.get("spr", 0.0)),
            spr_label          = data.get("spr_label", ""),
            spr_comment        = data.get("spr_comment", ""),
            pot_odds           = float(data.get("pot_odds", 0.0)),
            mdf                = float(data.get("mdf", 0.0)),
            action_explanation = data.get("action_explanations", ""),
            raw_json           = data,
            latency_ms         = latency_ms,
        )
        log.info(
            f"Conseil reçu : {advice.recommended_action} | "
            f"equity={advice.win_probability:.1%} | {advice.summary[:60]}"
        )
        return advice


# ---------------------------------------------------------------------------
# Cache simple (évite les appels redondants si l'état n'a pas changé)
# ---------------------------------------------------------------------------

class CachedClaudeClient(ClaudePokerClient):
    """
    Sous-classe qui met en cache la dernière réponse.
    Si le game_state est identique au précédent, retourne le cache
    sans appeler l'API (utile si la boucle de capture tourne vite).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_key:   Optional[str]        = None
        self._cache_value: Optional[PokerAdvice] = None

    def get_advice(self, game_state: dict, equity_result: dict,
                   opponent_profiles: Optional[dict] = None,
                   use_history: bool = True) -> PokerAdvice:

        key = json.dumps(
            {"gs": game_state, "eq": equity_result},
            sort_keys=True, ensure_ascii=False
        )

        if key == self._cache_key and self._cache_value is not None:
            log.debug("Cache hit — pas d'appel API.")
            return self._cache_value

        advice = super().get_advice(
            game_state, equity_result, opponent_profiles, use_history
        )
        self._cache_key   = key
        self._cache_value = advice
        return advice

    def new_hand(self) -> None:
        super().new_hand()
        self._cache_key   = None
        self._cache_value = None


# ---------------------------------------------------------------------------
# Intégration directe GameState + EquityResult
# ---------------------------------------------------------------------------

def analyse_situation(
    game_state,        # GameState (capture.py)
    equity_result,     # EquityResult (engine.py)
    client: Optional[ClaudePokerClient] = None,
    opponent_profiles: Optional[dict] = None,
) -> PokerAdvice:
    """
    Raccourci tout-en-un :
    Accepte directement les objets Python de capture.py et engine.py.

    Usage :
        from capture import PokerExtractor
        from engine  import PokerEngine
        from claude_client import analyse_situation

        extractor = PokerExtractor()
        engine    = PokerEngine()
        client    = CachedClaudeClient()

        for state in extractor.run_capture_loop():
            result = engine.analyse_from_state(state)
            advice = analyse_situation(state, result, client)
            print(advice.to_overlay_text())
    """
    if client is None:
        client = CachedClaudeClient()

    return client.get_advice(
        game_state        = game_state.to_dict(),
        equity_result     = equity_result.to_dict(),
        opponent_profiles = opponent_profiles,
    )


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Simulation d'un état de jeu (remplace par de vraies données en prod)
    mock_game_state = {
        "player_cards": ["Ks", "7h"],
        "board_cards":  ["Kd", "7d", "2c"],
        "pot":          120.0,
        "player_stack": 980.0,
        "current_bet":  40.0,
        "num_players":  4,
        "stage":        "flop",
        "timestamp":    time.time(),
    }

    mock_equity = {
        "win_probability":    0.87,
        "tie_probability":    0.02,
        "lose_probability":   0.11,
        "hand_strength":      1820,
        "hand_class":         "Deux Paires",
        "num_combos_total":   1200,
        "num_combos_beating": 132,
        "outs":               4,
        "out_cards":          ["Ks", "7s", "Kc", "7c"],
        "recommended_action": "RAISE",
        "recommended_sizing": "420$ (2.5× le pot)",
        "ev_estimate":        68.4,
        "method":             "montecarlo",
    }

    mock_opponents = {
        "player1": {"vpip": 35, "pfr": 12, "af": 1.5, "tendency": "loose passive"},
        "player2": {"vpip": 22, "pfr": 18, "af": 2.8, "tendency": "TAG"},
        "player3": {"vpip": 55, "pfr": 8,  "af": 0.9, "tendency": "calling station"},
    }

    print("=" * 60)
    print("  Test ClaudePokerClient")
    print("=" * 60)

    client = CachedClaudeClient()
    advice = client.get_advice(
        game_state        = mock_game_state,
        equity_result     = mock_equity,
        opponent_profiles = mock_opponents,
    )

    print("\n--- Overlay text ---")
    print(advice.to_overlay_text())

    print("\n--- JSON complet ---")
    print(json.dumps(advice.raw_json, indent=2, ensure_ascii=False))
    print(f"\nLatence API : {advice.latency_ms:.0f}ms")

    if advice.error:
        print(f"\n⚠ Erreur : {advice.error}")
