"""
engine.py — Moteur de calcul d'équité poker pour l'assistant temps réel
Dépendances : treys
Installation : pip install treys

La librairie `treys` implémente un évaluateur de main 5/6/7 cartes ultra-rapide
basé sur des lookup tables (Cactus Kev). Elle est le standard Python pour ce type
de calcul.
"""

import itertools
import logging
import random
from dataclasses import dataclass, field
from typing import Optional

from treys import Card, Deck, Evaluator

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

ALL_RANKS = "23456789TJQKA"
ALL_SUITS = "shdc"

# Tous les combos possibles dans un deck (52 cartes)
FULL_DECK: list[str] = [r + s for r in ALL_RANKS for s in ALL_SUITS]

# Nombre de simulations Monte Carlo par défaut (équilibre vitesse/précision)
DEFAULT_SIMULATIONS = 2000

# Seuils pour les recommandations (ajustables)
FOLD_THRESHOLD   = 0.30   # < 30 % d'equity → envisager fold
CALL_THRESHOLD   = 0.50   # 30–50 % → call/check
RAISE_THRESHOLD  = 0.65   # > 65 % → raise/bet pour valeur

# ---------------------------------------------------------------------------
# Constantes SPR (Stack-to-Pot Ratio)
# ---------------------------------------------------------------------------
# SPR = stack_effectif / pot
# Interprétation standard :
#   SPR < 1   → situation de shove/call/fold seulement (peu de jeu post-flop)
#   SPR 1–4   → SPR court : favorise les mains nutted, peu de bluff
#   SPR 4–13  → SPR moyen : jeu post-flop standard, raise/call/fold équilibrés
#   SPR > 13  → SPR profond : jeu complexe, draws et spéculation rentables

SPR_VERY_SHORT  = 1.0    # ≤ 1  → shove ou fold direct
SPR_SHORT       = 4.0    # 1–4  → peu de jeu, favorise TP+
SPR_MEDIUM_HIGH = 13.0   # 4–13 → jeu standard
# > 13 → SPR profond, spéculation possible

# Catégories SPR → label lisible
SPR_LABELS = {
    (0.0,  1.0):  "ultra-court (shove/fold)",
    (1.0,  4.0):  "court (TP+ requis)",
    (4.0,  13.0): "moyen (jeu standard)",
    (13.0, 999):  "profond (spéculation)",
}

# ---------------------------------------------------------------------------
# Classes de mains (treys rank : 1 = Royal Flush, 7462 = pire haute carte)
HAND_CLASS_NAMES = {
    1: "Quinte Flush Royale",
    2: "Quinte Flush",
    3: "Carré",
    4: "Full House",
    5: "Couleur",
    6: "Quinte",
    7: "Brelan",
    8: "Deux Paires",
    9: "Paire",
    10: "Haute Carte",
}


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------

@dataclass
class EquityResult:
    """Résultat complet d'un calcul d'équité."""
    win_probability:      float = 0.0    # probabilité de gagner (0.0 – 1.0)
    tie_probability:      float = 0.0    # probabilité d'égalité
    lose_probability:     float = 0.0    # probabilité de perdre
    hand_strength:        int   = 0      # score treys (1 = meilleur, 7462 = pire)
    hand_class:           str   = ""     # ex. "Paire", "Couleur"
    num_combos_total:     int   = 0      # combos adverses évalués
    num_combos_beating:   int   = 0      # combos qui nous battent actuellement
    outs:                 int   = 0      # nombre de outs (draws)
    out_cards:            list[str] = field(default_factory=list)
    recommended_action:   str   = ""
    recommended_sizing:   str   = ""
    ev_estimate:          float = 0.0    # EV estimée (en BB si stack fourni)
    method:               str   = ""     # "exact" ou "montecarlo"
    # ── Champs SPR ────────────────────────────────────────────────────────
    spr:                  float = 0.0    # Stack-to-Pot Ratio
    spr_label:            str   = ""     # ex. "court (TP+ requis)"
    spr_category:         str   = ""     # "ultra_short"|"short"|"medium"|"deep"
    effective_stack:      float = 0.0    # min(stack joueur, stack adversaire)
    pot_odds:             float = 0.0    # % du pot à payer pour caller
    mdf:                  float = 0.0    # Minimum Defense Frequency
    spr_comment:          str   = ""     # conseil SPR en texte libre

    def to_dict(self) -> dict:
        return {
            "win_probability":    round(self.win_probability, 4),
            "tie_probability":    round(self.tie_probability, 4),
            "lose_probability":   round(self.lose_probability, 4),
            "hand_strength":      self.hand_strength,
            "hand_class":         self.hand_class,
            "num_combos_total":   self.num_combos_total,
            "num_combos_beating": self.num_combos_beating,
            "outs":               self.outs,
            "out_cards":          self.out_cards,
            "recommended_action": self.recommended_action,
            "recommended_sizing": self.recommended_sizing,
            "ev_estimate":        round(self.ev_estimate, 2),
            "method":             self.method,
            "spr":                round(self.spr, 2),
            "spr_label":          self.spr_label,
            "spr_category":       self.spr_category,
            "effective_stack":    round(self.effective_stack, 2),
            "pot_odds":           round(self.pot_odds, 4),
            "mdf":                round(self.mdf, 4),
            "spr_comment":        self.spr_comment,
        }


# ---------------------------------------------------------------------------
# Conversion format string ↔ treys Card
# ---------------------------------------------------------------------------

def str_to_treys(card_str: str) -> int:
    """
    Convertit "Ks" → entier treys.
    treys attend le format "Ks", "Ah", "Td", "2c" — identique au nôtre.
    """
    try:
        return Card.new(card_str)
    except Exception as e:
        raise ValueError(f"Carte invalide : '{card_str}' — {e}")


def treys_to_str(card_int: int) -> str:
    """Convertit un entier treys → string lisible "Ks"."""
    return Card.int_to_str(card_int)


def cards_to_treys(cards: list[str]) -> list[int]:
    return [str_to_treys(c) for c in cards]


def remaining_deck(known_cards: list[str]) -> list[str]:
    """Retourne les cartes du deck qui ne sont pas encore révélées."""
    known_upper = {c.upper() for c in known_cards}
    return [c for c in FULL_DECK if c.upper() not in known_upper]


# ---------------------------------------------------------------------------
# Évaluation de la main actuelle
# ---------------------------------------------------------------------------

evaluator = Evaluator()


def evaluate_hand(hole_cards: list[str], board: list[str]) -> tuple[int, str]:
    """
    Évalue la meilleure main possible avec les cartes données.
    Retourne (score_treys, nom_classe).
    score_treys : 1 = meilleur possible (Quinte Flush Royale), 7462 = pire.
    """
    if len(board) < 3:
        # Préflop — pas d'évaluation postflop possible
        return 7462, "Préflop"

    h = cards_to_treys(hole_cards)
    b = cards_to_treys(board)

    try:
        score = evaluator.evaluate(b, h)
        rank_class = evaluator.get_rank_class(score)
        class_name = HAND_CLASS_NAMES.get(rank_class, "Inconnue")
        return score, class_name
    except Exception as e:
        log.error(f"Erreur évaluation main : {e}")
        return 7462, "Erreur"


# ---------------------------------------------------------------------------
# Calcul exact (préflop / flop avec peu d'adversaires)
# ---------------------------------------------------------------------------

def compute_equity_exact(
    hole_cards: list[str],
    board: list[str],
    num_opponents: int,
) -> EquityResult:
    """
    Calcul exhaustif : énumère TOUS les combos adverses possibles.
    Utilisable en préflop (lent) ou sur flop/turn avec 1–2 adversaires.
    Recommandé uniquement si le nombre de combos < ~50 000.
    """
    known = hole_cards + board
    deck  = remaining_deck(known)

    # Nombre de cartes manquantes au board
    board_cards_needed = 5 - len(board)

    wins = ties = losses = 0
    combos_beating = 0
    total_combos = 0

    # Pairs de cartes adverses possibles
    opp_combos = list(itertools.combinations(deck, 2 * num_opponents))

    # Limiter si trop grand (fallback Monte Carlo)
    if len(opp_combos) > 80_000:
        log.info(f"{len(opp_combos)} combos — basculement vers Monte Carlo")
        return compute_equity_montecarlo(hole_cards, board, num_opponents)

    h = cards_to_treys(hole_cards)
    b = cards_to_treys(board)

    for combo in opp_combos:
        # Convertir en liste de paires pour chaque adversaire
        opp_hands = [
            list(cards_to_treys([Card.int_to_str(combo[i*2]), Card.int_to_str(combo[i*2+1])]))
            for i in range(num_opponents)
        ]

        # Cartes restantes pour compléter le board
        used = set(combo)
        remaining = [c for c in cards_to_treys(deck)
                     if c not in used and c not in b and c not in h]

        if board_cards_needed > 0:
            if len(remaining) < board_cards_needed:
                continue
            run_outs = list(itertools.combinations(remaining, board_cards_needed))
        else:
            run_outs = [()]

        for run in run_outs:
            full_board = b + list(run)
            try:
                my_score = evaluator.evaluate(full_board, h)
                opp_scores = [evaluator.evaluate(full_board, oh) for oh in opp_hands]
                best_opp = min(opp_scores)  # score treys : plus petit = meilleur

                total_combos += 1
                if my_score < best_opp:
                    wins += 1
                elif my_score == best_opp:
                    ties += 1
                else:
                    losses += 1
                    if len(board) >= 3:
                        combos_beating += 1
            except Exception:
                continue

    if total_combos == 0:
        return EquityResult(method="exact")

    result = EquityResult(
        win_probability    = wins  / total_combos,
        tie_probability    = ties  / total_combos,
        lose_probability   = losses / total_combos,
        num_combos_total   = total_combos,
        num_combos_beating = combos_beating,
        method             = "exact",
    )
    _enrich_result(result, hole_cards, board)
    return result


# ---------------------------------------------------------------------------
# Monte Carlo (rapide, précis à ±2 % avec 2000 simulations)
# ---------------------------------------------------------------------------

def compute_equity_montecarlo(
    hole_cards: list[str],
    board: list[str],
    num_opponents: int,
    num_simulations: int = DEFAULT_SIMULATIONS,
) -> EquityResult:
    """
    Simulation Monte Carlo : tire aléatoirement des mains adverses et des
    run-outs, calcule le taux de victoire sur `num_simulations` itérations.
    Précision : ±2 % à 2000 sims, ±1 % à 5000 sims.
    """
    known   = hole_cards + board
    deck    = remaining_deck(known)
    h       = cards_to_treys(hole_cards)
    b       = cards_to_treys(board)
    cards_needed_board = 5 - len(board)
    cards_needed_total = cards_needed_board + 2 * num_opponents

    wins = ties = losses = 0
    combos_beating = 0

    for _ in range(num_simulations):
        if len(deck) < cards_needed_total:
            break

        sample = random.sample(deck, cards_needed_total)
        run_out  = cards_to_treys(sample[:cards_needed_board])
        opp_pool = sample[cards_needed_board:]

        full_board = b + run_out

        opp_hands = [
            cards_to_treys([opp_pool[i*2], opp_pool[i*2+1]])
            for i in range(num_opponents)
        ]

        try:
            my_score  = evaluator.evaluate(full_board, h)
            opp_scores = [evaluator.evaluate(full_board, oh) for oh in opp_hands]
            best_opp   = min(opp_scores)

            if my_score < best_opp:
                wins += 1
            elif my_score == best_opp:
                ties += 1
            else:
                losses += 1
                combos_beating += 1
        except Exception:
            continue

    total = wins + ties + losses
    if total == 0:
        return EquityResult(method="montecarlo")

    result = EquityResult(
        win_probability    = wins   / total,
        tie_probability    = ties   / total,
        lose_probability   = losses / total,
        num_combos_total   = total,
        num_combos_beating = combos_beating,
        method             = "montecarlo",
    )
    _enrich_result(result, hole_cards, board)
    return result


# ---------------------------------------------------------------------------
# Calcul des outs (draws)
# ---------------------------------------------------------------------------

def compute_outs(
    hole_cards: list[str],
    board: list[str],
) -> tuple[int, list[str]]:
    """
    Identifie les cartes qui améliorent notre main d'au moins une classe.
    Retourne (nombre_de_outs, liste_des_cartes_out).
    Applicable sur flop (15 outs max) et turn (quelques outs).
    """
    if len(board) < 3:
        return 0, []

    known = hole_cards + board
    deck  = remaining_deck(known)

    current_score, _ = evaluate_hand(hole_cards, board)
    out_cards = []

    for card in deck:
        new_board = board + [card]
        new_score, _ = evaluate_hand(hole_cards, new_board)
        if new_score < current_score:   # score plus bas = main meilleure
            out_cards.append(card)

    return len(out_cards), out_cards


# ---------------------------------------------------------------------------
# EV estimée
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Calcul SPR
# ---------------------------------------------------------------------------

def compute_spr(
    player_stack:    float,
    pot:             float,
    opp_stack:       float = 0.0,
) -> float:
    """
    SPR = stack_effectif / pot

    stack_effectif = min(stack joueur, stack adversaire)
    Si opp_stack non fourni, utilise player_stack seul.
    Retourne 0.0 si pot == 0.
    """
    if pot <= 0:
        return 0.0
    effective = min(player_stack, opp_stack) if opp_stack > 0 else player_stack
    return round(effective / pot, 2)


def classify_spr(spr: float) -> tuple[str, str]:
    """
    Classe le SPR en catégorie et retourne (label, category).

    category : "ultra_short" | "short" | "medium" | "deep"
    """
    for (low, high), label in SPR_LABELS.items():
        if low <= spr < high:
            if spr < SPR_VERY_SHORT:
                return label, "ultra_short"
            elif spr < SPR_SHORT:
                return label, "short"
            elif spr < SPR_MEDIUM_HIGH:
                return label, "medium"
            else:
                return label, "deep"
    return "profond (spéculation)", "deep"


def compute_pot_odds(call_amount: float, pot: float) -> float:
    """
    Pot odds = call / (pot + call)
    Exprimé en fraction (0.0–1.0).
    Ex : call=40, pot=120 → 40/160 = 0.25 (25%)
    """
    total = pot + call_amount
    if total <= 0:
        return 0.0
    return round(call_amount / total, 4)


def compute_mdf(bet_size: float, pot: float) -> float:
    """
    MDF (Minimum Defense Frequency) = pot / (pot + bet)

    Fréquence minimale à laquelle on doit défendre (call ou raise)
    pour rendre le bluff adverse non-rentable.
    Ex : bet=60 dans pot=100 → MDF = 100/160 = 62.5%
    Si on fold plus que (1-MDF), on est exploitable.
    """
    total = pot + bet_size
    if total <= 0:
        return 0.0
    return round(pot / total, 4)


def spr_strategy_comment(
    spr:          float,
    stage:        str,
    hand_class:   str,
    win_prob:     float,
    position:     str = "",
) -> str:
    """
    Génère un commentaire stratégique basé sur le SPR.
    Retourne une phrase d'action courte.
    """
    strong_hands = {"Quinte Flush Royale", "Quinte Flush", "Carré",
                    "Full House", "Couleur", "Quinte", "Brelan"}
    medium_hands = {"Deux Paires", "Paire"}
    is_strong  = hand_class in strong_hands
    is_medium  = hand_class in medium_hands
    is_draw    = hand_class in {"Haute Carte"} and win_prob > 0.30

    if spr <= SPR_VERY_SHORT:
        # SPR ultra-court : décision binaire
        if win_prob >= 0.45:
            return f"SPR {spr:.1f} — shove ou call direct, pas de fold avec {win_prob:.0%} d'equity."
        return f"SPR {spr:.1f} — shove ou fold, pas de call partiel."

    if spr <= SPR_SHORT:
        # SPR court : favorise les mains complètes, défavorise les draws
        if is_strong:
            return f"SPR {spr:.1f} court — commit ton stack avec {hand_class}, peu de fold equity adverse."
        if is_draw:
            return f"SPR {spr:.1f} court — les draws manquent d'implied odds, préfère fold ou semi-bluff."
        if is_medium and win_prob >= 0.55:
            return f"SPR {spr:.1f} — {hand_class} jouable pour valeur, évite les gros bluffs."
        return f"SPR {spr:.1f} court — joue serré, TP+ minimum pour continuer."

    if spr <= SPR_MEDIUM_HIGH:
        # SPR moyen : jeu post-flop équilibré
        pos_str = f" en {position}" if position else ""
        if is_strong:
            return f"SPR {spr:.1f} moyen — construis le pot avec {hand_class}{pos_str}."
        if is_draw:
            return f"SPR {spr:.1f} — les draws ont des implied odds corrects, semi-bluff rentable."
        return f"SPR {spr:.1f} moyen — jeu standard{pos_str}, range balanced."

    # SPR profond : spéculation rentable
    if is_draw:
        return f"SPR {spr:.1f} profond — excellent pour les draws, implied odds maximaux."
    if is_strong:
        return f"SPR {spr:.1f} profond — construis lentement le pot avec {hand_class}."
    return f"SPR {spr:.1f} profond — bluffs et spéculation rentables, position cruciale."


def spr_adjust_recommendation(
    action:       str,
    sizing:       str,
    spr:          float,
    win_prob:     float,
    pot:          float,
    player_stack: float,
    hand_class:   str,
    stage:        str,
) -> tuple[str, str]:
    """
    Ajuste l'action et le sizing recommandés selon le SPR.
    Retourne (action_ajustée, sizing_ajusté).
    """
    strong_hands = {"Quinte Flush Royale", "Quinte Flush", "Carré",
                    "Full House", "Couleur", "Quinte", "Brelan"}

    # SPR ultra-court → shove ou fold
    if spr <= SPR_VERY_SHORT and player_stack > 0:
        if win_prob >= 0.45:
            return "ALL-IN", f"{player_stack:.0f}$ (SPR {spr:.1f} → shove optimal)"
        elif win_prob < 0.30:
            return "FOLD", ""

    # SPR court → sizing réduit, ne pas over-bet
    if spr <= SPR_SHORT:
        if action in ("BET", "RAISE"):
            # Réduire le sizing à 50-60% du pot (pot control)
            safe_bet = int(pot * 0.55)
            return action, f"{safe_bet}$ (55% pot — SPR {spr:.1f} court)"

    # SPR profond avec draw fort → augmenter sizing pour extraire
    if spr > SPR_MEDIUM_HIGH and hand_class in strong_hands:
        if action in ("BET", "RAISE"):
            large_bet = int(pot * 0.85)
            return action, f"{large_bet}$ (85% pot — SPR profond, extraction max)"

    return action, sizing


# ---------------------------------------------------------------------------
# EV
# ---------------------------------------------------------------------------

def estimate_ev(
    win_prob: float,
    pot: float,
    call_amount: float,
    player_stack: float = 0.0,
) -> float:
    """
    EV simplifiée d'un call :
    EV = (win_prob × pot) - (lose_prob × call_amount)
    Valeur positive → call rentable.
    """
    lose_prob = 1.0 - win_prob
    ev = (win_prob * pot) - (lose_prob * call_amount)
    return round(ev, 2)


# ---------------------------------------------------------------------------
# Recommandation stratégique
# ---------------------------------------------------------------------------

# Ajustements de range selon la position
POSITION_FOLD_BONUS = {
    "UTG":   0.08,  # fold threshold +8% en UTG (jouer très serré)
    "UTG+1": 0.06,
    "UTG+2": 0.05,
    "MP":    0.03,
    "MP+1":  0.02,
    "SB":    0.04,  # désavantage post-flop
    "BB":    -0.05, # pot odds souvent favorables
    "HJ":    0.01,
    "CO":    -0.02, # légère agressivité
    "BTN":   -0.05, # très agressif au bouton
}

POSITION_RAISE_BONUS = {
    "BTN":   -0.08, # raise avec moins d'equity au bouton
    "CO":    -0.05,
    "HJ":    -0.03,
    "SB":    0.03,
    "BB":    0.02,
    "UTG":   0.05,  # raise seulement avec beaucoup d'equity en UTG
    "UTG+1": 0.04,
}


def recommend_action(
    result: EquityResult,
    pot: float = 0.0,
    call_amount: float = 0.0,
    player_stack: float = 0.0,
    stage: str = "preflop",
    position: str = "",
    spr: float = 0.0,
) -> tuple[str, str, float]:
    """
    Retourne (action, sizing_suggéré, ev_estimée).
    Actions possibles : FOLD, CHECK, CALL, BET, RAISE, ALL-IN
    Intègre les ajustements de position ET du SPR pour une stratégie optimale.
    """
    wp = result.win_probability
    ev = estimate_ev(wp, pot, call_amount, player_stack)

    # Ajustements selon la position
    fold_adj  = POSITION_FOLD_BONUS.get(position, 0.0)
    raise_adj = POSITION_RAISE_BONUS.get(position, 0.0)

    effective_fold_threshold  = FOLD_THRESHOLD  + fold_adj
    effective_raise_threshold = RAISE_THRESHOLD + raise_adj

    # Sizing adapté à la position
    def sizing_for_position(base_ratio: float) -> float:
        """BTN ouvre plus large, UTG plus petit pour protection."""
        pos_mult = {
            "BTN": 1.15, "CO": 1.10, "HJ": 1.05,
            "SB":  0.90, "UTG": 0.85, "UTG+1": 0.88,
        }.get(position, 1.0)
        return base_ratio * pos_mult

    # Cas check possible (pas de mise adverse)
    if call_amount == 0:
        if wp >= effective_raise_threshold:
            ratio  = sizing_for_position(0.70)
            sizing = f"{int(pot * ratio)}$ ({int(ratio*100)}% du pot)"
            return "BET", sizing, ev
        elif wp >= CALL_THRESHOLD:
            return "CHECK", "", ev
        else:
            return "CHECK", "", ev

    # Cas avec mise adverse à caller
    if wp < effective_fold_threshold:
        return "FOLD", "", ev

    pot_odds = call_amount / (pot + call_amount) if (pot + call_amount) > 0 else 1.0

    if wp < pot_odds:
        return "FOLD", "", ev

    if wp >= effective_raise_threshold:
        if player_stack > 0 and call_amount > player_stack * 0.5:
            return "ALL-IN", f"{player_stack}$ (tapis)", ev
        ratio  = sizing_for_position(2.5)
        sizing = f"{int((pot + call_amount) * ratio)}$ ({ratio:.1f}× le pot après call)"
        return "RAISE", sizing, ev

    # ── Ajustement SPR final ──────────────────────────────────────────────
    if spr > 0:
        action_candidate = (
            "BET"  if call_amount == 0 and wp >= effective_raise_threshold else
            "RAISE" if call_amount > 0 and wp >= effective_raise_threshold else
            "CALL"  if call_amount > 0 else
            "CHECK"
        )
        if action_candidate in ("BET", "RAISE"):
            sizing_candidate = (
                f"{int(pot * sizing_for_position(0.70))}$ ({int(sizing_for_position(0.70)*100)}% du pot)"
                if call_amount == 0 else
                f"{int((pot + call_amount) * sizing_for_position(2.5))}$"
            )
        else:
            sizing_candidate = f"{call_amount}$" if call_amount > 0 else ""

        adj_action, adj_sizing = spr_adjust_recommendation(
            action_candidate, sizing_candidate, spr, wp,
            pot, player_stack, result.hand_class, stage
        )
        return adj_action, adj_sizing, ev

    return "CALL", f"{call_amount}$", ev


# ---------------------------------------------------------------------------
# Enrichissement du résultat
# ---------------------------------------------------------------------------

def _enrich_result(
    result: EquityResult,
    hole_cards: list[str],
    board: list[str],
    pot: float = 0.0,
    call_amount: float = 0.0,
    player_stack: float = 0.0,
    stage: str = "",
) -> None:
    """Complète un EquityResult avec la force de main, les outs, et la recommandation."""
    score, class_name = evaluate_hand(hole_cards, board)
    result.hand_strength = score
    result.hand_class    = class_name

    if len(board) in (3, 4):
        outs, out_cards = compute_outs(hole_cards, board)
        result.outs      = outs
        result.out_cards = out_cards

    action, sizing, ev = recommend_action(
        result, pot, call_amount, player_stack, stage
    )
    result.recommended_action  = action
    result.recommended_sizing  = sizing
    result.ev_estimate         = ev


# ---------------------------------------------------------------------------
# Interface principale
# ---------------------------------------------------------------------------

class PokerEngine:
    """
    Point d'entrée principal du moteur.
    Choisit automatiquement entre calcul exact et Monte Carlo
    selon la street et le nombre d'adversaires.
    """

    # Seuil de combos au-dessus duquel on bascule en Monte Carlo
    EXACT_LIMIT = 50_000

    def __init__(self, simulations: int = DEFAULT_SIMULATIONS):
        self.simulations = simulations

    def analyse(
        self,
        hole_cards:    list[str],
        board:         list[str],
        num_opponents: int,
        pot:           float = 0.0,
        call_amount:   float = 0.0,
        player_stack:  float = 0.0,
        stage:         str   = "",
        position:      str   = "",
    ) -> EquityResult:
        """
        Analyse complète d'une situation de jeu.

        Paramètres :
          hole_cards    : ex. ["Ks", "7h"]
          board         : ex. ["Ah", "3d", "9c"]  (vide si préflop)
          num_opponents : nombre d'adversaires actifs
          pot           : taille du pot actuel
          call_amount   : montant à caller (0 si check possible)
          player_stack  : stack du joueur (pour les décisions all-in)
          stage         : "preflop" / "flop" / "turn" / "river"
          position      : "BTN" / "CO" / "HJ" / "MP" / "UTG" / "SB" / "BB"
        """
        if not hole_cards or len(hole_cards) < 2:
            log.warning("Cartes joueur manquantes.")
            return EquityResult()

        num_opponents = max(1, num_opponents)

        # Choisir la méthode
        combos_estimate = _estimate_combo_count(len(board), num_opponents)
        log.info(f"Combos estimés : {combos_estimate:,} — méthode : {'exact' if combos_estimate < self.EXACT_LIMIT else 'montecarlo'}")

        if combos_estimate < self.EXACT_LIMIT:
            result = compute_equity_exact(hole_cards, board, num_opponents)
        else:
            result = compute_equity_montecarlo(
                hole_cards, board, num_opponents, self.simulations
            )

        # Enrichir avec le contexte de mise
        result.hand_strength, result.hand_class = evaluate_hand(hole_cards, board)

        if len(board) in (3, 4):
            result.outs, result.out_cards = compute_outs(hole_cards, board)

        # ── Calcul SPR ────────────────────────────────────────────────────
        spr_value = compute_spr(player_stack, pot)
        spr_lbl, spr_cat = classify_spr(spr_value)
        pot_odds_val = compute_pot_odds(call_amount, pot)
        mdf_val      = compute_mdf(call_amount, pot)
        spr_comment  = spr_strategy_comment(
            spr_value, stage, result.hand_class,
            result.win_probability, position
        )

        result.spr             = spr_value
        result.spr_label       = spr_lbl
        result.spr_category    = spr_cat
        result.effective_stack = min(player_stack, player_stack)  # enrichi si opp_stack fourni
        result.pot_odds        = pot_odds_val
        result.mdf             = mdf_val
        result.spr_comment     = spr_comment

        action, sizing, ev = recommend_action(
            result, pot, call_amount, player_stack, stage, position, spr_value
        )
        result.recommended_action = action
        result.recommended_sizing = sizing
        result.ev_estimate        = ev

        log.info(
            f"Résultat : win={result.win_probability:.1%} | "
            f"main={result.hand_class} | SPR={spr_value:.1f} ({spr_cat}) | "
            f"action={result.recommended_action}"
        )
        return result

    def analyse_from_state(self, state) -> EquityResult:
        """
        Raccourci : accepte un GameState (issu de capture.py) directement.
        Transmet automatiquement la position détectée par capture.py.
        """
        return self.analyse(
            hole_cards    = state.player_cards,
            board         = state.board_cards,
            num_opponents = max(1, state.num_players - 1),
            pot           = state.pot,
            call_amount   = state.current_bet,
            player_stack  = getattr(state, "player_stack", 0.0),
            stage         = state.stage,
            position      = getattr(state, "position", ""),
        )


# ---------------------------------------------------------------------------
# Utilitaire interne
# ---------------------------------------------------------------------------

def _estimate_combo_count(board_len: int, num_opponents: int) -> int:
    """Estimation rapide du nombre de combos à énumérer selon la street."""
    remaining = 52 - 2 - board_len  # cartes inconnues
    opp_cards = 2 * num_opponents
    if remaining < opp_cards:
        return 0
    # C(remaining, opp_cards) approximé
    from math import comb
    return comb(remaining, opp_cards)


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    engine = PokerEngine(simulations=2000)

    scenarios = [
        {
            "label": "Préflop — Ks 7h vs 8 adversaires",
            "hole": ["Ks", "7h"], "board": [],
            "opponents": 8, "pot": 0, "call": 0,
        },
        {
            "label": "Flop — Ks 7h | Kd 7d 2c (top two pair)",
            "hole": ["Ks", "7h"], "board": ["Kd", "7d", "2c"],
            "opponents": 3, "pot": 120, "call": 40,
        },
        {
            "label": "Turn — Ks 7h | Kd 7d 2c As (draw flush adverse possible)",
            "hole": ["Ks", "7h"], "board": ["Kd", "7d", "2c", "As"],
            "opponents": 2, "pot": 200, "call": 80,
        },
        {
            "label": "River — As Kh | Ah Kd 3c 7s 2h (top two pair amélioré)",
            "hole": ["As", "Kh"], "board": ["Ah", "Kd", "3c", "7s", "2h"],
            "opponents": 1, "pot": 500, "call": 200,
        },
    ]

    for s in scenarios:
        print(f"\n{'='*60}")
        print(f"  {s['label']}")
        print('='*60)
        result = engine.analyse(
            hole_cards    = s["hole"],
            board         = s["board"],
            num_opponents = s["opponents"],
            pot           = s["pot"],
            call_amount   = s["call"],
        )
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
