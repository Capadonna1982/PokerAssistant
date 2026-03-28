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

def recommend_action(
    result: EquityResult,
    pot: float = 0.0,
    call_amount: float = 0.0,
    player_stack: float = 0.0,
    stage: str = "preflop",
) -> tuple[str, str, float]:
    """
    Retourne (action, sizing_suggéré, ev_estimée).
    Actions possibles : FOLD, CHECK, CALL, BET, RAISE, ALL-IN
    """
    wp = result.win_probability
    ev = estimate_ev(wp, pot, call_amount, player_stack)

    # Cas check possible (pas de mise adverse)
    if call_amount == 0:
        if wp >= RAISE_THRESHOLD:
            sizing = f"{int(pot * 0.70)}$ (70 % du pot)"
            return "BET", sizing, ev
        elif wp >= CALL_THRESHOLD:
            return "CHECK", "", ev
        else:
            return "CHECK", "", ev   # check/fold au prochain bet

    # Cas avec mise adverse à caller
    if wp < FOLD_THRESHOLD:
        return "FOLD", "", ev

    pot_odds = call_amount / (pot + call_amount) if (pot + call_amount) > 0 else 1.0

    if wp < pot_odds:
        return "FOLD", "", ev

    if wp >= RAISE_THRESHOLD:
        if player_stack > 0 and call_amount > player_stack * 0.5:
            return "ALL-IN", f"{player_stack}$ (tapis)", ev
        sizing = f"{int((pot + call_amount) * 2.5)}$ (2.5× le pot après call)"
        return "RAISE", sizing, ev

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

        action, sizing, ev = recommend_action(
            result, pot, call_amount, player_stack, stage
        )
        result.recommended_action = action
        result.recommended_sizing = sizing
        result.ev_estimate        = ev

        log.info(
            f"Résultat : win={result.win_probability:.1%} | "
            f"main={result.hand_class} | action={result.recommended_action}"
        )
        return result

    def analyse_from_state(self, state) -> EquityResult:
        """
        Raccourci : accepte un GameState (issu de capture.py) directement.
        """
        return self.analyse(
            hole_cards    = state.player_cards,
            board         = state.board_cards,
            num_opponents = max(1, state.num_players - 1),
            pot           = state.pot,
            call_amount   = state.current_bet,
            stage         = state.stage,
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
