"""
auto_new_hand.py — Détection automatique des nouvelles mains PokerStars
Semaine 4 — Automatisation

Détecte le début d'une nouvelle main par 3 mécanismes complémentaires :

  1. Changement de cartes joueur   (le plus fiable — cartes nouvelles = nouvelle main)
  2. Changement de pot à zéro      (pot retombe à 0 ou quasi entre deux mains)
  3. Changement de board           (board redevient vide entre deux mains)
  4. Timeout de main               (plus de changement depuis N secondes = main terminée)

Chaque signal est pondéré — il faut atteindre un score minimum pour confirmer
la nouvelle main (évite les faux positifs sur les fluctuations OCR).

Usage dans main.py :
    from auto_new_hand import AutoNewHandDetector

    detector = AutoNewHandDetector(callback=assistant.new_hand)

    # Dans _capture_loop, remplace le bloc "nouvelle main manuelle" :
    detector.update(state)          # détecte et appelle callback si nécessaire
    detector.force()                # forcer depuis la console ('n')
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Poids de chaque signal (total ≥ TRIGGER_SCORE → nouvelle main confirmée)
WEIGHT_CARDS_CHANGE   = 10   # cartes joueur changent → très fiable
WEIGHT_CARDS_APPEAR   = 8    # cartes apparaissent (étaient vides) → fiable
WEIGHT_BOARD_CLEARED  = 5    # board redevient vide → bon signal
WEIGHT_POT_RESET      = 3    # pot retombe bas → signal faible seul
WEIGHT_TIMEOUT        = 6    # aucun changement depuis N secondes

TRIGGER_SCORE         = 8    # score minimum pour confirmer
HAND_TIMEOUT_S        = 90   # secondes sans activité → fin de main probable
POT_RESET_THRESHOLD   = 5.0  # $ — pot en dessous = probable début de main


@dataclass
class HandState:
    """Snapshot de l'état courant pour comparaison."""
    player_cards: list  = field(default_factory=list)
    board_cards:  list  = field(default_factory=list)
    pot:          float = 0.0
    stage:        str   = ""
    timestamp:    float = field(default_factory=time.time)

    def is_empty(self) -> bool:
        return not self.player_cards

    def cards_key(self) -> str:
        """Clé de hachage rapide pour les cartes."""
        return "|".join(sorted(self.player_cards))


@dataclass
class DetectionEvent:
    """Événement de détection de nouvelle main."""
    reason:     str         # description du signal déclencheur
    score:      int         # score cumulé
    timestamp:  float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Détecteur principal
# ---------------------------------------------------------------------------

class AutoNewHandDetector:
    """
    Détecte automatiquement les nouvelles mains et appelle un callback.

    Usage :
        def on_new_hand():
            print("Nouvelle main détectée !")

        detector = AutoNewHandDetector(callback=on_new_hand)

        # Dans la boucle de capture :
        for state in capture_loop():
            detector.update(state)
            process(state)
    """

    def __init__(
        self,
        callback:         Optional[Callable]  = None,
        min_interval_s:   float               = 5.0,
        hand_timeout_s:   float               = HAND_TIMEOUT_S,
        pot_threshold:    float               = POT_RESET_THRESHOLD,
        trigger_score:    int                 = TRIGGER_SCORE,
        enabled:          bool                = True,
    ):
        """
        callback        : fonction appelée à chaque nouvelle main détectée
        min_interval_s  : délai minimum entre deux déclenchements (anti-spam)
        hand_timeout_s  : délai sans activité avant fin de main
        pot_threshold   : montant de pot considéré comme "zéro"
        trigger_score   : score minimum pour confirmer une nouvelle main
        enabled         : activer/désactiver la détection automatique
        """
        self.callback       = callback
        self.min_interval_s = min_interval_s
        self.hand_timeout_s = hand_timeout_s
        self.pot_threshold  = pot_threshold
        self.trigger_score  = trigger_score
        self.enabled        = enabled

        # État courant suivi
        self._prev:            Optional[HandState] = None
        self._last_trigger_ts: float = 0.0
        self._last_change_ts:  float = time.time()
        self._hand_count:      int   = 0
        self._consecutive_empty:int  = 0   # frames consécutives sans cartes

        # Historique des événements (debug)
        self._events: list[DetectionEvent] = []

        log.info(
            f"AutoNewHandDetector initialisé — "
            f"{'activé' if enabled else 'désactivé'}, "
            f"score_min={trigger_score}, "
            f"timeout={hand_timeout_s}s, "
            f"intervalle_min={min_interval_s}s"
        )

    # ------------------------------------------------------------------
    # API principale
    # ------------------------------------------------------------------

    def update(self, state) -> bool:
        """
        Analyse le GameState courant et déclenche le callback si une
        nouvelle main est détectée.

        Retourne True si une nouvelle main a été déclenchée.
        """
        if not self.enabled:
            return False

        current = HandState(
            player_cards = list(state.player_cards or []),
            board_cards  = list(state.board_cards  or []),
            pot          = float(state.pot          or 0.0),
            stage        = str(state.stage          or ""),
        )

        # Premier état — initialiser sans déclencher
        if self._prev is None:
            self._prev = current
            self._last_change_ts = time.time()
            return False

        score, reasons = self._compute_score(current)

        # Mise à jour du timestamp de dernier changement
        if self._has_changed(current):
            self._last_change_ts = time.time()

        triggered = False
        if score >= self.trigger_score:
            triggered = self._trigger(score, reasons)

        # Détection par timeout (main inactive trop longtemps)
        elapsed_no_change = time.time() - self._last_change_ts
        if (elapsed_no_change > self.hand_timeout_s and
                not current.is_empty() and
                not self._prev.is_empty()):
            timeout_triggered = self._trigger(
                WEIGHT_TIMEOUT,
                [f"timeout {elapsed_no_change:.0f}s sans changement"]
            )
            if timeout_triggered:
                triggered = True
                self._last_change_ts = time.time()

        self._prev = current
        return triggered

    def force(self, reason: str = "commande manuelle") -> None:
        """Force une nouvelle main (équivalent à taper 'n' dans la console)."""
        self._do_trigger(0, [reason])

    def reset(self) -> None:
        """Réinitialise l'état interne (nouvelle session)."""
        self._prev            = None
        self._last_trigger_ts = 0.0
        self._last_change_ts  = time.time()
        self._consecutive_empty = 0
        log.info("AutoNewHandDetector réinitialisé.")

    @property
    def hand_count(self) -> int:
        return self._hand_count

    @property
    def last_events(self) -> list[DetectionEvent]:
        return self._events[-10:]

    # ------------------------------------------------------------------
    # Calcul du score de détection
    # ------------------------------------------------------------------

    def _compute_score(self, current: HandState) -> tuple[int, list[str]]:
        """
        Calcule le score de détection de nouvelle main.
        Retourne (score_total, liste_de_raisons).
        """
        score   = 0
        reasons = []
        prev    = self._prev

        # ── Signal 1 : les cartes joueur ont changé ───────────────────────
        if (current.player_cards and
                prev.player_cards and
                current.cards_key() != prev.cards_key()):
            score += WEIGHT_CARDS_CHANGE
            reasons.append(
                f"cartes changées {prev.player_cards} → {current.player_cards}"
            )

        # ── Signal 2 : les cartes sont apparues (étaient vides) ──────────
        elif current.player_cards and not prev.player_cards:
            score += WEIGHT_CARDS_APPEAR
            self._consecutive_empty = 0
            reasons.append(f"cartes apparues : {current.player_cards}")

        # ── Signal 3 : les cartes ont disparu (inter-main) ───────────────
        elif not current.player_cards and prev.player_cards:
            self._consecutive_empty += 1
            # Ne déclenche pas seul, mais prépare la détection suivante
            log.debug(f"Cartes disparues (frame {self._consecutive_empty})")

        # ── Signal 4 : le board redevient vide ───────────────────────────
        if not current.board_cards and prev.board_cards:
            score += WEIGHT_BOARD_CLEARED
            reasons.append(f"board vidé : {prev.board_cards} → []")

        # ── Signal 5 : pot retombe à zéro ────────────────────────────────
        if (current.pot <= self.pot_threshold and
                prev.pot > self.pot_threshold * 3):
            score += WEIGHT_POT_RESET
            reasons.append(f"pot reset {prev.pot:.0f}$ → {current.pot:.0f}$")

        # ── Signal 6 : stage retourne à preflop avec nouvelles cartes ─────
        if (current.stage == "preflop" and
                prev.stage in ("flop", "turn", "river") and
                current.player_cards):
            score += WEIGHT_CARDS_APPEAR
            reasons.append(f"retour preflop ({prev.stage} → preflop)")

        return score, reasons

    def _has_changed(self, current: HandState) -> bool:
        """Vérifie si quelque chose a changé depuis le dernier état."""
        if self._prev is None:
            return True
        return (
            current.player_cards != self._prev.player_cards or
            current.board_cards  != self._prev.board_cards  or
            abs(current.pot - self._prev.pot) > 0.5
        )

    # ------------------------------------------------------------------
    # Déclenchement
    # ------------------------------------------------------------------

    def _trigger(self, score: int, reasons: list[str]) -> bool:
        """Vérifie les conditions et déclenche si OK."""
        now = time.time()

        # Respect du délai minimum entre deux déclenchements
        if now - self._last_trigger_ts < self.min_interval_s:
            return False

        return self._do_trigger(score, reasons)

    def _do_trigger(self, score: int, reasons: list[str]) -> bool:
        """Déclenche effectivement la nouvelle main."""
        self._hand_count      += 1
        self._last_trigger_ts  = time.time()
        self._consecutive_empty = 0

        reason_str = " + ".join(reasons) if reasons else "déclenchement manuel"
        event = DetectionEvent(reason=reason_str, score=score)
        self._events.append(event)

        log.info(
            f"Nouvelle main #{self._hand_count} détectée "
            f"(score={score}) — {reason_str}"
        )

        if self.callback:
            try:
                self.callback()
            except Exception as e:
                log.error(f"Erreur callback nouvelle main : {e}")

        return True

    # ------------------------------------------------------------------
    # Diagnostic
    # ------------------------------------------------------------------

    def status_str(self) -> str:
        """Retourne une ligne de statut lisible."""
        elapsed = time.time() - self._last_change_ts
        return (
            f"AutoNewHand : mains={self._hand_count} "
            f"dernier_changement={elapsed:.0f}s "
            f"{'[ACTIVÉ]' if self.enabled else '[DÉSACTIVÉ]'}"
        )
