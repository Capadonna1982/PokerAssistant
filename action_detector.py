"""
action_detector.py — Détection des actions adverses en temps réel sur PokerStars
Dépendances : opencv-python, numpy, pytesseract (déjà installés)

Détecte automatiquement :
  - FOLD   : siège qui disparaît / carte retournée face verso
  - CALL   : montant égalisé / badge "Call" visible
  - RAISE  : badge "Raise" / augmentation du montant de mise
  - CHECK  : badge "Check" / aucune mise ajoutée
  - ALL-IN : badge rouge "All-In" / stack à zéro
  - BET    : première mise dans le tour

Approche multicouche :
  1. OCR des badges d'action affichés par PokerStars (~2s après l'action)
  2. Analyse différentielle des stacks (variation entre deux captures)
  3. Détection visuelle des cartes retournées (fold → cartes grisées)
  4. Analyse de couleur des zones de mise (badge coloré = action récente)
"""

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

class Action(str, Enum):
    FOLD   = "FOLD"
    CALL   = "CALL"
    RAISE  = "RAISE"
    CHECK  = "CHECK"
    BET    = "BET"
    ALL_IN = "ALL-IN"
    BLIND  = "BLIND"
    UNKNOWN = "UNKNOWN"

# Mots-clés OCR → Action (PokerStars FR + EN)
ACTION_KEYWORDS: dict[str, Action] = {
    # Fold
    "fold":    Action.FOLD, "couché":  Action.FOLD, "passe":   Action.FOLD,
    "se couche": Action.FOLD,
    # Call
    "call":    Action.CALL, "suivi":   Action.CALL, "suit":    Action.CALL,
    "appel":   Action.CALL,
    # Raise
    "raise":   Action.RAISE, "relance": Action.RAISE, "relancé": Action.RAISE,
    "reraise": Action.RAISE, "3bet":    Action.RAISE,
    # Check
    "check":   Action.CHECK, "passe sans": Action.CHECK, "parole": Action.CHECK,
    # Bet
    "bet":     Action.BET,   "mise":    Action.BET,   "misé":   Action.BET,
    # All-In
    "all-in":  Action.ALL_IN, "allin":  Action.ALL_IN, "tapis":  Action.ALL_IN,
    "all in":  Action.ALL_IN,
    # Blind
    "sb":      Action.BLIND, "bb":      Action.BLIND, "blind":   Action.BLIND,
    "ante":    Action.BLIND,
}

# Couleurs HSV des badges d'action PokerStars
# (approximations — à affiner selon le thème de table)
BADGE_COLORS_HSV = {
    Action.FOLD:   {"low": np.array([0,   0,   80]),  "high": np.array([180, 30,  180])},  # gris
    Action.CALL:   {"low": np.array([35,  60,  120]), "high": np.array([75,  255, 255])},  # vert
    Action.RAISE:  {"low": np.array([0,   100, 120]), "high": np.array([15,  255, 255])},  # rouge/orange
    Action.CHECK:  {"low": np.array([90,  60,  120]), "high": np.array([130, 255, 255])},  # bleu
    Action.BET:    {"low": np.array([0,   100, 120]), "high": np.array([15,  255, 255])},  # rouge/orange
    Action.ALL_IN: {"low": np.array([0,   150, 150]), "high": np.array([10,  255, 255])},  # rouge vif
}

# Positions des sièges adverses sur une table 9-max 1920×1080
# Format : (x_centre, y_centre) en ratio de l'écran (0.0–1.0)
# Siège 0 = joueur (bas-centre), sièges 1–8 = adversaires (sens horaire)
SEAT_POSITIONS_RATIO = {
    0: (0.50, 0.90),   # joueur (bas-centre)
    1: (0.80, 0.82),   # bas-droite
    2: (0.95, 0.60),   # droite
    3: (0.88, 0.32),   # haut-droite
    4: (0.65, 0.15),   # haut-droite-centre
    5: (0.50, 0.10),   # haut-centre
    6: (0.35, 0.15),   # haut-gauche-centre
    7: (0.12, 0.32),   # haut-gauche
    8: (0.05, 0.60),   # gauche
    9: (0.20, 0.82),   # bas-gauche
}

# Zone de badge autour de chaque siège (ratio de l'image)
BADGE_ZONE_W = 0.12
BADGE_ZONE_H = 0.06

# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------

@dataclass
class OpponentAction:
    """Action détectée pour un adversaire donné."""
    seat:       int            # numéro de siège (1–8, 0 = joueur)
    action:     Action         # FOLD / CALL / RAISE / CHECK / BET / ALL-IN
    amount:     float  = 0.0   # montant si CALL / RAISE / BET
    confidence: float  = 0.0   # 0.0 – 1.0
    method:     str    = ""    # "ocr" | "visual" | "stack_diff" | "color"
    timestamp:  float  = field(default_factory=time.time)

    def __str__(self) -> str:
        amt = f" {self.amount:.0f}$" if self.amount > 0 else ""
        return f"Siège {self.seat}: {self.action.value}{amt} (conf={self.confidence:.0%})"


@dataclass
class ActionEvent:
    """Événement d'action avec contexte de jeu."""
    actions:       list[OpponentAction]
    stage:         str   = ""
    pot_before:    float = 0.0
    pot_after:     float = 0.0
    timestamp:     float = field(default_factory=time.time)

    @property
    def has_aggression(self) -> bool:
        return any(a.action in (Action.RAISE, Action.BET, Action.ALL_IN)
                   for a in self.actions)

    @property
    def num_folds(self) -> int:
        return sum(1 for a in self.actions if a.action == Action.FOLD)


# ---------------------------------------------------------------------------
# Extracteur OCR d'actions
# ---------------------------------------------------------------------------

class ActionOCR:
    """
    Détecte les actions par OCR sur les zones de badge de chaque siège.
    PokerStars affiche "Fold", "Call 40$", "Raise 120$" pendant ~1.5s.
    """

    def __init__(self):
        self._ocr_config = (
            "--psm 7 "
            "-c tessedit_char_whitelist="
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789$€.,- "
        )

    def extract_action_from_zone(self, zone_img: np.ndarray) -> tuple[Action, float, float]:
        """
        OCR sur une zone de badge et parse l'action + montant.
        Retourne (action, montant, confiance).
        """
        try:
            import pytesseract
            from PIL import Image, ImageEnhance

            # Prétraitement pour badge coloré sur fond sombre
            h, w = zone_img.shape[:2]
            large = cv2.resize(zone_img, (w * 3, h * 3),
                               interpolation=cv2.INTER_CUBIC)
            gray  = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            gray  = clahe.apply(gray)
            _, bw = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            pil  = Image.fromarray(bw)
            pil  = ImageEnhance.Sharpness(pil).enhance(2.0)
            text = pytesseract.image_to_string(pil, config=self._ocr_config).strip().lower()

            if not text:
                return Action.UNKNOWN, 0.0, 0.0

            return self._parse_action_text(text)

        except Exception as e:
            log.debug(f"OCR action échoué : {e}")
            return Action.UNKNOWN, 0.0, 0.0

    @staticmethod
    def _parse_action_text(text: str) -> tuple[Action, float, float]:
        """Parse 'raise 120$' → (RAISE, 120.0, 0.9)."""
        text = text.strip().lower()

        # Chercher le montant (ex : "120", "40.5", "1,200")
        amount = 0.0
        amount_match = re.search(r"[\$€]?\s*(\d[\d,. ]*)", text)
        if amount_match:
            try:
                amount = float(amount_match.group(1).replace(",", "").replace(" ", ""))
            except ValueError:
                pass

        # Chercher l'action
        for keyword, action in ACTION_KEYWORDS.items():
            if keyword in text:
                confidence = 0.85 if amount > 0 else 0.75
                return action, amount, confidence

        return Action.UNKNOWN, amount, 0.3


# ---------------------------------------------------------------------------
# Détection visuelle (couleur des badges)
# ---------------------------------------------------------------------------

class ActionColorDetector:
    """
    Détecte les badges d'action par leur couleur HSV.
    Plus rapide que l'OCR, moins précis sur le montant.
    """

    def detect_action_color(self, zone_img: np.ndarray) -> tuple[Action, float]:
        """
        Analyse la couleur dominante d'une zone pour identifier l'action.
        Retourne (action, confiance).
        """
        if zone_img.size == 0:
            return Action.UNKNOWN, 0.0

        hsv = cv2.cvtColor(zone_img, cv2.COLOR_BGR2HSV)
        total_pixels = zone_img.shape[0] * zone_img.shape[1]

        best_action = Action.UNKNOWN
        best_ratio  = 0.0

        for action, color_range in BADGE_COLORS_HSV.items():
            mask  = cv2.inRange(hsv, color_range["low"], color_range["high"])
            ratio = cv2.countNonZero(mask) / max(total_pixels, 1)
            if ratio > best_ratio and ratio > 0.15:
                best_ratio  = ratio
                best_action = action

        confidence = min(best_ratio * 2.5, 0.80)
        return best_action, confidence


# ---------------------------------------------------------------------------
# Détection par différentiel de stack
# ---------------------------------------------------------------------------

class StackDiffDetector:
    """
    Compare les stacks entre deux captures pour déduire les actions.

    Logique :
    - Stack diminue → le joueur a misé / callé / raisé
    - Stack à zéro  → all-in ou fold (croiser avec cartes visibles)
    - Pot augmente  → au moins un joueur a ajouté des jetons
    """

    def __init__(self):
        self._prev_stacks: dict[int, float] = {}
        self._prev_pot:    float = 0.0

    def update(
        self,
        stacks:     dict[int, float],  # {seat: stack_amount}
        pot:        float,
    ) -> list[OpponentAction]:
        """
        Compare les stacks actuels avec les précédents.
        Retourne la liste des actions déduites.
        """
        actions = []

        if not self._prev_stacks:
            self._prev_stacks = stacks.copy()
            self._prev_pot    = pot
            return actions

        pot_diff = pot - self._prev_pot

        for seat, current_stack in stacks.items():
            prev_stack = self._prev_stacks.get(seat, current_stack)
            diff       = prev_stack - current_stack

            if abs(diff) < 0.50:   # moins de 0.50$ de variation → rien
                continue

            if diff > 0:
                # Le stack a diminué → mise, call, raise ou all-in
                if current_stack < 0.10:
                    action = Action.ALL_IN
                    conf   = 0.85
                elif pot_diff > diff * 1.5:
                    action = Action.RAISE    # pot a augmenté + que le diff
                    conf   = 0.75
                else:
                    action = Action.CALL
                    conf   = 0.70
            else:
                # Stack augmenté → gagné le pot ou erreur de lecture
                continue

            actions.append(OpponentAction(
                seat       = seat,
                action     = action,
                amount     = abs(diff),
                confidence = conf,
                method     = "stack_diff",
            ))

        self._prev_stacks = stacks.copy()
        self._prev_pot    = pot
        return actions

    def reset(self) -> None:
        self._prev_stacks.clear()
        self._prev_pot = 0.0


# ---------------------------------------------------------------------------
# Détection visuelle des folds (cartes retournées / siège grisé)
# ---------------------------------------------------------------------------

class FoldVisualDetector:
    """
    Détecte les folds en cherchant les sièges grisés / cartes retournées.
    PokerStars grise le siège et retourne les cartes face-down après un fold.
    """

    def __init__(self):
        self._prev_frame: Optional[np.ndarray] = None

    def detect_folds(
        self,
        frame: np.ndarray,
        seat_positions: dict[int, tuple],   # {seat: (cx_ratio, cy_ratio)}
        img_shape: tuple,
    ) -> list[int]:
        """
        Retourne la liste des sièges qui viennent de folder.
        Compare avec le frame précédent pour détecter les changements.
        """
        h, w = img_shape[:2]
        folded_seats = []

        if self._prev_frame is None:
            self._prev_frame = frame.copy()
            return []

        for seat, (cx_r, cy_r) in seat_positions.items():
            if seat == 0:   # ignorer le siège du joueur
                continue

            cx = int(cx_r * w)
            cy = int(cy_r * h)
            zw = int(BADGE_ZONE_W * w)
            zh = int(BADGE_ZONE_H * h * 2)

            # Zone autour du siège
            x1 = max(0, cx - zw // 2)
            y1 = max(0, cy - zh // 2)
            x2 = min(w, cx + zw // 2)
            y2 = min(h, cy + zh // 2)

            zone_curr = frame[y1:y2, x1:x2]
            zone_prev = self._prev_frame[y1:y2, x1:x2]

            if zone_curr.size == 0 or zone_prev.size == 0:
                continue

            # Calculer le niveau de gris moyen (zone grisée = fold probable)
            gray_curr = cv2.cvtColor(zone_curr, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(zone_prev, cv2.COLOR_BGR2GRAY)

            mean_curr = float(np.mean(gray_curr))
            mean_prev = float(np.mean(gray_prev))

            # Différence de frame (changement visuel brusque)
            diff     = cv2.absdiff(zone_curr, zone_prev)
            diff_pct = float(np.mean(diff)) / 255.0

            # Un fold : zone devient plus grise ET changement visuel net
            became_grayer = (mean_curr - mean_prev) > 15    # grisée
            sudden_change = diff_pct > 0.12                 # changement brusque

            # Vérifier également la saturation (gris = faible saturation)
            hsv_curr = cv2.cvtColor(zone_curr, cv2.COLOR_BGR2HSV)
            sat_mean = float(np.mean(hsv_curr[:, :, 1]))
            low_sat  = sat_mean < 40

            if (became_grayer or low_sat) and sudden_change:
                folded_seats.append(seat)
                log.debug(
                    f"Siège {seat} fold détecté visuellement : "
                    f"gris={mean_curr:.0f} (prev={mean_prev:.0f}) "
                    f"diff={diff_pct:.2%} sat={sat_mean:.0f}"
                )

        self._prev_frame = frame.copy()
        return folded_seats


# ---------------------------------------------------------------------------
# Détecteur principal — orchestre les 4 méthodes
# ---------------------------------------------------------------------------

class OpponentActionDetector:
    """
    Orchestre la détection des actions adverses par cascade :

    1. OCR des badges (le plus fiable si badge visible)
    2. Détection couleur des badges (rapide)
    3. Différentiel de stack (toujours actif en arrière-plan)
    4. Détection visuelle des folds

    Usage :
        detector = OpponentActionDetector()

        # Dans la boucle de capture :
        events = detector.detect(
            frame       = full_screen_img,
            stacks      = {1: 980.0, 2: 1200.0, ...},
            pot         = 120.0,
            stage       = "flop",
        )
        for evt in events:
            for action in evt.actions:
                print(action)
    """

    def __init__(
        self,
        seat_positions: Optional[dict] = None,
        use_ocr:        bool = True,
        use_color:      bool = True,
        use_stack_diff: bool = True,
        use_visual:     bool = True,
    ):
        self.seat_positions = seat_positions or SEAT_POSITIONS_RATIO
        self.use_ocr        = use_ocr
        self.use_color      = use_color
        self.use_stack_diff = use_stack_diff
        self.use_visual     = use_visual

        self._ocr        = ActionOCR()
        self._color      = ActionColorDetector()
        self._stack_diff = StackDiffDetector()
        self._fold_vis   = FoldVisualDetector()

        # Historique des actions de la main en cours
        self._action_history:  list[OpponentAction] = []
        self._active_seats:    set[int]  = set()
        self._folded_seats:    set[int]  = set()
        self._last_actions:    dict[int, OpponentAction] = {}

    # ------------------------------------------------------------------
    # API principale
    # ------------------------------------------------------------------

    def detect(
        self,
        frame:    np.ndarray,
        stacks:   Optional[dict[int, float]] = None,
        pot:      float = 0.0,
        stage:    str   = "",
    ) -> list[OpponentAction]:
        """
        Détecte toutes les actions adverses dans le frame courant.
        Retourne une liste d'OpponentAction (peut être vide).

        frame  : image BGR de la table complète (full screen)
        stacks : {seat_idx: montant_stack} — optionnel mais améliore la précision
        pot    : montant total du pot
        stage  : "preflop" / "flop" / "turn" / "river"
        """
        h, w    = frame.shape[:2]
        actions = []

        # ── 1. OCR + couleur par siège ────────────────────────────────────
        for seat, (cx_r, cy_r) in self.seat_positions.items():
            if seat == 0 or seat in self._folded_seats:
                continue

            cx = int(cx_r * w)
            cy = int(cy_r * h)
            zw = int(BADGE_ZONE_W * w)
            zh = int(BADGE_ZONE_H * h)

            x1 = max(0, cx - zw // 2)
            y1 = max(0, cy - zh)       # badge au-dessus du siège
            x2 = min(w, cx + zw // 2)
            y2 = min(h, cy + zh // 4)

            zone = frame[y1:y2, x1:x2]
            if zone.size == 0:
                continue

            best_action = Action.UNKNOWN
            best_conf   = 0.0
            best_amount = 0.0
            best_method = ""

            # OCR (priorité 1)
            if self.use_ocr:
                ocr_action, ocr_amount, ocr_conf = self._ocr.extract_action_from_zone(zone)
                if ocr_conf > best_conf and ocr_action != Action.UNKNOWN:
                    best_action = ocr_action
                    best_conf   = ocr_conf
                    best_amount = ocr_amount
                    best_method = "ocr"

            # Couleur (priorité 2 si OCR faible)
            if self.use_color and best_conf < 0.6:
                col_action, col_conf = self._color.detect_action_color(zone)
                if col_conf > best_conf and col_action != Action.UNKNOWN:
                    best_action = col_action
                    best_conf   = col_conf
                    best_method = "color"

            # Filtrer les actions non significatives et les doublons récents
            if (best_action != Action.UNKNOWN and
                    best_conf >= 0.45 and
                    not self._is_duplicate(seat, best_action)):

                action_obj = OpponentAction(
                    seat       = seat,
                    action     = best_action,
                    amount     = best_amount,
                    confidence = best_conf,
                    method     = best_method,
                )
                actions.append(action_obj)
                self._register_action(action_obj)

        # ── 2. Différentiel de stack ──────────────────────────────────────
        if self.use_stack_diff and stacks:
            stack_actions = self._stack_diff.update(stacks, pot)
            for sa in stack_actions:
                if not self._is_duplicate(sa.seat, sa.action):
                    actions.append(sa)
                    self._register_action(sa)

        # ── 3. Détection visuelle des folds ───────────────────────────────
        if self.use_visual:
            folded = self._fold_vis.detect_folds(
                frame, self.seat_positions, frame.shape
            )
            for seat in folded:
                if seat not in self._folded_seats:
                    fold_action = OpponentAction(
                        seat       = seat,
                        action     = Action.FOLD,
                        amount     = 0.0,
                        confidence = 0.80,
                        method     = "visual",
                    )
                    actions.append(fold_action)
                    self._register_action(fold_action)
                    self._folded_seats.add(seat)

        if actions:
            log.info(f"Actions détectées ({stage}) : " +
                     " | ".join(str(a) for a in actions))

        return actions

    # ------------------------------------------------------------------
    # Gestion de l'historique
    # ------------------------------------------------------------------

    def _is_duplicate(self, seat: int, action: Action, window_s: float = 2.0) -> bool:
        """Évite de remonter la même action plusieurs fois en quelques secondes."""
        last = self._last_actions.get(seat)
        if last and last.action == action:
            return (time.time() - last.timestamp) < window_s
        return False

    def _register_action(self, action: OpponentAction) -> None:
        self._last_actions[action.seat] = action
        self._action_history.append(action)

    def new_hand(self) -> None:
        """Réinitialise le contexte pour une nouvelle main."""
        self._action_history.clear()
        self._folded_seats.clear()
        self._last_actions.clear()
        self._stack_diff.reset()
        log.info("ActionDetector réinitialisé pour nouvelle main.")

    @property
    def action_history(self) -> list[OpponentAction]:
        return self._action_history.copy()

    @property
    def folded_seats(self) -> set[int]:
        return self._folded_seats.copy()

    def get_aggression_profile(self) -> dict:
        """
        Calcule le profil d'agressivité de chaque adversaire sur la main.
        Utile pour alimenter claude_client.py en temps réel.
        """
        profiles: dict[int, dict] = {}
        for action in self._action_history:
            seat = action.seat
            if seat not in profiles:
                profiles[seat] = {
                    "vpip_actions":  0,
                    "pfr_actions":   0,
                    "aggressive":    0,
                    "passive":       0,
                    "total":         0,
                    "total_invested": 0.0,
                }
            p = profiles[seat]
            p["total"] += 1
            p["total_invested"] += action.amount

            if action.action in (Action.CALL, Action.RAISE, Action.BET, Action.ALL_IN):
                p["vpip_actions"] += 1
            if action.action in (Action.RAISE, Action.BET, Action.ALL_IN):
                p["pfr_actions"]  += 1
                p["aggressive"]   += 1
            elif action.action in (Action.CALL, Action.CHECK):
                p["passive"]      += 1

        # Calculer les ratios
        for seat, p in profiles.items():
            t = max(p["total"], 1)
            p["aggression_factor"] = round(
                p["aggressive"] / max(p["passive"], 1), 2
            )
            p["vpip_est"] = round(p["vpip_actions"] / t * 100, 1)
            p["pfr_est"]  = round(p["pfr_actions"]  / t * 100, 1)

        return profiles


# ---------------------------------------------------------------------------
# Visualisateur debug (affiche les zones de détection sur l'écran)
# ---------------------------------------------------------------------------

def draw_detection_zones(
    frame: np.ndarray,
    detector: OpponentActionDetector,
    recent_actions: list[OpponentAction],
) -> np.ndarray:
    """
    Dessine les zones de badge et les actions détectées sur le frame.
    Utile pour le debug et la calibration.
    """
    vis    = frame.copy()
    h, w   = vis.shape[:2]
    font   = cv2.FONT_HERSHEY_SIMPLEX

    action_colors = {
        Action.FOLD:   (120, 120, 120),
        Action.CALL:   (50,  200,  50),
        Action.RAISE:  (50,   50, 220),
        Action.CHECK:  (200, 150,  50),
        Action.BET:    (50,  150, 220),
        Action.ALL_IN: (30,   30, 220),
        Action.UNKNOWN:(200, 200, 200),
    }

    for seat, (cx_r, cy_r) in detector.seat_positions.items():
        if seat == 0:
            continue

        cx = int(cx_r * w)
        cy = int(cy_r * h)
        zw = int(BADGE_ZONE_W * w)
        zh = int(BADGE_ZONE_H * h)

        # Zone de badge
        x1, y1 = max(0, cx - zw // 2), max(0, cy - zh)
        x2, y2 = min(w, cx + zw // 2), min(h, cy + zh // 4)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (80, 80, 80), 1)

        # Cercle du siège
        color = (100, 100, 100)
        if seat in detector.folded_seats:
            color = (50, 50, 50)
        cv2.circle(vis, (cx, cy), 12, color, 2)
        cv2.putText(vis, str(seat), (cx - 5, cy + 4),
                    font, 0.4, color, 1, cv2.LINE_AA)

    # Actions récentes
    for action in recent_actions:
        seat    = action.seat
        pos     = detector.seat_positions.get(seat, (0.5, 0.5))
        cx      = int(pos[0] * w)
        cy      = int(pos[1] * h)
        color   = action_colors.get(action.action, (200, 200, 200))
        label   = f"{action.action.value}"
        if action.amount > 0:
            label += f" {action.amount:.0f}$"

        cv2.rectangle(vis,
                      (cx - 45, cy - 28),
                      (cx + 45, cy - 8),
                      color, -1)
        cv2.putText(vis, label, (cx - 42, cy - 12),
                    font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


# ---------------------------------------------------------------------------
# Test en direct
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import mss

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Détecteur d'actions adverses")
    parser.add_argument("--screen",  type=int, default=1)
    parser.add_argument("--debug",   action="store_true",
                        help="Afficher les zones de détection dans une fenêtre")
    parser.add_argument("--no-ocr",  action="store_true",
                        help="Désactiver l'OCR (couleur + stack diff uniquement)")
    args = parser.parse_args()

    detector = OpponentActionDetector(
        use_ocr    = not args.no_ocr,
        use_color  = True,
        use_stack_diff = True,
        use_visual = True,
    )

    print("\nDétection d'actions en cours — Ctrl+C pour arrêter\n")

    with mss.mss() as sct:
        monitor = sct.monitors[args.screen]

        while True:
            try:
                raw   = sct.grab(monitor)
                frame = np.array(raw)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                actions = detector.detect(frame, pot=0.0, stage="flop")

                for action in actions:
                    print(f"  {action}")

                if args.debug:
                    vis = draw_detection_zones(frame, detector, actions)
                    small = cv2.resize(vis, (1280, 720))
                    cv2.imshow("Action Detector Debug", small)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                time.sleep(0.5)

            except KeyboardInterrupt:
                print("\nArrêt.")
                profile = detector.get_aggression_profile()
                if profile:
                    print("\nProfil d'agressivité de la main :")
                    for seat, p in profile.items():
                        print(f"  Siège {seat} : AF={p['aggression_factor']} "
                              f"VPIP≈{p['vpip_est']}% PFR≈{p['pfr_est']}%")
                break

    cv2.destroyAllWindows()
