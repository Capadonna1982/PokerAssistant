"""
hud_calibrator.py — Outil visuel de calibration des régions de capture
Dépendances : mss, opencv-python, Pillow, numpy (déjà installés)

Capture l'écran en direct, affiche une fenêtre OpenCV où l'utilisateur
dessine les régions de capture à la souris, puis sauvegarde dans config.json.

Usage :
    python hud_calibrator.py
    python hud_calibrator.py --screen 1      (écran secondaire)
    python hud_calibrator.py --load          (charger config existante)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import mss
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent / "config.json"

# Ordre de calibration des régions (nom → label affiché)
REGIONS_ORDER = [
    ("player_cards", "Cartes du JOUEUR (vos 2 cartes en bas)"),
    ("board",        "BOARD (cartes communes au centre)"),
    ("pot",          "POT (montant total au centre-haut)"),
    ("player_stack", "STACK du joueur (votre stack en bas)"),
    ("current_bet",  "MISE en cours (votre bet actuel)"),
    ("seats",        "ZONE des SIÈGES (toute la table, pour compter joueurs)"),
]

# Couleurs BGR par région
REGION_COLORS = {
    "player_cards": (0,   255,  80),   # vert vif
    "board":        (0,   180, 255),   # orange
    "pot":          (255, 220,   0),   # cyan
    "player_stack": (255,  80, 200),   # rose
    "current_bet":  (80,  200, 255),   # jaune
    "seats":        (180, 180, 180),   # gris
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = 0.45
FONT_MED   = 0.6
FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX


# ---------------------------------------------------------------------------
# Capture d'écran
# ---------------------------------------------------------------------------

def capture_full_screen(monitor_idx: int = 1) -> np.ndarray:
    """Capture l'écran complet et retourne un array BGR."""
    with mss.mss() as sct:
        monitors = sct.monitors
        if monitor_idx >= len(monitors):
            monitor_idx = 1
        monitor = monitors[monitor_idx]
        raw = sct.grab(monitor)
        img = np.array(raw)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


# ---------------------------------------------------------------------------
# État de l'outil
# ---------------------------------------------------------------------------

class CalibratorState:
    """Maintient l'état complet de la session de calibration."""

    def __init__(self, screen_img: np.ndarray, monitor: dict):
        self.screen_img     = screen_img.copy()
        self.display_img    = screen_img.copy()
        self.monitor        = monitor
        self.regions: dict  = {}

        # État du dessin en cours
        self.drawing        = False
        self.start_pt       = (0, 0)
        self.current_pt     = (0, 0)
        self.current_region = 0   # index dans REGIONS_ORDER

        # Zoom et pan
        self.zoom           = 1.0
        self.pan_x          = 0
        self.pan_y          = 0
        self.zoom_origin    = (0, 0)

        # Viewport dimensions (mis à jour à chaque redraw)
        self.win_w          = 0
        self.win_h          = 0

    @property
    def region_name(self) -> str:
        if self.current_region < len(REGIONS_ORDER):
            return REGIONS_ORDER[self.current_region][0]
        return ""

    @property
    def region_label(self) -> str:
        if self.current_region < len(REGIONS_ORDER):
            return REGIONS_ORDER[self.current_region][1]
        return "Terminé"

    @property
    def is_done(self) -> bool:
        return self.current_region >= len(REGIONS_ORDER)

    def screen_to_img(self, x: int, y: int) -> tuple[int, int]:
        """Convertit coordonnées fenêtre → coordonnées image originale."""
        ix = int((x / self.zoom) + self.pan_x)
        iy = int((y / self.zoom) + self.pan_y)
        return ix, iy

    def img_to_screen(self, x: int, y: int) -> tuple[int, int]:
        """Convertit coordonnées image originale → coordonnées fenêtre."""
        sx = int((x - self.pan_x) * self.zoom)
        sy = int((y - self.pan_y) * self.zoom)
        return sx, sy


# ---------------------------------------------------------------------------
# Rendu
# ---------------------------------------------------------------------------

def render(state: CalibratorState) -> np.ndarray:
    """Construit l'image affichée avec overlay, régions et instructions."""
    h, w = state.screen_img.shape[:2]

    # Appliquer zoom et pan
    x1 = int(state.pan_x)
    y1 = int(state.pan_y)
    x2 = min(w, int(state.pan_x + state.win_w / state.zoom))
    y2 = min(h, int(state.pan_y + state.win_h / state.zoom))
    x1, y1 = max(0, x1), max(0, y1)

    crop = state.screen_img[y1:y2, x1:x2]
    if crop.size == 0:
        crop = state.screen_img.copy()

    disp_w = max(1, state.win_w)
    disp_h = max(1, state.win_h)
    canvas = cv2.resize(crop, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

    scale_x = disp_w / max(x2 - x1, 1)
    scale_y = disp_h / max(y2 - y1, 1)

    def to_canvas(ix, iy):
        cx = int((ix - x1) * scale_x)
        cy = int((iy - y1) * scale_y)
        return cx, cy

    # ── Dessiner les régions déjà validées ────────────────────────────
    for name, region in state.regions.items():
        color  = REGION_COLORS.get(name, (200, 200, 200))
        pt1    = to_canvas(region["left"], region["top"])
        pt2    = to_canvas(region["left"] + region["width"],
                            region["top"]  + region["height"])

        # Rectangle semi-transparent
        overlay = canvas.copy()
        cv2.rectangle(overlay, pt1, pt2, color, -1)
        cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)
        cv2.rectangle(canvas, pt1, pt2, color, 2)

        # Label de la région
        label_idx = next((i for i, (n, _) in enumerate(REGIONS_ORDER) if n == name), 0)
        label     = REGIONS_ORDER[label_idx][1].split("(")[0].strip()
        lx, ly    = pt1[0] + 4, pt1[1] + 16
        cv2.rectangle(canvas,
                      (lx - 2, ly - 13),
                      (lx + len(label) * 7 + 4, ly + 4),
                      (0, 0, 0), -1)
        cv2.putText(canvas, label, (lx, ly), FONT, FONT_SMALL, color, 1, cv2.LINE_AA)

    # ── Rectangle en cours de dessin ──────────────────────────────────
    if state.drawing:
        color = REGION_COLORS.get(state.region_name, (255, 255, 255))
        ix1, iy1 = state.screen_to_img(*state.start_pt)
        ix2, iy2 = state.screen_to_img(*state.current_pt)
        pt1 = to_canvas(min(ix1, ix2), min(iy1, iy2))
        pt2 = to_canvas(max(ix1, ix2), max(iy1, iy2))
        overlay = canvas.copy()
        cv2.rectangle(overlay, pt1, pt2, color, -1)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
        cv2.rectangle(canvas, pt1, pt2, color, 2)

        # Dimensions en pixels originaux
        pw = abs(ix2 - ix1)
        ph = abs(iy2 - iy1)
        size_txt = f"{pw} x {ph} px"
        cv2.putText(canvas, size_txt, (pt1[0] + 4, pt2[1] - 6),
                    FONT, FONT_SMALL, color, 1, cv2.LINE_AA)

    # ── Panneau d'instructions (bas de l'écran) ───────────────────────
    panel_h = 90
    panel   = np.zeros((panel_h, disp_w, 3), dtype=np.uint8)
    panel[:] = (20, 20, 30)

    if not state.is_done:
        progress = f"[{state.current_region + 1}/{len(REGIONS_ORDER)}]"
        color    = REGION_COLORS.get(state.region_name, (255, 255, 255))

        cv2.putText(panel, progress, (10, 22),
                    FONT_BOLD, FONT_MED, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(panel, state.region_label, (80, 22),
                    FONT_BOLD, FONT_MED, color, 1, cv2.LINE_AA)

        cv2.putText(panel, "Clic gauche + glisser = dessiner la region",
                    (10, 45), FONT, FONT_SMALL, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(panel, "R = recommencer cette region   Z = zoom molette   Esc = quitter",
                    (10, 63), FONT, FONT_SMALL, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(panel, "S = sauvegarder et quitter   Retour = annuler derniere region",
                    (10, 81), FONT, FONT_SMALL, (120, 120, 120), 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "Toutes les regions sont calibrees !",
                    (10, 28), FONT_BOLD, FONT_MED, (80, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(panel, "S = Sauvegarder config.json   Esc = Quitter sans sauvegarder",
                    (10, 52), FONT, FONT_SMALL, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(panel, "Retour = Revenir a la derniere region",
                    (10, 70), FONT, FONT_SMALL, (120, 120, 120), 1, cv2.LINE_AA)

    # ── Barre de progression ──────────────────────────────────────────
    bar_w   = int(disp_w * state.current_region / len(REGIONS_ORDER))
    bar_col = REGION_COLORS.get(state.region_name, (80, 80, 80))
    cv2.rectangle(panel, (0, 85), (bar_w, panel_h), bar_col, -1)

    # ── Indicateur zoom ───────────────────────────────────────────────
    zoom_txt = f"Zoom: {state.zoom:.1f}x"
    cv2.putText(canvas, zoom_txt, (disp_w - 100, 20),
                FONT, FONT_SMALL, (180, 180, 180), 1, cv2.LINE_AA)

    return np.vstack([canvas, panel])


# ---------------------------------------------------------------------------
# Callbacks souris
# ---------------------------------------------------------------------------

def make_mouse_callback(state: CalibratorState):
    def callback(event, x, y, flags, param):
        if state.is_done and event != cv2.EVENT_MOUSEWHEEL:
            if event == cv2.EVENT_LBUTTONDOWN:
                # Permettre de redessiner en mode terminé
                pass

        if event == cv2.EVENT_LBUTTONDOWN:
            state.drawing  = True
            state.start_pt = (x, y)
            state.current_pt = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            state.current_pt = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if state.drawing:
                state.drawing = False
                ix1, iy1 = state.screen_to_img(*state.start_pt)
                ix2, iy2 = state.screen_to_img(x, y)

                left   = min(ix1, ix2)
                top    = min(iy1, iy2)
                width  = abs(ix2 - ix1)
                height = abs(iy2 - iy1)

                if width > 10 and height > 5:
                    # Convertir en coordonnées absolues écran
                    name = state.region_name
                    state.regions[name] = {
                        "left":   left   + state.monitor["left"],
                        "top":    top    + state.monitor["top"],
                        "width":  width,
                        "height": height,
                    }
                    log.info(
                        f"Région '{name}' : "
                        f"left={state.regions[name]['left']} "
                        f"top={state.regions[name]['top']} "
                        f"w={width} h={height}"
                    )
                    state.current_region += 1
                else:
                    log.warning("Région trop petite — recommencez.")

        elif event == cv2.EVENT_MOUSEWHEEL:
            # Zoom centré sur le curseur
            zoom_step = 0.15
            if flags > 0:
                new_zoom = min(state.zoom + zoom_step, 8.0)
            else:
                new_zoom = max(state.zoom - zoom_step, 0.5)

            # Ajuster le pan pour garder le point sous le curseur fixe
            h, w = state.screen_img.shape[:2]
            img_x = x / state.zoom + state.pan_x
            img_y = y / state.zoom + state.pan_y
            state.zoom  = new_zoom
            state.pan_x = img_x - x / state.zoom
            state.pan_y = img_y - y / state.zoom
            state.pan_x = max(0, min(state.pan_x, w - 1))
            state.pan_y = max(0, min(state.pan_y, h - 1))

    return callback


# ---------------------------------------------------------------------------
# Sauvegarde / chargement
# ---------------------------------------------------------------------------

def save_config(regions: dict, path: Path = CONFIG_PATH) -> None:
    """Sauvegarde les régions dans config.json."""
    config = {"regions": regions}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    log.info(f"Configuration sauvegardée dans {path}")
    print("\n" + "=" * 55)
    print("  CONFIG.JSON SAUVEGARDÉ")
    print("=" * 55)
    for name, region in regions.items():
        print(f"  {name:<16} : "
              f"left={region['left']:4d}  top={region['top']:4d}  "
              f"w={region['width']:4d}  h={region['height']:4d}")
    print("=" * 55)


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Charge une configuration existante."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("regions", {})


def build_default_regions(monitor: dict) -> dict:
    """
    Génère des régions par défaut estimées pour 1920×1080.
    Utilisé comme point de départ si aucune config n'existe.
    """
    mw = monitor["width"]
    mh = monitor["height"]
    ml = monitor["left"]
    mt = monitor["top"]

    # Ratios basés sur la mise en page PokerStars 1920×1080
    return {
        "player_cards": {
            "left": ml + int(mw * 0.416), "top": mt + int(mh * 0.768),
            "width": int(mw * 0.167),     "height": int(mh * 0.111),
        },
        "board": {
            "left": ml + int(mw * 0.291), "top": mt + int(mh * 0.407),
            "width": int(mw * 0.417),     "height": int(mh * 0.120),
        },
        "pot": {
            "left": ml + int(mw * 0.390), "top": mt + int(mh * 0.365),
            "width": int(mw * 0.218),     "height": int(mh * 0.046),
        },
        "player_stack": {
            "left": ml + int(mw * 0.442), "top": mt + int(mh * 0.879),
            "width": int(mw * 0.115),     "height": int(mh * 0.037),
        },
        "current_bet": {
            "left": ml + int(mw * 0.442), "top": mt + int(mh * 0.703),
            "width": int(mw * 0.115),     "height": int(mh * 0.037),
        },
        "seats": {
            "left": ml + int(mw * 0.052), "top": mt + int(mh * 0.092),
            "width": int(mw * 0.896),     "height": int(mh * 0.740),
        },
    }


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------

def run_calibrator(monitor_idx: int = 1, load_existing: bool = False) -> Optional[dict]:
    """
    Lance l'outil de calibration visuelle.
    Retourne le dictionnaire de régions calibrées, ou None si annulé.
    """
    print("\n" + "=" * 55)
    print("  ♠ POKER HUD — CALIBRATEUR DE RÉGIONS")
    print("=" * 55)
    print(f"  Capture de l'écran {monitor_idx}…")

    # Capture initiale
    with mss.mss() as sct:
        monitors = sct.monitors
        if monitor_idx >= len(monitors):
            log.warning(f"Écran {monitor_idx} non trouvé. Utilisation de l'écran 1.")
            monitor_idx = 1
        monitor = monitors[monitor_idx]

    screen_img = capture_full_screen(monitor_idx)
    h, w       = screen_img.shape[:2]
    log.info(f"Écran capturé : {w}×{h} px")

    # Taille de fenêtre (adapter à l'écran disponible)
    win_h = min(h, 900)
    win_w = min(w, int(win_h * w / h))

    state         = CalibratorState(screen_img, monitor)
    state.win_w   = win_w
    state.win_h   = win_h - 90   # réserver 90px pour le panneau

    # Pré-charger une config existante
    if load_existing:
        existing = load_config()
        if existing:
            # Convertir en coordonnées relatives à l'écran capturé
            for name, region in existing.items():
                state.regions[name] = region
            state.current_region = len(existing)
            log.info(f"Config existante chargée : {len(existing)} régions.")

    # Créer la fenêtre
    win_name = "♠ Calibrateur HUD — Dessinez les régions"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, win_w, win_h + 10)
    cv2.setMouseCallback(win_name, make_mouse_callback(state))

    print("\n  MODE D'EMPLOI")
    print("  ─────────────────────────────────────────")
    print("  1. Lancez PokerStars et ouvrez une table")
    print("  2. Alt+Tab pour revenir ici")
    print("  3. Dessinez chaque zone à la souris")
    print("  4. Appuyez sur S pour sauvegarder")
    print("  ─────────────────────────────────────────\n")

    last_capture = time.time()
    result       = None

    while True:
        # Rafraîchir la capture toutes les 3s (pour voir les changements de table)
        if time.time() - last_capture > 3.0:
            new_img = capture_full_screen(monitor_idx)
            state.screen_img = new_img
            last_capture = time.time()

        frame = render(state)
        cv2.imshow(win_name, frame)

        key = cv2.waitKey(30) & 0xFF

        # S — Sauvegarder
        if key == ord("s") or key == ord("S"):
            if state.regions:
                save_config(state.regions)
                result = state.regions
                break
            else:
                log.warning("Aucune région dessinée — impossible de sauvegarder.")

        # Esc — Quitter sans sauvegarder
        elif key == 27:
            log.info("Calibration annulée.")
            break

        # R — Recommencer la région actuelle
        elif key == ord("r") or key == ord("R"):
            if state.region_name in state.regions:
                del state.regions[state.region_name]
                log.info(f"Région '{state.region_name}' réinitialisée.")

        # Backspace — Annuler la dernière région
        elif key == 8:
            if state.current_region > 0:
                state.current_region -= 1
                name = REGIONS_ORDER[state.current_region][0]
                if name in state.regions:
                    del state.regions[name]
                    log.info(f"Région '{name}' supprimée.")

        # D — Régions par défaut (estimation automatique)
        elif key == ord("d") or key == ord("D"):
            defaults = build_default_regions(monitor)
            state.regions        = defaults
            state.current_region = len(REGIONS_ORDER)
            log.info("Régions par défaut appliquées — vérifiez et ajustez si nécessaire.")

        # F5 — Rafraîchir la capture maintenant
        elif key == 116:   # F5 en OpenCV
            state.screen_img = capture_full_screen(monitor_idx)
            log.info("Capture rafraîchie.")

        # Fenêtre fermée
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    return result


# ---------------------------------------------------------------------------
# Vérification rapide (preview des régions)
# ---------------------------------------------------------------------------

def preview_regions(regions: dict, monitor_idx: int = 1) -> None:
    """
    Affiche un aperçu des régions calibrées avec leur contenu OCR simulé.
    Utile pour vérifier que les régions capturent bien le bon contenu.
    """
    screen_img = capture_full_screen(monitor_idx)

    with mss.mss() as sct:
        monitor = sct.monitors[monitor_idx]

    h, w = screen_img.shape[:2]
    preview = screen_img.copy()

    for name, region in regions.items():
        color = REGION_COLORS.get(name, (200, 200, 200))

        # Convertir en coordonnées relatives si nécessaire
        left   = region["left"]   - monitor["left"]
        top    = region["top"]    - monitor["top"]
        width  = region["width"]
        height = region["height"]

        left   = max(0, min(left,  w - 1))
        top    = max(0, min(top,   h - 1))
        width  = min(width,  w - left)
        height = min(height, h - top)

        pt1 = (left, top)
        pt2 = (left + width, top + height)

        overlay = preview.copy()
        cv2.rectangle(overlay, pt1, pt2, color, -1)
        cv2.addWeighted(overlay, 0.2, preview, 0.8, 0, preview)
        cv2.rectangle(preview, pt1, pt2, color, 2)

        label = name.replace("_", " ").upper()
        cv2.putText(preview, label,
                    (left + 4, top + 18),
                    FONT_BOLD, FONT_SMALL, color, 1, cv2.LINE_AA)

        # Afficher le crop de chaque région
        crop = screen_img[top:top+height, left:left+width]
        if crop.size > 0:
            crop_win = f"Preview — {name}"
            cv2.namedWindow(crop_win, cv2.WINDOW_NORMAL)
            crop_disp = cv2.resize(crop, (max(crop.shape[1]*2, 200),
                                          max(crop.shape[0]*2, 60)))
            cv2.imshow(crop_win, crop_disp)

    win_name = "Vérification des régions (appuyez sur une touche pour fermer)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    disp_h = min(h, 800)
    disp_w = int(disp_h * w / h)
    cv2.resizeWindow(win_name, disp_w, disp_h)
    cv2.imshow(win_name, cv2.resize(preview, (disp_w, disp_h)))

    print("\nAperçu affiché — appuyez sur une touche pour fermer.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Test de capture OCR sur une région
# ---------------------------------------------------------------------------

def test_region_ocr(region_name: str) -> None:
    """
    Capture une région calibrée et tente l'OCR dessus.
    Utile pour vérifier que les coordonnées sont correctes.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        log.error("pytesseract ou Pillow non installé.")
        return

    regions = load_config()
    if region_name not in regions:
        log.error(f"Région '{region_name}' non trouvée dans config.json")
        return

    region = regions[region_name]
    with mss.mss() as sct:
        raw  = sct.grab(region)
        img  = np.array(raw)
        img  = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Prétraitement identique à capture.py
    scale = 2.5
    ih, iw = img.shape[:2]
    img = cv2.resize(img, (int(iw * scale), int(ih * scale)),
                     interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    pil_img = Image.fromarray(thresh)
    text = pytesseract.image_to_string(pil_img, config="--psm 7").strip()

    print(f"\n  OCR [{region_name}] : '{text}'")
    cv2.imshow(f"OCR test — {region_name}", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="♠ Calibrateur visuel des régions de capture HUD"
    )
    parser.add_argument("--screen",  type=int,  default=1,
                        help="Index de l'écran à capturer (défaut : 1)")
    parser.add_argument("--load",    action="store_true",
                        help="Charger la configuration existante et continuer")
    parser.add_argument("--preview", action="store_true",
                        help="Afficher un aperçu de la config actuelle")
    parser.add_argument("--test",    type=str,  default=None,
                        metavar="REGION",
                        help="Tester l'OCR sur une région (ex: --test player_cards)")
    parser.add_argument("--default", action="store_true",
                        help="Générer et sauvegarder les régions par défaut (1920×1080)")
    args = parser.parse_args()

    if args.preview:
        regions = load_config()
        if not regions:
            log.error("Aucune config trouvée. Lancez d'abord la calibration.")
            sys.exit(1)
        preview_regions(regions, args.screen)
        return

    if args.test:
        test_region_ocr(args.test)
        return

    if args.default:
        with mss.mss() as sct:
            monitor = sct.monitors[args.screen]
        defaults = build_default_regions(monitor)
        save_config(defaults)
        print("\nRégions par défaut générées. Lancez --preview pour vérifier.")
        return

    # Lancement normal de la calibration
    regions = run_calibrator(
        monitor_idx    = args.screen,
        load_existing  = args.load,
    )

    if regions:
        print("\n  Calibration terminée avec succès !")
        print(f"  {len(regions)} régions sauvegardées dans config.json")
        print("\n  Relancez main.py pour utiliser la nouvelle configuration.")

        # Proposer un aperçu
        try:
            ans = input("\n  Voulez-vous un aperçu de vérification ? (o/n) : ")
            if ans.lower() in ("o", "oui", "y", "yes"):
                preview_regions(regions, args.screen)
        except (EOFError, KeyboardInterrupt):
            pass
    else:
        print("\n  Calibration annulée — config.json non modifié.")


if __name__ == "__main__":
    main()
