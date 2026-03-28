"""
capture.py — Module de capture d'écran et OCR pour l'assistant poker PokerStars
Dépendances : mss, Pillow, pytesseract, opencv-python, numpy
Installation  : pip install mss Pillow pytesseract opencv-python numpy
               + Tesseract OCR : https://github.com/UB-Mannheim/tesseract/wiki
"""

import re
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import cv2
import mss
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

# ---------------------------------------------------------------------------
# Configuration du logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chemin vers tesseract (Windows — adapter si Linux/macOS)
# ---------------------------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------
SUITS = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]

@dataclass
class GameState:
    """État complet d'une main détecté sur l'écran."""
    player_cards: list[str]          = field(default_factory=list)   # ex. ["Ks", "7h"]
    board_cards:  list[str]          = field(default_factory=list)   # ex. ["Ah", "3d", "9c"]
    pot:          float              = 0.0
    player_stack: float              = 0.0
    current_bet:  float              = 0.0
    num_players:  int                = 0
    stage:        str                = "preflop"                      # preflop/flop/turn/river
    raw_screenshot: Optional[np.ndarray] = field(default=None, repr=False)
    timestamp:    float              = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "player_cards": self.player_cards,
            "board_cards":  self.board_cards,
            "pot":          self.pot,
            "player_stack": self.player_stack,
            "current_bet":  self.current_bet,
            "num_players":  self.num_players,
            "stage":        self.stage,
            "timestamp":    self.timestamp,
        }


# ---------------------------------------------------------------------------
# Régions de capture (à calibrer selon votre résolution)
# Ces valeurs ciblent une table PokerStars en 1920×1080 plein écran.
# Chargez config.json pour surcharger.
# ---------------------------------------------------------------------------
DEFAULT_REGIONS = {
    # Région complète de la table (fallback)
    "table":        {"top": 0,    "left": 0,    "width": 1920, "height": 1080},

    # Cartes du joueur (bas-centre)
    "player_cards": {"top": 830,  "left": 800,  "width": 320,  "height": 120},

    # Board (centre de la table)
    "board":        {"top": 440,  "left": 560,  "width": 800,  "height": 130},

    # Pot (au-dessus du board)
    "pot":          {"top": 395,  "left": 750,  "width": 420,  "height": 50},

    # Stack du joueur (bas)
    "player_stack": {"top": 950,  "left": 850,  "width": 220,  "height": 40},

    # Mise actuelle (centre-bas)
    "current_bet":  {"top": 760,  "left": 850,  "width": 220,  "height": 40},

    # Zone des sièges occupés (pour compter les joueurs)
    "seats":        {"top": 100,  "left": 100,  "width": 1720, "height": 800},
}

CONFIG_PATH = Path(__file__).parent / "config.json"


def load_regions() -> dict:
    """Charge les régions depuis config.json, sinon utilise les valeurs par défaut."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            log.info("Régions chargées depuis config.json")
            return cfg.get("regions", DEFAULT_REGIONS)
        except Exception as e:
            log.warning(f"Impossible de lire config.json : {e}. Utilisation des valeurs par défaut.")
    return DEFAULT_REGIONS


# ---------------------------------------------------------------------------
# Capture d'écran
# ---------------------------------------------------------------------------
class ScreenCapture:
    """Capture une région de l'écran avec mss (performant, cross-platform)."""

    def __init__(self, regions: Optional[dict] = None):
        self.regions = regions or load_regions()
        self._sct = mss.mss()

    def capture_region(self, region_name: str) -> np.ndarray:
        """Retourne un ndarray BGR de la région demandée."""
        region = self.regions.get(region_name)
        if region is None:
            raise ValueError(f"Région inconnue : '{region_name}'")
        raw = self._sct.grab(region)
        img = np.array(raw)                       # BGRA
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def capture_table(self) -> np.ndarray:
        return self.capture_region("table")

    def save_debug(self, img: np.ndarray, name: str = "debug.png") -> None:
        cv2.imwrite(name, img)
        log.debug(f"Image debug sauvegardée : {name}")


# ---------------------------------------------------------------------------
# Pré-traitement image pour l'OCR
# ---------------------------------------------------------------------------
def preprocess_for_ocr(img_bgr: np.ndarray, scale: float = 2.5) -> Image.Image:
    """
    Pipeline de prétraitement optimisé pour le texte PokerStars :
      1. Agrandissement ×2.5 (tesseract préfère les grandes images)
      2. Niveaux de gris
      3. CLAHE (contraste adaptatif)
      4. Seuillage Otsu
      5. Légère netteté via Pillow
    """
    # Agrandissement
    h, w = img_bgr.shape[:2]
    img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)

    # Niveaux de gris
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Contraste adaptatif
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Seuillage Otsu
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Conversion Pillow + netteté
    pil_img = Image.fromarray(thresh)
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(2.0)
    return pil_img


def ocr_text(img_bgr: np.ndarray, config: str = "--psm 7 -c tessedit_char_whitelist=0123456789.$ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ") -> str:
    """Extrait le texte brut d'une région pré-traitée."""
    pil_img = preprocess_for_ocr(img_bgr)
    text = pytesseract.image_to_string(pil_img, config=config).strip()
    log.debug(f"OCR brut : {repr(text)}")
    return text


# ---------------------------------------------------------------------------
# Détection des cartes
# ---------------------------------------------------------------------------

# Regex qui reconnaît les formats courants : "As", "Th", "2d", "Kc"
_CARD_RE = re.compile(r"\b([2-9TJQKA][shdc])\b", re.IGNORECASE)

# Dictionnaire de correction OCR classique (O→0, l→1, etc.)
_OCR_FIXES = {
    "0": "O",  # chiffre zéro → non utilisé en rank
    "1": "A",  # rare mais arrive pour As
    "I": "1",  # I capital → ignoré (pas de rang '1')
    "o": "0",  # 'o' minuscule
}

def _normalize_card(raw: str) -> Optional[str]:
    """
    Tente de normaliser un token OCR en carte valide (ex 'ks' → 'Ks').
    Retourne None si non reconnu.
    """
    if len(raw) < 2:
        return None
    rank = raw[0].upper()
    suit = raw[-1].lower()
    if rank not in [r.upper() for r in RANKS]:
        return None
    if suit not in SUITS:
        return None
    return f"{rank}{suit}"


def extract_cards_from_text(text: str) -> list[str]:
    """Extrait les cartes valides d'une chaîne OCR."""
    tokens = _CARD_RE.findall(text)
    cards = []
    for t in tokens:
        card = _normalize_card(t)
        if card and card not in cards:
            cards.append(card)
    return cards


def detect_cards_by_color(img_bgr: np.ndarray, region_name: str = "board") -> list[str]:
    """
    Détection complémentaire par couleur HSV :
    - Cartes rouges (cœur ♥, carreau ♦) : teinte rouge HSV
    - Cartes noires (pique ♠, trèfle ♣) : teinte très sombre
    Retourne une liste ordonnée de positions (utile pour valider l'ordre du board).
    Note : méthode de fallback — l'OCR reste le canal principal.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Masque rouge (deux plages HSV)
    mask_red1 = cv2.inRange(hsv, np.array([0, 100, 100]),  np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    mask_red  = cv2.bitwise_or(mask_red1, mask_red2)

    # Masque blanc (fond des cartes)
    mask_white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))

    # Contours des zones blanches (chaque carte = un rectangle blanc)
    contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect = w / max(h, 1)
        # Filtre sur taille et forme d'une carte (~70×100px à l'écran)
        if 3000 < area < 20000 and 0.5 < aspect < 0.9:
            card_regions.append((x, y, w, h))

    card_regions.sort(key=lambda r: r[0])  # ordre gauche→droite
    log.debug(f"Régions cartes détectées par couleur ({region_name}) : {len(card_regions)}")
    return card_regions  # retourne les bbox, pas encore les valeurs


# ---------------------------------------------------------------------------
# Extraction des valeurs numériques (pot, stack, mise)
# ---------------------------------------------------------------------------

_AMOUNT_RE = re.compile(r"[\$€]?\s*(\d[\d,. ]*\d|\d+)")

def extract_amount(text: str) -> float:
    """Extrait le premier montant numérique d'une chaîne OCR."""
    text = text.replace(",", ".").replace(" ", "")
    match = _AMOUNT_RE.search(text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass
    return 0.0


def extract_num_players(img_bgr: np.ndarray) -> int:
    """
    Détecte le nombre de joueurs actifs en cherchant les avatars/cercles de siège.
    Méthode : détection de cercles avec HoughCircles dans la zone 'seats'.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=80,
        param1=50,
        param2=30,
        minRadius=25,
        maxRadius=55,
    )
    if circles is not None:
        n = len(circles[0])
        log.debug(f"Cercles (sièges) détectés : {n}")
        return min(n, 9)
    return 0


# ---------------------------------------------------------------------------
# Détection du stage (preflop / flop / turn / river)
# ---------------------------------------------------------------------------

def detect_stage(board_cards: list[str]) -> str:
    n = len(board_cards)
    if n == 0:
        return "preflop"
    elif n == 3:
        return "flop"
    elif n == 4:
        return "turn"
    elif n >= 5:
        return "river"
    return "preflop"


# ---------------------------------------------------------------------------
# Extracteur principal
# ---------------------------------------------------------------------------
class PokerExtractor:
    """
    Orchestre la capture et l'extraction de l'état de jeu complet.
    Usage :
        extractor = PokerExtractor()
        state = extractor.extract()
        print(state.to_dict())
    """

    def __init__(self, regions: Optional[dict] = None, debug: bool = False):
        self.capture  = ScreenCapture(regions)
        self.debug    = debug
        self._prev_state: Optional[GameState] = None

    def extract(self) -> GameState:
        """
        Capture et analyse l'état complet de la table.
        Retourne un GameState mis à jour.
        """
        state = GameState()

        # --- Cartes du joueur ---
        try:
            img = self.capture.capture_region("player_cards")
            if self.debug:
                self.capture.save_debug(img, "debug_player_cards.png")
            text = ocr_text(img)
            state.player_cards = extract_cards_from_text(text)
            log.info(f"Cartes joueur : {state.player_cards}")
        except Exception as e:
            log.error(f"Erreur capture cartes joueur : {e}")

        # --- Board ---
        try:
            img = self.capture.capture_region("board")
            if self.debug:
                self.capture.save_debug(img, "debug_board.png")
            text = ocr_text(img)
            state.board_cards = extract_cards_from_text(text)
            log.info(f"Board : {state.board_cards}")
        except Exception as e:
            log.error(f"Erreur capture board : {e}")

        # --- Pot ---
        try:
            img = self.capture.capture_region("pot")
            text = ocr_text(img)
            state.pot = extract_amount(text)
            log.info(f"Pot : {state.pot}")
        except Exception as e:
            log.error(f"Erreur capture pot : {e}")

        # --- Stack joueur ---
        try:
            img = self.capture.capture_region("player_stack")
            text = ocr_text(img)
            state.player_stack = extract_amount(text)
            log.info(f"Stack : {state.player_stack}")
        except Exception as e:
            log.error(f"Erreur capture stack : {e}")

        # --- Mise en cours ---
        try:
            img = self.capture.capture_region("current_bet")
            text = ocr_text(img)
            state.current_bet = extract_amount(text)
            log.info(f"Mise courante : {state.current_bet}")
        except Exception as e:
            log.error(f"Erreur capture mise : {e}")

        # --- Nombre de joueurs ---
        try:
            img = self.capture.capture_region("seats")
            state.num_players = extract_num_players(img)
            log.info(f"Joueurs actifs : {state.num_players}")
        except Exception as e:
            log.error(f"Erreur capture sièges : {e}")

        # --- Stage ---
        state.stage = detect_stage(state.board_cards)

        self._prev_state = state
        return state

    def has_changed(self, new_state: GameState) -> bool:
        """Vérifie si l'état a changé depuis la dernière capture (évite les appels API inutiles)."""
        if self._prev_state is None:
            return True
        return (
            new_state.player_cards != self._prev_state.player_cards
            or new_state.board_cards  != self._prev_state.board_cards
            or new_state.pot          != self._prev_state.pot
            or new_state.current_bet  != self._prev_state.current_bet
        )


# ---------------------------------------------------------------------------
# Boucle de capture continue
# ---------------------------------------------------------------------------

def run_capture_loop(interval: float = 1.5, debug: bool = False):
    """
    Boucle principale : capture l'état toutes les `interval` secondes.
    Yield un GameState à chaque changement détecté.

    Usage dans main.py :
        for state in run_capture_loop():
            conseil = analyse(state)
            overlay.update(conseil)
    """
    extractor = PokerExtractor(debug=debug)
    log.info(f"Boucle de capture démarrée (intervalle : {interval}s)")

    while True:
        try:
            state = extractor.extract()
            if extractor.has_changed(state):
                log.info(f"Changement détecté — stage : {state.stage}")
                yield state
            else:
                log.debug("Pas de changement, attente...")
        except KeyboardInterrupt:
            log.info("Arrêt de la boucle de capture.")
            break
        except Exception as e:
            log.error(f"Erreur dans la boucle : {e}")
        time.sleep(interval)


# ---------------------------------------------------------------------------
# Utilitaire de calibration (à lancer une fois pour ajuster les régions)
# ---------------------------------------------------------------------------

def calibrate_regions():
    """
    Capture la table entière et affiche une fenêtre interactive pour
    délimiter les régions manuellement (sauvegarde dans config.json).
    """
    cap = ScreenCapture()
    table_img = cap.capture_table()

    regions = {}
    current_region = {"name": None}

    def on_select(event, x, y, flags, param):
        pass  # Implémentation ROI interactive si besoin

    print("=== Calibration ===")
    print("Appuyez sur 'q' pour quitter, 's' pour sauvegarder les régions.")
    cv2.imshow("Table PokerStars — Calibration", table_img)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            with open(CONFIG_PATH, "w") as f:
                json.dump({"regions": DEFAULT_REGIONS}, f, indent=2)
            print(f"Régions sauvegardées dans {CONFIG_PATH}")
            break

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test du module de capture poker")
    parser.add_argument("--debug",     action="store_true", help="Sauvegarder les images de debug")
    parser.add_argument("--calibrate", action="store_true", help="Lancer l'outil de calibration")
    parser.add_argument("--once",      action="store_true", help="Capturer une seule fois et afficher l'état")
    args = parser.parse_args()

    if args.calibrate:
        calibrate_regions()
    elif args.once:
        extractor = PokerExtractor(debug=args.debug)
        state = extractor.extract()
        import json as _json
        print("\n=== État détecté ===")
        print(_json.dumps(state.to_dict(), indent=2, ensure_ascii=False))
    else:
        print("Boucle continue (Ctrl+C pour arrêter) :")
        for game_state in run_capture_loop(interval=1.5, debug=args.debug):
            print(json.dumps(game_state.to_dict(), indent=2, ensure_ascii=False))
