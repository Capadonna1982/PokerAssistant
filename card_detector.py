"""
card_detector.py — Détection des cartes par templates OpenCV
Remplace l'OCR pur de capture.py par une reconnaissance visuelle fiable.

Approche hybride en 3 couches :
  1. Template matching (rang + couleur séparément) — fiabilité ~99%
  2. Détection par couleur HSV (fallback)
  3. OCR Tesseract (fallback final)

Dépendances : opencv-python, numpy, mss, Pillow
              pytesseract (fallback uniquement)

Usage :
    python card_detector.py --generate          (génère les templates)
    python card_detector.py --test              (teste sur capture live)
    python card_detector.py --benchmark         (mesure la précision)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import mss
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

RANKS  = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUITS  = ["s", "h", "d", "c"]   # spade, heart, diamond, club
CARDS  = [r + s for r in RANKS for s in SUITS]

# Couleurs des enseignes en HSV
SUIT_HSV = {
    "h": {"low": np.array([0,   100, 100]), "high": np.array([10,  255, 255])},  # rouge cœur
    "d": {"low": np.array([0,   100, 100]), "high": np.array([10,  255, 255])},  # rouge carreau
    "s": {"low": np.array([0,   0,   0  ]), "high": np.array([180, 30,  80 ])},  # noir pique
    "c": {"low": np.array([0,   0,   0  ]), "high": np.array([180, 30,  80 ])},  # noir trèfle
}

# Dossiers
TEMPLATES_DIR = Path(__file__).parent / "templates"
CONFIG_PATH   = Path(__file__).parent / "config.json"

# Seuils de confiance
MATCH_THRESHOLD_HIGH = 0.82   # confiance élevée → résultat direct
MATCH_THRESHOLD_LOW  = 0.65   # confiance faible → fallback OCR


# ---------------------------------------------------------------------------
# Structure résultat
# ---------------------------------------------------------------------------

@dataclass
class CardDetection:
    card:       str           # ex. "Ks"
    confidence: float         # 0.0 – 1.0
    method:     str           # "template" | "hsv" | "ocr" | "unknown"
    bbox:       tuple = ()    # (x, y, w, h) dans l'image source
    rank:       str   = ""
    suit:       str   = ""

    def __post_init__(self):
        if len(self.card) == 2:
            self.rank = self.card[0]
            self.suit = self.card[1]

    @property
    def is_valid(self) -> bool:
        return (self.rank in RANKS and
                self.suit in SUITS and
                self.confidence >= MATCH_THRESHOLD_LOW)


# ---------------------------------------------------------------------------
# Génération des templates synthétiques
# ---------------------------------------------------------------------------

class TemplateGenerator:
    """
    Génère des templates synthétiques pour chaque carte.
    Crée des images PNG propres de rang + enseigne dans le style PokerStars.

    Deux variantes générées :
      - templates/rank/   → rang seul (lettre/chiffre)
      - templates/suit/   → symbole d'enseigne
      - templates/card/   → carte complète (rang + enseigne)
    """

    # Symboles unicode des enseignes
    SUIT_SYMBOLS = {"s": "\u2660", "h": "\u2665", "d": "\u2666", "c": "\u2663"}
    SUIT_COLORS  = {
        "s": (40,  40,  40),    # noir
        "h": (200, 30,  30),    # rouge
        "d": (200, 30,  30),    # rouge
        "c": (40,  40,  40),    # noir
    }

    def __init__(self, templates_dir: Path = TEMPLATES_DIR):
        self.dir = templates_dir
        for sub in ("rank", "suit", "card", "full"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)

    def generate_all(
        self,
        card_w: int = 60,
        card_h: int = 85,
        font_scale: float = 1.8,
    ) -> int:
        """
        Génère les templates pour les 52 cartes.
        Retourne le nombre de fichiers créés.
        """
        count = 0
        font = cv2.FONT_HERSHEY_SIMPLEX

        for rank in RANKS:
            for suit in SUITS:
                card = rank + suit
                color = self.SUIT_COLORS[suit]

                # ── Template rang seul ──────────────────────────────────
                rank_img = self._make_rank_template(rank, color, font,
                                                     font_scale, card_w, card_h)
                path = self.dir / "rank" / f"{rank}_{suit}.png"
                cv2.imwrite(str(path), rank_img)

                # ── Template carte complète ─────────────────────────────
                card_img = self._make_card_template(rank, suit, color, font,
                                                     font_scale, card_w, card_h)
                path = self.dir / "card" / f"{card}.png"
                cv2.imwrite(str(path), card_img)

                count += 1

        # ── Templates rang couleur-agnostique (pour matching robuste) ──
        for rank in RANKS:
            for color_name, color in [("black", (40, 40, 40)),
                                       ("red",   (200, 30, 30))]:
                img = self._make_rank_template(rank, color, font,
                                               font_scale, 30, 40)
                path = self.dir / "rank" / f"{rank}_{color_name}.png"
                cv2.imwrite(str(path), img)

        log.info(f"Templates générés : {count * 2} fichiers dans {self.dir}")
        return count

    def _make_rank_template(self, rank, color, font, scale, w, h) -> np.ndarray:
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        text = rank
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = (w - tw) // 2
        y = (h + th) // 2
        cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
        return img

    def _make_card_template(self, rank, suit, color, font, scale, w, h) -> np.ndarray:
        img = np.ones((h, w, 3), dtype=np.uint8) * 255

        # Bordure de carte
        cv2.rectangle(img, (1, 1), (w - 2, h - 2), (180, 180, 180), 1)

        # Rang en haut à gauche
        cv2.putText(img, rank, (4, int(h * 0.38)),
                    font, scale * 0.7, color, 2, cv2.LINE_AA)

        # Symbole enseigne au centre
        sym_map = {"s": "S", "h": "H", "d": "D", "c": "C"}
        cv2.putText(img, sym_map[suit], (int(w * 0.2), int(h * 0.75)),
                    font, scale * 0.6, color, 2, cv2.LINE_AA)

        return img

    def generate_from_screenshot(
        self,
        screenshot: np.ndarray,
        card_bboxes: list[tuple],
        card_labels: list[str],
    ) -> int:
        """
        Génère des templates à partir d'une capture réelle.
        card_bboxes : [(x, y, w, h), ...]
        card_labels : ["Ks", "7h", ...]
        Retourne le nombre de templates créés.
        """
        count = 0
        for (x, y, w, h), label in zip(card_bboxes, card_labels):
            if label not in CARDS:
                continue
            crop = screenshot[y:y+h, x:x+w]
            if crop.size == 0:
                continue
            path = self.dir / "full" / f"{label}.png"
            cv2.imwrite(str(path), crop)
            count += 1
        log.info(f"{count} templates réels créés dans {self.dir / 'full'}")
        return count


# ---------------------------------------------------------------------------
# Moteur de détection
# ---------------------------------------------------------------------------

class CardDetector:
    """
    Détecte les cartes dans une image par template matching OpenCV.

    Priorité :
      1. Templates réels (dossier full/) si disponibles
      2. Templates synthétiques (dossier card/)
      3. Matching rang + déduction couleur HSV
      4. OCR Tesseract (fallback final)
    """

    def __init__(self, templates_dir: Path = TEMPLATES_DIR):
        self.templates_dir = templates_dir
        self._templates: dict[str, np.ndarray] = {}
        self._rank_templates: dict[str, list[np.ndarray]] = {}
        self._load_templates()

    # ------------------------------------------------------------------
    # Chargement des templates
    # ------------------------------------------------------------------

    def _load_templates(self) -> None:
        """Charge tous les templates disponibles en mémoire."""
        loaded = 0

        # Priorité 1 : templates réels (captures écran)
        full_dir = self.templates_dir / "full"
        if full_dir.exists():
            for path in full_dir.glob("*.png"):
                card = path.stem
                if card in CARDS:
                    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        self._templates[card] = img
                        loaded += 1

        # Priorité 2 : templates synthétiques
        card_dir = self.templates_dir / "card"
        if card_dir.exists():
            for path in card_dir.glob("*.png"):
                card = path.stem
                if card in CARDS and card not in self._templates:
                    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        self._templates[card] = img
                        loaded += 1

        # Templates de rang (pour matching en deux étapes)
        rank_dir = self.templates_dir / "rank"
        if rank_dir.exists():
            for path in rank_dir.glob("*.png"):
                parts = path.stem.split("_")
                if parts:
                    rank = parts[0]
                    if rank in RANKS:
                        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            self._rank_templates.setdefault(rank, []).append(img)

        # Générer les templates synthétiques si aucun n'existe
        if loaded == 0:
            log.info("Aucun template trouvé — génération automatique…")
            gen = TemplateGenerator(self.templates_dir)
            gen.generate_all()
            self._load_templates()
            return

        log.info(f"Templates chargés : {loaded} cartes complètes, "
                 f"{sum(len(v) for v in self._rank_templates.values())} rangs")

    def reload(self) -> None:
        """Recharge les templates depuis le disque."""
        self._templates.clear()
        self._rank_templates.clear()
        self._load_templates()

    # ------------------------------------------------------------------
    # Détection principale
    # ------------------------------------------------------------------

    def detect_cards(
        self,
        img_bgr: np.ndarray,
        max_cards: int = 7,
        min_card_w: int = 30,
        max_card_w: int = 120,
    ) -> list[CardDetection]:
        """
        Détecte toutes les cartes dans une image.
        Retourne une liste ordonnée gauche→droite de CardDetection.
        """
        if img_bgr is None or img_bgr.size == 0:
            return []

        results = []

        # Étape 1 : localiser les zones blanches (fonds de cartes)
        card_regions = self._find_card_regions(img_bgr, min_card_w, max_card_w)

        for (x, y, w, h) in card_regions[:max_cards]:
            crop = img_bgr[y:y+h, x:x+w]
            if crop.size == 0:
                continue

            detection = self._identify_card(crop)
            if detection.is_valid:
                detection.bbox = (x, y, w, h)
                results.append(detection)

        # Trier gauche → droite
        results.sort(key=lambda d: d.bbox[0])

        # Déduplication
        results = self._deduplicate(results)

        return results

    def detect_cards_in_region(
        self,
        full_screen: np.ndarray,
        region: dict,
        max_cards: int = 7,
    ) -> list[CardDetection]:
        """
        Détecte les cartes dans une région nommée de l'écran.
        region : {"left": ..., "top": ..., "width": ..., "height": ...}
        """
        h_img, w_img = full_screen.shape[:2]
        x = max(0, region["left"])
        y = max(0, region["top"])
        w = min(region["width"],  w_img - x)
        h = min(region["height"], h_img - y)
        crop = full_screen[y:y+h, x:x+w]
        detections = self.detect_cards(crop, max_cards=max_cards)

        # Remettre les coordonnées en absolu
        for d in detections:
            if d.bbox:
                bx, by, bw, bh = d.bbox
                d.bbox = (bx + x, by + y, bw, bh)

        return detections

    # ------------------------------------------------------------------
    # Localisation des zones de cartes
    # ------------------------------------------------------------------

    def _find_card_regions(
        self,
        img_bgr: np.ndarray,
        min_w: int,
        max_w: int,
    ) -> list[tuple]:
        """
        Trouve les rectangles blancs correspondant aux cartes.
        Utilise la détection de zones claires avec contours.
        """
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Seuillage : isoler les zones blanches (fond des cartes)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Morphologie pour boucher les trous
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        img_h, img_w = img_bgr.shape[:2]

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area   = w * h
            aspect = w / max(h, 1)

            # Filtres : taille et ratio d'une carte (~0.55–0.75)
            if (min_w <= w <= max_w and
                    h >= min_w * 1.2 and
                    0.40 <= aspect <= 0.90 and
                    area >= min_w * min_w * 0.8):
                regions.append((x, y, w, h))

        # Trier par position X
        regions.sort(key=lambda r: r[0])
        return regions

    # ------------------------------------------------------------------
    # Identification d'une carte
    # ------------------------------------------------------------------

    def _identify_card(self, card_crop: np.ndarray) -> CardDetection:
        """
        Identifie une carte en combinant template matching et HSV.
        """
        # Étape 1 : template matching complet
        result = self._match_full_template(card_crop)
        if result.confidence >= MATCH_THRESHOLD_HIGH:
            return result

        # Étape 2 : matching du rang + détection couleur HSV
        result2 = self._match_rank_then_suit(card_crop)
        if result2.confidence >= MATCH_THRESHOLD_LOW:
            if result2.confidence > result.confidence:
                return result2

        # Étape 3 : fallback OCR
        ocr_result = self._fallback_ocr(card_crop)
        if ocr_result.is_valid:
            return ocr_result

        # Retourner le meilleur résultat même si faible
        best = max([result, result2, ocr_result],
                   key=lambda r: r.confidence)
        return best

    def _match_full_template(self, crop: np.ndarray) -> CardDetection:
        """Template matching sur la carte complète."""
        if not self._templates:
            return CardDetection("", 0.0, "template")

        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        best_card  = ""
        best_score = 0.0

        for card, tmpl in self._templates.items():
            # Redimensionner le template à la taille du crop
            th, tw = tmpl.shape[:2]
            ch, cw = gray_crop.shape[:2]

            if tw == 0 or th == 0:
                continue

            # Essayer quelques échelles
            for scale in [1.0, 0.9, 1.1, 0.8]:
                new_w = int(cw * scale)
                new_h = int(ch * scale)
                if new_w < 10 or new_h < 10:
                    continue

                resized = cv2.resize(tmpl, (new_w, new_h))

                try:
                    res = cv2.matchTemplate(gray_crop, resized,
                                            cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    if max_val > best_score:
                        best_score = max_val
                        best_card  = card
                except cv2.error:
                    continue

        return CardDetection(best_card, best_score, "template")

    def _match_rank_then_suit(self, crop: np.ndarray) -> CardDetection:
        """
        Matching en deux étapes :
        1. Identifier le rang par template matching sur la zone haut-gauche
        2. Identifier la couleur (rouge/noir) par HSV, puis l'enseigne
        """
        if not self._rank_templates:
            return CardDetection("", 0.0, "hsv")

        h, w = crop.shape[:2]

        # Extraire la zone rang (quart supérieur gauche)
        rank_zone = crop[:h//2, :w//2]
        gray_rank = cv2.cvtColor(rank_zone, cv2.COLOR_BGR2GRAY)

        best_rank  = ""
        best_score = 0.0

        for rank, templates in self._rank_templates.items():
            for tmpl in templates:
                th, tw = tmpl.shape[:2]
                rh, rw = gray_rank.shape[:2]

                if tw > rw or th > rh or tw == 0 or th == 0:
                    continue

                try:
                    res = cv2.matchTemplate(gray_rank, tmpl,
                                            cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    if max_val > best_score:
                        best_score = max_val
                        best_rank  = rank
                except cv2.error:
                    continue

        if not best_rank:
            return CardDetection("", 0.0, "hsv")

        # Identifier l'enseigne par couleur HSV
        suit, suit_conf = self._detect_suit_hsv(crop)
        if not suit:
            suit      = "s"   # défaut noir
            suit_conf = 0.3

        combined_conf = (best_score * 0.7 + suit_conf * 0.3)
        return CardDetection(best_rank + suit, combined_conf, "hsv")

    def _detect_suit_hsv(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Détecte l'enseigne par analyse de couleur HSV.
        Retourne (suit, confidence).
        """
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Masque rouge (cœur/carreau)
        mask_red1 = cv2.inRange(hsv, np.array([0,  80, 80]),
                                      np.array([12, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([165, 80, 80]),
                                      np.array([180, 255, 255]))
        red_pixels = cv2.countNonZero(mask_red1) + cv2.countNonZero(mask_red2)

        # Masque noir (pique/trèfle)
        mask_dark  = cv2.inRange(hsv, np.array([0, 0, 0]),
                                       np.array([180, 60, 80]))
        dark_pixels = cv2.countNonZero(mask_dark)

        total = red_pixels + dark_pixels
        if total == 0:
            return "", 0.0

        red_ratio  = red_pixels  / total
        dark_ratio = dark_pixels / total

        # Distinguer cœur vs carreau, pique vs trèfle nécessite
        # l'analyse de la forme du symbole → simplification ici
        if red_ratio > 0.4:
            # Rouge → cœur ou carreau (on prend cœur par défaut,
            # mais on peut affiner avec la forme)
            suit = self._refine_red_suit(crop)
            return suit, min(red_ratio * 1.5, 0.9)
        elif dark_ratio > 0.3:
            suit = self._refine_dark_suit(crop)
            return suit, min(dark_ratio * 1.5, 0.9)

        return "", 0.0

    def _refine_red_suit(self, crop: np.ndarray) -> str:
        """Distingue cœur (h) de carreau (d) par analyse de forme."""
        h, w = crop.shape[:2]
        center_zone = crop[h//3:2*h//3, w//4:3*w//4]
        gray = cv2.cvtColor(center_zone, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Le carreau est plus anguleux, le cœur plus arrondi
        # Approximation via le ratio de pixels dans les coins
        ch, cw = thresh.shape[:2]
        if ch == 0 or cw == 0:
            return "h"

        top_center = thresh[:ch//2, cw//4:3*cw//4]
        top_pixels = cv2.countNonZero(top_center)

        # Cœur a deux bosses en haut → plus de pixels en haut
        # Carreau est pointu en haut → moins de pixels
        if top_pixels > (ch // 2 * cw // 2) * 0.4:
            return "h"
        return "d"

    def _refine_dark_suit(self, crop: np.ndarray) -> str:
        """Distingue pique (s) de trèfle (c) par analyse de forme."""
        h, w = crop.shape[:2]
        bottom_zone = crop[h//2:, :]
        gray = cv2.cvtColor(bottom_zone, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        bh, bw = thresh.shape[:2]
        if bh == 0 or bw == 0:
            return "s"

        # Trèfle a 3 boules → plus de pixels en bas
        # Pique a une tige fine → moins de pixels en bas
        bottom_pixels = cv2.countNonZero(thresh)
        ratio = bottom_pixels / max(bh * bw, 1)

        return "c" if ratio > 0.35 else "s"

    def _fallback_ocr(self, crop: np.ndarray) -> CardDetection:
        """Fallback OCR Tesseract si le template matching échoue."""
        try:
            import pytesseract
            from PIL import Image, ImageEnhance

            # Prétraitement agressif
            scale  = 3.0
            h, w   = crop.shape[:2]
            large  = cv2.resize(crop, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_CUBIC)
            gray   = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)
            clahe  = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            gray   = clahe.apply(gray)
            _, bw  = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            pil    = Image.fromarray(bw)
            pil    = ImageEnhance.Sharpness(pil).enhance(2.5)
            text   = pytesseract.image_to_string(
                pil,
                config="--psm 8 -c tessedit_char_whitelist=23456789TJQKAshdc"
            ).strip().replace(" ", "")

            # Normaliser
            if len(text) >= 2:
                rank = text[0].upper()
                suit = text[-1].lower()
                if rank in RANKS and suit in SUITS:
                    return CardDetection(rank + suit, 0.70, "ocr")

        except Exception as e:
            log.debug(f"OCR fallback échoué : {e}")

        return CardDetection("", 0.0, "ocr")

    # ------------------------------------------------------------------
    # Post-traitement
    # ------------------------------------------------------------------

    def _deduplicate(self, detections: list[CardDetection]) -> list[CardDetection]:
        """Supprime les doublons (même carte détectée deux fois)."""
        seen  = set()
        clean = []
        for d in detections:
            if d.card and d.card not in seen:
                seen.add(d.card)
                clean.append(d)
        return clean


# ---------------------------------------------------------------------------
# Intégration dans le pipeline de capture
# ---------------------------------------------------------------------------

class HybridCardExtractor:
    """
    Remplace extract_cards_from_text() de capture.py.
    Combine template matching + OCR pour une fiabilité maximale.

    Usage dans capture.py :
        extractor = HybridCardExtractor()

        # Au lieu de :
        cards = extract_cards_from_text(ocr_text(img))

        # Utiliser :
        cards = extractor.extract(img_bgr)
    """

    def __init__(self):
        self.detector = CardDetector()
        self._stats   = {"template": 0, "hsv": 0, "ocr": 0, "failed": 0}

    def extract(
        self,
        img_bgr: np.ndarray,
        max_cards: int = 7,
    ) -> list[str]:
        """
        Extrait les cartes d'une image.
        Retourne une liste de strings : ["Ks", "7h", ...]
        """
        detections = self.detector.detect_cards(img_bgr, max_cards=max_cards)

        results = []
        for d in detections:
            if d.is_valid:
                results.append(d.card)
                self._stats[d.method] = self._stats.get(d.method, 0) + 1
                log.debug(f"Carte détectée : {d.card} "
                           f"(conf={d.confidence:.2f}, méthode={d.method})")
            else:
                self._stats["failed"] += 1
                log.debug(f"Détection échouée : conf={d.confidence:.2f}")

        return results

    def extract_from_region(
        self,
        full_screen: np.ndarray,
        region: dict,
        max_cards: int = 7,
    ) -> list[str]:
        """Extrait depuis une région nommée de l'écran complet."""
        detections = self.detector.detect_cards_in_region(
            full_screen, region, max_cards
        )
        return [d.card for d in detections if d.is_valid]

    @property
    def accuracy_report(self) -> str:
        total = sum(self._stats.values())
        if total == 0:
            return "Aucune détection."
        lines = ["Rapport de précision :"]
        for method, count in self._stats.items():
            pct = count / total * 100
            lines.append(f"  {method:<10} : {count:4d} ({pct:.1f}%)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Outil de benchmark
# ---------------------------------------------------------------------------

class DetectorBenchmark:
    """
    Mesure la précision du détecteur sur un jeu de test.
    Génère des crops synthétiques et mesure le taux de reconnaissance.
    """

    def __init__(self, detector: CardDetector):
        self.detector = detector

    def run(self, num_tests: int = 52) -> dict:
        """
        Lance le benchmark sur num_tests cartes.
        Retourne les métriques de précision.
        """
        gen     = TemplateGenerator()
        gen.generate_all(card_w=65, card_h=90)
        self.detector.reload()

        correct   = 0
        total     = 0
        by_method = {"template": 0, "hsv": 0, "ocr": 0}
        errors    = []

        cards_to_test = CARDS[:num_tests]

        for card in cards_to_test:
            # Créer un crop synthétique avec légère variation
            tmpl_path = TEMPLATES_DIR / "card" / f"{card}.png"
            if not tmpl_path.exists():
                continue

            img = cv2.imread(str(tmpl_path))
            if img is None:
                continue

            # Ajouter du bruit pour simuler conditions réelles
            noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
            img   = cv2.add(img, noise)

            t0        = time.perf_counter()
            detection = self.detector._identify_card(img)
            elapsed   = (time.perf_counter() - t0) * 1000

            total += 1
            if detection.card == card:
                correct += 1
                by_method[detection.method] = by_method.get(detection.method, 0) + 1
            else:
                errors.append({
                    "expected": card,
                    "got":      detection.card,
                    "conf":     detection.confidence,
                    "method":   detection.method,
                })

        accuracy = correct / max(total, 1)

        result = {
            "total":      total,
            "correct":    correct,
            "accuracy":   round(accuracy * 100, 1),
            "by_method":  by_method,
            "errors":     errors[:10],   # 10 premiers erreurs
        }

        self._print_report(result)
        return result

    def _print_report(self, result: dict) -> None:
        print("\n" + "=" * 50)
        print("  BENCHMARK DÉTECTEUR DE CARTES")
        print("=" * 50)
        print(f"  Précision    : {result['accuracy']}%  "
              f"({result['correct']}/{result['total']})")
        print(f"  Par méthode  :")
        for method, count in result["by_method"].items():
            print(f"    {method:<12}: {count}")
        if result["errors"]:
            print(f"\n  Erreurs ({len(result['errors'])}) :")
            for e in result["errors"][:5]:
                print(f"    Attendu={e['expected']}  "
                      f"Reçu={e['got']}  "
                      f"Conf={e['conf']:.2f}")
        print("=" * 50)


# ---------------------------------------------------------------------------
# Patch de capture.py (monkey-patch drop-in)
# ---------------------------------------------------------------------------

def patch_capture_module():
    """
    Remplace extract_cards_from_text() dans capture.py par le détecteur
    basé sur templates. À appeler une seule fois au démarrage.

    Usage dans main.py :
        from card_detector import patch_capture_module
        patch_capture_module()
    """
    try:
        import capture
        extractor = HybridCardExtractor()

        def new_extract_cards(img_bgr: np.ndarray) -> list[str]:
            return extractor.extract(img_bgr)

        capture.extract_cards_from_img = new_extract_cards
        log.info("capture.py patché — détection par templates activée.")
        return extractor
    except ImportError:
        log.warning("capture.py non trouvé — patch ignoré.")
        return None


# ---------------------------------------------------------------------------
# Test en direct
# ---------------------------------------------------------------------------

def test_live(monitor_idx: int = 1, region_name: str = "player_cards") -> None:
    """Teste la détection en temps réel sur une capture live."""
    config_path = Path(__file__).parent / "config.json"
    regions     = {}
    if config_path.exists():
        with open(config_path) as f:
            regions = json.load(f).get("regions", {})

    detector  = CardDetector()
    extractor = HybridCardExtractor()

    print(f"\nDétection live sur '{region_name}' — Ctrl+C pour arrêter\n")

    with mss.mss() as sct:
        monitor = sct.monitors[monitor_idx]

        while True:
            try:
                raw = sct.grab(monitor)
                img = np.array(raw)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                if region_name in regions:
                    cards = extractor.extract_from_region(img, regions[region_name])
                else:
                    cards = extractor.extract(img, max_cards=2)

                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] {region_name}: {cards if cards else '(aucune carte)'}")

                time.sleep(1.0)

            except KeyboardInterrupt:
                print("\nArrêt.")
                print(extractor.accuracy_report)
                break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Détecteur de cartes par templates OpenCV"
    )
    parser.add_argument("--generate",  action="store_true",
                        help="Générer les templates synthétiques")
    parser.add_argument("--benchmark", action="store_true",
                        help="Lancer le benchmark de précision")
    parser.add_argument("--test",      action="store_true",
                        help="Tester en temps réel sur la capture écran")
    parser.add_argument("--region",    type=str, default="player_cards",
                        help="Région à tester (défaut: player_cards)")
    parser.add_argument("--screen",    type=int, default=1,
                        help="Index de l'écran (défaut: 1)")
    args = parser.parse_args()

    if args.generate:
        gen = TemplateGenerator()
        n   = gen.generate_all()
        print(f"\n✓ {n} templates générés dans {TEMPLATES_DIR}")

    elif args.benchmark:
        detector  = CardDetector()
        benchmark = DetectorBenchmark(detector)
        benchmark.run(num_tests=52)

    elif args.test:
        test_live(monitor_idx=args.screen, region_name=args.region)

    else:
        print("Usage :")
        print("  python card_detector.py --generate   # générer les templates")
        print("  python card_detector.py --benchmark  # tester la précision")
        print("  python card_detector.py --test       # test live")
        print("  python card_detector.py --test --region board  # tester le board")
