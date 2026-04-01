"""
diagnose.py — Outil de diagnostic visuel pour déboguer la détection
Lance ce script PENDANT que PokerStars est ouvert pour voir exactement
ce que le programme capture et détecte.

Usage :
    python diagnose.py              # screenshot + analyse complète
    python diagnose.py --live       # affichage temps réel (nécessite opencv)
    python diagnose.py --calibrate  # mode calibration interactive
    python diagnose.py --resolution # affiche ta résolution d'écran

Produit :
    diagnose_output/
        screenshot_full.png        — capture complète annotée
        region_player_cards.png    — zone cartes joueur
        region_board.png           — zone board
        region_pot.png             — zone pot
        region_stack.png           — zone stack
        diagnose_report.txt        — rapport texte complet
"""

import json
import logging
import sys
import time
from pathlib import Path

import cv2
import mss
import numpy as np

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "diagnose_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Couleurs des annotations (BGR)
# ---------------------------------------------------------------------------
COLOR_OK      = (50,  200,  50)    # vert  — détecté
COLOR_EMPTY   = (50,  50,  200)    # bleu  — région vide
COLOR_ERROR   = (50,  50,  220)    # rouge — erreur
COLOR_REGION  = (200, 150,  50)    # orange — contour région
COLOR_TEXT_BG = (20,  20,   20)    # fond des labels

# ---------------------------------------------------------------------------
# Capture d'écran
# ---------------------------------------------------------------------------

def take_screenshot(monitor_idx: int = 1) -> np.ndarray:
    with mss.mss() as sct:
        monitors = sct.monitors
        log.info(f"Moniteurs disponibles : {len(monitors) - 1}")
        for i, m in enumerate(monitors[1:], 1):
            log.info(f"  Moniteur {i} : {m['width']}×{m['height']} à ({m['left']},{m['top']})")

        if monitor_idx >= len(monitors):
            monitor_idx = 1
        monitor = monitors[monitor_idx]
        raw = sct.grab(monitor)
        frame = np.array(raw)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

# ---------------------------------------------------------------------------
# Dessin des annotations
# ---------------------------------------------------------------------------

def draw_region(
    frame: np.ndarray,
    region: dict,
    label: str,
    color: tuple,
    content: str = "",
) -> None:
    """Dessine un rectangle annoté sur le frame."""
    x  = region["left"]
    y  = region["top"]
    w  = region["width"]
    h  = region["height"]

    # Rectangle de la région
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Label avec fond
    label_full = f"{label}: {content}" if content else label
    (tw, th), _ = cv2.getTextSize(label_full, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), COLOR_TEXT_BG, -1)
    cv2.putText(frame, label_full,
                (x + 3, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

def draw_crosshair(frame: np.ndarray, x: int, y: int, color: tuple, size: int = 15) -> None:
    cv2.line(frame, (x - size, y), (x + size, y), color, 1)
    cv2.line(frame, (x, y - size), (x, y + size), color, 1)

# ---------------------------------------------------------------------------
# Analyse d'une région
# ---------------------------------------------------------------------------

def analyse_region(
    frame: np.ndarray,
    region: dict,
    name: str,
) -> dict:
    """Extrait et analyse une région du frame."""
    x, y = region["left"], region["top"]
    w, h = region["width"], region["height"]

    h_frame, w_frame = frame.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_frame, x + w)
    y2 = min(h_frame, y + h)

    crop = frame[y1:y2, x1:x2]

    result = {
        "name":      name,
        "region":    region,
        "in_bounds": (x >= 0 and y >= 0 and x + w <= w_frame and y + h <= h_frame),
        "crop_size": crop.shape[:2] if crop.size > 0 else (0, 0),
        "is_empty":  False,
        "mean_brightness": 0.0,
        "ocr_text":  "",
        "cards":     [],
        "error":     "",
    }

    if crop.size == 0:
        result["error"] = "Région hors écran"
        return result

    # Sauvegarder le crop
    out_path = OUTPUT_DIR / f"region_{name}.png"
    cv2.imwrite(str(out_path), crop)

    # Analyser la luminosité
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    result["mean_brightness"] = float(np.mean(gray))

    # Vérifier si vide (tout noir ou quasi)
    if result["mean_brightness"] < 5:
        result["is_empty"] = True
        result["error"] = "Image noire — PokerStars minimisé ou mauvaise région"
        return result

    # Tenter OCR
    try:
        import pytesseract
        from PIL import Image, ImageEnhance

        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        pil = ImageEnhance.Contrast(pil).enhance(2.0)
        big = pil.resize((pil.width * 3, pil.height * 3), Image.LANCZOS)
        result["ocr_text"] = pytesseract.image_to_string(
            big,
            config="--psm 7 -c tessedit_char_whitelist=0123456789AaKkQqJjTt23456789shdc$€.,♠♥♦♣ "
        ).strip()
    except Exception as e:
        result["ocr_text"] = f"[OCR indispo: {e}]"

    # Tenter détection de cartes
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from capture import extract_cards_from_image
        result["cards"] = extract_cards_from_image(crop, name)
    except Exception:
        pass

    return result

# ---------------------------------------------------------------------------
# Détection de la fenêtre PokerStars
# ---------------------------------------------------------------------------

def find_pokerstars_window() -> dict:
    """Tente de trouver la fenêtre PokerStars via win32 (Windows uniquement)."""
    result = {"found": False, "rect": None, "title": ""}
    try:
        import win32gui, win32con
        def callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if "pokerstars" in title.lower() or "poker stars" in title.lower():
                    rect = win32gui.GetWindowRect(hwnd)
                    windows.append({"hwnd": hwnd, "title": title, "rect": rect})
        windows = []
        win32gui.EnumWindows(callback, windows)
        if windows:
            w = windows[0]
            result["found"] = True
            result["title"] = w["title"]
            x1, y1, x2, y2 = w["rect"]
            result["rect"] = {
                "left": x1, "top": y1,
                "width": x2 - x1, "height": y2 - y1
            }
            log.info(f"Fenêtre PokerStars trouvée : {w['title']}")
            log.info(f"  Position : ({x1}, {y1}) — Taille : {x2-x1}×{y2-y1}")
    except ImportError:
        log.debug("win32gui non disponible (pip install pywin32 pour détecter la fenêtre)")
    except Exception as e:
        log.debug(f"Détection fenêtre : {e}")
    return result

# ---------------------------------------------------------------------------
# Rapport principal
# ---------------------------------------------------------------------------

def run_diagnosis(monitor_idx: int = 1, save_annotated: bool = True) -> dict:
    """Lance le diagnostic complet et génère les fichiers de sortie."""

    log.info("=" * 55)
    log.info("  DIAGNOSTIC DE CAPTURE — PokerStars")
    log.info("=" * 55)

    # 1. Screenshot
    log.info("\n[1] Capture d'écran…")
    try:
        frame = take_screenshot(monitor_idx)
        h_frame, w_frame = frame.shape[:2]
        log.info(f"  Résolution capturée : {w_frame}×{h_frame}")
    except Exception as e:
        log.error(f"Impossible de capturer l'écran : {e}")
        return {}

    # Sauvegarder le screenshot brut
    raw_path = OUTPUT_DIR / "screenshot_full_raw.png"
    cv2.imwrite(str(raw_path), frame)
    log.info(f"  Screenshot sauvegardé : {raw_path}")

    # 2. Charger les régions
    log.info("\n[2] Chargement des régions…")
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from capture import load_regions, DEFAULT_REGIONS
        regions = load_regions()
        source = "config.json" if (Path(__file__).parent / "config.json").exists() else "défaut"
        log.info(f"  Régions chargées depuis : {source}")
    except Exception as e:
        log.warning(f"  Impossible de charger capture.py ({e}) — utilisation des régions codées en dur")
        regions = {
            "table":        {"top": 0,   "left": 0,   "width": w_frame, "height": h_frame},
            "player_cards": {"top": 830, "left": 800, "width": 320, "height": 120},
            "board":        {"top": 440, "left": 560, "width": 800, "height": 130},
            "pot":          {"top": 395, "left": 750, "width": 420, "height": 50},
            "player_stack": {"top": 950, "left": 850, "width": 220, "height": 40},
            "current_bet":  {"top": 760, "left": 850, "width": 220, "height": 40},
        }

    # 3. Détecter la fenêtre PokerStars
    log.info("\n[3] Recherche de la fenêtre PokerStars…")
    ps_window = find_pokerstars_window()

    # 4. Analyser chaque région
    log.info("\n[4] Analyse des régions…")
    report = {
        "resolution":    f"{w_frame}×{h_frame}",
        "needs_rescale": w_frame != 1920 or h_frame != 1080,
        "scale_x":       w_frame / 1920,
        "scale_y":       h_frame / 1080,
        "ps_window":     ps_window,
        "regions":       {},
        "issues":        [],
        "suggestions":   [],
    }

    annotated = frame.copy()
    key_regions = ["player_cards", "board", "pot", "player_stack", "current_bet"]

    for name, region in regions.items():
        if name == "table":
            continue

        # Adapter les régions si la résolution est différente de 1920×1080
        scaled_region = region.copy()
        if report["needs_rescale"] and name != "action_log":
            scaled_region = {
                "left":   int(region["left"]   * report["scale_x"]),
                "top":    int(region["top"]    * report["scale_y"]),
                "width":  int(region["width"]  * report["scale_x"]),
                "height": int(region["height"] * report["scale_y"]),
            }

        analysis = analyse_region(frame, scaled_region, name)
        report["regions"][name] = analysis

        # Choisir la couleur selon le résultat
        if analysis["error"]:
            color = COLOR_ERROR
        elif analysis["cards"] or (analysis["ocr_text"] and analysis["ocr_text"] != "[OCR indispo"):
            color = COLOR_OK
        else:
            color = COLOR_EMPTY

        content = ""
        if analysis["cards"]:
            content = f"cartes={analysis['cards']}"
        elif analysis["ocr_text"] and not analysis["ocr_text"].startswith("["):
            content = f"ocr='{analysis['ocr_text'][:20]}'"
        elif analysis["error"]:
            content = analysis["error"][:30]
        else:
            content = f"lum={analysis['mean_brightness']:.0f}"

        draw_region(annotated, scaled_region, name, color, content)

        status = "✓" if color == COLOR_OK else ("⚠" if color == COLOR_EMPTY else "✗")
        log.info(f"  {status} {name:<15} brightness={analysis['mean_brightness']:.0f}"
                 f"  ocr='{analysis['ocr_text'][:25]}'"
                 f"  {'IN BOUNDS' if analysis['in_bounds'] else 'HORS ÉCRAN'}")

    # 5. Identifier les problèmes
    log.info("\n[5] Analyse des problèmes…")

    if report["needs_rescale"]:
        sx, sy = report["scale_x"], report["scale_y"]
        msg = (f"Résolution {w_frame}×{h_frame} ≠ 1920×1080. "
               f"Facteur : {sx:.2f}×{sy:.2f}")
        report["issues"].append(msg)
        log.warning(f"  ⚠ {msg}")

        # Générer le config.json corrigé
        corrected = {}
        for name, region in regions.items():
            corrected[name] = {
                "left":   int(region["left"]   * sx),
                "top":    int(region["top"]    * sy),
                "width":  int(region["width"]  * sx),
                "height": int(region["height"] * sy),
            }
        config_path = Path(__file__).parent / "config_corrected.json"
        with open(config_path, "w") as f:
            json.dump({"regions": corrected}, f, indent=2)
        log.info(f"  → config_corrected.json généré (copie-le en config.json)")
        report["suggestions"].append(
            f"Copie config_corrected.json → config.json pour corriger les régions"
        )

    all_black = all(
        r["mean_brightness"] < 5
        for r in report["regions"].values()
        if r.get("crop_size", (0,0)) != (0,0)
    )
    if all_black:
        report["issues"].append("Toutes les régions sont noires — PokerStars est peut-être minimisé ou en mode plein-écran exclusif")
        log.error("  ✗ Toutes les régions sont noires !")
        report["suggestions"].append(
            "Lance PokerStars en mode Fenêtré (pas plein-écran exclusif) :\n"
            "  PokerStars → Options → Table → Mode d'affichage → Fenêtré"
        )

    ps_cards  = report["regions"].get("player_cards", {})
    board_reg = report["regions"].get("board", {})

    if ps_cards.get("mean_brightness", 0) > 5 and not ps_cards.get("cards"):
        report["issues"].append("Cartes joueur : région visible mais aucune carte détectée")
        log.warning("  ⚠ Cartes joueur non détectées — calibration nécessaire")
        report["suggestions"].append(
            "Lance la calibration : python main.py --calibrate\n"
            "  ou génère les templates : python main.py --gen-templates"
        )

    if not ps_window["found"]:
        report["suggestions"].append(
            "PokerStars non détecté — assure-toi qu'il est ouvert et visible"
        )

    # 6. Sauvegarder le frame annoté
    if save_annotated:
        # Ajouter un panneau de légende
        legend_y = h_frame - 60
        cv2.rectangle(annotated, (0, legend_y), (w_frame, h_frame), (20, 20, 20), -1)
        cv2.putText(annotated, f"Résolution: {w_frame}x{h_frame}",
                    (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(annotated, f"  vert=détecté  bleu=vide  rouge=erreur  orange=région",
                    (10, legend_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)

        ann_path = OUTPUT_DIR / "screenshot_annotated.png"
        cv2.imwrite(str(ann_path), annotated)
        log.info(f"\n  Screenshot annoté : {ann_path}")

    # 7. Rapport texte
    _write_text_report(report, w_frame, h_frame)

    # 8. Résumé
    log.info("\n" + "=" * 55)
    if report["issues"]:
        log.info(f"  {len(report['issues'])} PROBLÈME(S) DÉTECTÉ(S) :")
        for issue in report["issues"]:
            log.warning(f"    ⚠ {issue}")
        log.info("\n  SUGGESTIONS :")
        for i, sug in enumerate(report["suggestions"], 1):
            log.info(f"    {i}. {sug}")
    else:
        log.info("  Aucun problème majeur détecté.")
    log.info("=" * 55)
    log.info(f"\n  Fichiers générés dans : {OUTPUT_DIR}/")

    return report


def _write_text_report(report: dict, w: int, h: int) -> None:
    lines = [
        "=== RAPPORT DE DIAGNOSTIC PokerStars ===",
        f"Date          : {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Résolution    : {w}×{h}",
        f"Scale vs 1080p: x={report['scale_x']:.2f} y={report['scale_y']:.2f}",
        "",
        "--- RÉGIONS ---",
    ]
    for name, r in report["regions"].items():
        status = "OK" if not r.get("error") and r.get("mean_brightness",0) > 5 else "PROBLÈME"
        lines.append(
            f"  {name:<15} {status:<10} "
            f"lum={r.get('mean_brightness',0):.0f}  "
            f"ocr='{r.get('ocr_text','')[:30]}'  "
            f"cartes={r.get('cards',[])}  "
            f"{'HORS_ECRAN' if not r.get('in_bounds', True) else ''}"
        )

    lines += ["", "--- PROBLÈMES ---"]
    for issue in report.get("issues", []):
        lines.append(f"  ⚠ {issue}")
    if not report.get("issues"):
        lines.append("  Aucun problème détecté.")

    lines += ["", "--- SUGGESTIONS ---"]
    for i, sug in enumerate(report.get("suggestions", []), 1):
        lines.append(f"  {i}. {sug}")

    path = OUTPUT_DIR / "diagnose_report.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"  Rapport texte : {path}")


# ---------------------------------------------------------------------------
# Mode live (affichage temps réel)
# ---------------------------------------------------------------------------

def run_live(monitor_idx: int = 1) -> None:
    """Affiche les régions en temps réel dans une fenêtre OpenCV."""
    log.info("Mode live — appuie sur Q pour quitter, S pour sauvegarder")

    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from capture import load_regions
        regions = load_regions()
    except Exception:
        log.warning("Utilisation des régions par défaut")
        regions = {}

    with mss.mss() as sct:
        monitor = sct.monitors[monitor_idx]
        while True:
            raw   = sct.grab(monitor)
            frame = np.array(raw)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            h, w  = frame.shape[:2]

            # Annoter les régions
            vis = frame.copy()
            sx  = w / 1920
            sy  = h / 1080

            for name, region in regions.items():
                if name == "table":
                    continue
                scaled = {
                    "left":   int(region["left"]   * sx),
                    "top":    int(region["top"]    * sy),
                    "width":  int(region["width"]  * sx),
                    "height": int(region["height"] * sy),
                }
                color = COLOR_REGION
                crop  = frame[
                    scaled["top"]:scaled["top"]+scaled["height"],
                    scaled["left"]:scaled["left"]+scaled["width"]
                ]
                if crop.size > 0 and np.mean(crop) > 10:
                    color = COLOR_OK
                draw_region(vis, scaled, name, color)

            # Réduire pour l'affichage
            display = cv2.resize(vis, (1280, 720))
            cv2.imshow("Diagnostic Live — Q pour quitter", display)

            key = cv2.waitKey(500) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.imwrite(str(OUTPUT_DIR / "live_capture.png"), frame)
                log.info("Screenshot sauvegardé.")

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Diagnostic de capture PokerStars",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python diagnose.py                    # analyse complète
  python diagnose.py --live             # affichage temps réel
  python diagnose.py --monitor 2        # deuxième écran
  python diagnose.py --resolution       # affiche la résolution

Fichiers générés dans diagnose_output/ :
  screenshot_annotated.png  — capture annotée avec les régions
  region_*.png              — chaque région individuellement
  diagnose_report.txt       — rapport texte complet
  config_corrected.json     — config.json corrigé si résolution ≠ 1920×1080
        """
    )
    parser.add_argument("--monitor",    type=int, default=1,
                        help="Numéro de moniteur (défaut: 1)")
    parser.add_argument("--live",       action="store_true",
                        help="Mode affichage temps réel")
    parser.add_argument("--resolution", action="store_true",
                        help="Afficher la résolution et quitter")
    args = parser.parse_args()

    if args.resolution:
        with mss.mss() as sct:
            for i, m in enumerate(sct.monitors[1:], 1):
                print(f"Moniteur {i} : {m['width']}×{m['height']} à ({m['left']},{m['top']})")
        sys.exit(0)

    if args.live:
        run_live(args.monitor)
    else:
        run_diagnosis(args.monitor)
