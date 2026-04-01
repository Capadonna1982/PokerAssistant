"""
alerts.py — Alertes sonores pour situations critiques au poker
Dépendances : aucune (utilise winsound sur Windows, playsound optionnel)
              pip install playsound  (cross-platform, optionnel)

Déclenche un son selon la situation :
  - RAISE fort (equity > 75%)    → son montant (positif)
  - ALL-IN recommandé             → son urgent
  - FOLD recommandé fort          → son descendant
  - Situation critique (SPR ≤ 1) → son d'alerte
  - Nouvelle main détectée        → son discret
  - Adversaire relance fort       → son d'avertissement

Usage :
    alerter = AlertManager()
    alerter.on_advice(advice, equity_result)   # depuis main.py
    alerter.on_opponent_action("RAISE", 200)   # action adverse
    alerter.test_all()                         # tester tous les sons
"""

import logging
import platform
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types d'alertes
# ---------------------------------------------------------------------------

class AlertType(str, Enum):
    STRONG_RAISE    = "strong_raise"    # RAISE/BET avec forte equity
    ALL_IN          = "all_in"          # ALL-IN recommandé
    STRONG_FOLD     = "strong_fold"     # FOLD avec faible equity (< 20%)
    CRITICAL_SPR    = "critical_spr"    # SPR ≤ 1, décision binaire
    NEW_HAND        = "new_hand"        # nouvelle main détectée
    OPP_AGGRESSION  = "opp_aggression"  # adversaire relance fort
    OPP_ALL_IN      = "opp_all_in"      # adversaire all-in
    BLUFF_DETECTED  = "bluff_detected"  # pattern de bluff confirmé
    HIGH_EQUITY     = "high_equity"     # equity > 90%, main très forte


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AlertConfig:
    """Configuration des alertes sonores."""
    enabled:            bool  = True
    volume:             float = 0.7     # 0.0 – 1.0
    min_interval_s:     float = 3.0     # délai minimum entre deux alertes

    # Seuils de déclenchement
    strong_raise_equity:float = 0.72    # equity minimale pour alerte RAISE
    strong_fold_equity: float = 0.22    # equity maximale pour alerte FOLD
    high_equity_thresh: float = 0.90    # equity pour alerte main très forte
    opp_raise_min:      float = 0.5     # ratio pot_min/pot pour alerte adverse

    # Activer/désactiver par type
    active: dict = None

    def __post_init__(self):
        if self.active is None:
            self.active = {t: True for t in AlertType}

    def is_active(self, alert_type: AlertType) -> bool:
        return self.enabled and self.active.get(alert_type, True)


# ---------------------------------------------------------------------------
# Moteur audio multi-plateforme
# ---------------------------------------------------------------------------

class SoundEngine:
    """
    Joue des sons via la méthode disponible sur la plateforme.
    Priorité : playsound → winsound (Windows) → beep ASCII (fallback)
    """

    SOUNDS_DIR = Path(__file__).parent / "sounds"

    # Sons synthétiques générés par beep (fréquence Hz, durée ms)
    BEEP_PATTERNS = {
        AlertType.STRONG_RAISE:   [(880, 120), (1100, 180)],              # montant
        AlertType.ALL_IN:         [(440, 80), (660, 80), (880, 200)],     # urgent triplet
        AlertType.STRONG_FOLD:    [(660, 120), (440, 200)],               # descendant
        AlertType.CRITICAL_SPR:   [(600, 60), (600, 60), (600, 200)],    # pulsé
        AlertType.NEW_HAND:       [(880, 60)],                            # discret
        AlertType.OPP_AGGRESSION: [(550, 100), (700, 100)],              # avertissement
        AlertType.OPP_ALL_IN:     [(400, 60), (800, 60), (400, 200)],    # alarme
        AlertType.BLUFF_DETECTED: [(1000, 80), (800, 80), (1000, 120)],  # signal
        AlertType.HIGH_EQUITY:    [(880, 80), (1100, 80), (1320, 200)],  # victorieux
    }

    def __init__(self):
        self._system      = platform.system()
        self._has_playsound = self._check_playsound()
        self._has_winsound  = self._check_winsound()
        log.info(
            f"SoundEngine : système={self._system} "
            f"playsound={self._has_playsound} "
            f"winsound={self._has_winsound}"
        )

    def _check_playsound(self) -> bool:
        try:
            import playsound
            return True
        except ImportError:
            return False

    def _check_winsound(self) -> bool:
        if self._system != "Windows":
            return False
        try:
            import winsound
            return True
        except ImportError:
            return False

    def play(self, alert_type: AlertType, volume: float = 0.7) -> None:
        """Joue le son associé à un type d'alerte (non-bloquant)."""
        thread = threading.Thread(
            target=self._play_async,
            args=(alert_type, volume),
            daemon=True,
        )
        thread.start()

    def _play_async(self, alert_type: AlertType, volume: float) -> None:
        """Joue le son dans un thread séparé."""
        # Essayer un fichier .wav personnalisé d'abord
        wav_path = self.SOUNDS_DIR / f"{alert_type.value}.wav"
        if wav_path.exists() and self._has_playsound:
            try:
                import playsound
                playsound.playsound(str(wav_path), block=True)
                return
            except Exception as e:
                log.debug(f"playsound erreur : {e}")

        # Fallback : beeps synthétiques
        pattern = self.BEEP_PATTERNS.get(alert_type, [(440, 200)])
        self._play_beeps(pattern, volume)

    def _play_beeps(self, pattern: list, volume: float) -> None:
        """Joue une séquence de beeps."""
        if self._has_winsound:
            import winsound
            for freq, duration in pattern:
                try:
                    winsound.Beep(int(freq), int(duration))
                    time.sleep(0.02)
                except Exception:
                    pass

        elif self._system == "Darwin":
            # macOS : utiliser osascript pour beep système
            import subprocess
            for _ in pattern:
                try:
                    subprocess.run(["osascript", "-e", "beep"], capture_output=True)
                    time.sleep(0.15)
                except Exception:
                    pass

        else:
            # Linux / fallback : beep ASCII dans le terminal
            for freq, duration in pattern:
                print(f"\a", end="", flush=True)   # ASCII bell
                time.sleep(duration / 1000)

    def generate_wav_files(self) -> None:
        """
        Génère des fichiers WAV synthétiques pour chaque type d'alerte.
        Nécessite numpy. Crée le dossier sounds/ si absent.
        """
        try:
            import struct
            import wave
            import math

            self.SOUNDS_DIR.mkdir(exist_ok=True)

            for alert_type, pattern in self.BEEP_PATTERNS.items():
                path = self.SOUNDS_DIR / f"{alert_type.value}.wav"
                if path.exists():
                    continue

                sample_rate = 44100
                samples = []

                for freq, duration_ms in pattern:
                    n_samples = int(sample_rate * duration_ms / 1000)
                    fade = min(200, n_samples // 4)

                    for i in range(n_samples):
                        # Onde sinusoïdale
                        val = math.sin(2 * math.pi * freq * i / sample_rate)
                        # Fade in/out
                        if i < fade:
                            val *= i / fade
                        elif i > n_samples - fade:
                            val *= (n_samples - i) / fade
                        samples.append(int(val * 16000))

                    # Silence entre les notes
                    samples.extend([0] * int(sample_rate * 0.03))

                with wave.open(str(path), "w") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))

            log.info(f"Fichiers WAV générés dans {self.SOUNDS_DIR}")

        except Exception as e:
            log.warning(f"Génération WAV échouée : {e}")


# ---------------------------------------------------------------------------
# Gestionnaire d'alertes principal
# ---------------------------------------------------------------------------

class AlertManager:
    """
    Orchestre les alertes sonores selon les conseils et actions détectés.

    Usage dans main.py :
        alerter = AlertManager()
        alerter.on_advice(advice, result)          # conseil Claude/engine
        alerter.on_opponent_action("RAISE", 200)   # action adverse
        alerter.on_new_hand()                      # nouvelle main
        alerter.on_spr(spr_value)                  # SPR critique
    """

    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self.engine = SoundEngine()
        self._last_alert_time: dict[AlertType, float] = {}
        self._last_alert_any:  float = 0.0

        # Générer les WAV synthétiques au démarrage
        try:
            self.engine.generate_wav_files()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Triggers publics
    # ------------------------------------------------------------------

    def on_advice(self, advice, equity_result=None) -> None:
        """
        Déclenche une alerte selon le conseil reçu (PokerAdvice).
        À appeler depuis _process_state() dans main.py.
        """
        action = getattr(advice, "recommended_action", "") or ""
        equity = getattr(advice, "win_probability", 0.0)
        spr    = getattr(advice, "spr", 0.0)

        # ALL-IN recommandé
        if action == "ALL-IN":
            self._trigger(AlertType.ALL_IN)
            return

        # RAISE/BET avec forte equity
        if action in ("RAISE", "BET") and equity >= self.config.strong_raise_equity:
            alert_type = (
                AlertType.HIGH_EQUITY
                if equity >= self.config.high_equity_thresh
                else AlertType.STRONG_RAISE
            )
            self._trigger(alert_type)
            return

        # FOLD avec très faible equity
        if action == "FOLD" and equity <= self.config.strong_fold_equity:
            self._trigger(AlertType.STRONG_FOLD)
            return

        # SPR critique (décision binaire imminente)
        if spr > 0 and spr <= 1.0:
            self._trigger(AlertType.CRITICAL_SPR)

    def on_opponent_action(self, action: str, amount: float = 0.0, pot: float = 0.0) -> None:
        """
        Alerte quand un adversaire fait une action agressive.
        À appeler depuis action_detector ou main.py.
        """
        action = action.upper()

        if action == "ALL-IN" or action == "ALL_IN":
            self._trigger(AlertType.OPP_ALL_IN)
        elif action in ("RAISE", "BET"):
            # Alerter seulement si c'est une grosse mise (> 50% du pot)
            if pot > 0 and amount / pot >= self.config.opp_raise_min:
                self._trigger(AlertType.OPP_AGGRESSION)
            elif pot == 0:
                self._trigger(AlertType.OPP_AGGRESSION)

    def on_new_hand(self) -> None:
        """Son discret signalant une nouvelle main détectée."""
        self._trigger(AlertType.NEW_HAND)

    def on_spr(self, spr: float) -> None:
        """Alerte si SPR ≤ 1 (situation de shove/fold)."""
        if spr <= 1.0 and spr > 0:
            self._trigger(AlertType.CRITICAL_SPR)

    def on_bluff_detected(self, player: str) -> None:
        """Alerte quand un pattern de bluff fiable est confirmé."""
        log.info(f"Bluff détecté pour {player} — alerte sonore")
        self._trigger(AlertType.BLUFF_DETECTED)

    # ------------------------------------------------------------------
    # Moteur interne
    # ------------------------------------------------------------------

    def _trigger(self, alert_type: AlertType) -> None:
        """Déclenche une alerte si les conditions le permettent."""
        if not self.config.is_active(alert_type):
            return

        now = time.time()

        # Délai minimum entre deux alertes du même type
        last = self._last_alert_time.get(alert_type, 0.0)
        if now - last < self.config.min_interval_s:
            return

        # Délai minimum absolu entre toutes alertes
        if now - self._last_alert_any < 1.0:
            return

        self._last_alert_time[alert_type] = now
        self._last_alert_any = now

        log.debug(f"Alerte sonore : {alert_type.value}")
        self.engine.play(alert_type, self.config.volume)

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def test_all(self) -> None:
        """Joue tous les sons avec une pause entre chaque."""
        print("\nTest des alertes sonores :")
        for alert_type in AlertType:
            print(f"  {alert_type.value}…")
            self.engine.play(alert_type, self.config.volume)
            time.sleep(1.2)
        print("Test terminé.")

    def set_enabled(self, enabled: bool) -> None:
        self.config.enabled = enabled
        log.info(f"Alertes sonores : {'activées' if enabled else 'désactivées'}")

    def set_volume(self, volume: float) -> None:
        self.config.volume = max(0.0, min(1.0, volume))

    def configure_alert(self, alert_type: AlertType, enabled: bool) -> None:
        self.config.active[alert_type] = enabled


# ---------------------------------------------------------------------------
# Intégration dans main.py (fonction raccourci)
# ---------------------------------------------------------------------------

def create_alert_manager(
    enabled:      bool  = True,
    volume:       float = 0.7,
    min_interval: float = 3.0,
) -> AlertManager:
    """
    Crée et configure un AlertManager depuis main.py.

    Usage dans PokerAssistant.__init__() :
        from alerts import create_alert_manager
        self.alerter = create_alert_manager()

    Usage dans _process_state() :
        self.alerter.on_advice(advice, result)

    Usage dans new_hand() :
        self.alerter.on_new_hand()
    """
    config = AlertConfig(
        enabled        = enabled,
        volume         = volume,
        min_interval_s = min_interval,
    )
    manager = AlertManager(config)
    log.info(
        f"AlertManager créé — "
        f"{'activé' if enabled else 'désactivé'}, "
        f"volume={volume:.0%}, "
        f"intervalle={min_interval}s"
    )
    return manager


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Alertes sonores poker")
    parser.add_argument("--test",    action="store_true",
                        help="Tester tous les sons")
    parser.add_argument("--gen-wav", action="store_true",
                        help="Générer les fichiers WAV synthétiques")
    parser.add_argument("--sound",   type=str, default=None,
                        help=f"Jouer un son spécifique ({', '.join(t.value for t in AlertType)})")
    args = parser.parse_args()

    manager = AlertManager()

    if args.gen_wav:
        manager.engine.generate_wav_files()
        print(f"Fichiers WAV générés dans {manager.engine.SOUNDS_DIR}")

    elif args.test:
        manager.test_all()

    elif args.sound:
        try:
            alert_type = AlertType(args.sound)
            print(f"Lecture : {alert_type.value}")
            manager.engine.play(alert_type)
            time.sleep(2)
        except ValueError:
            print(f"Type inconnu. Disponibles : {[t.value for t in AlertType]}")

    else:
        print("Alertes sonores — commandes :")
        print("  --test         Tester tous les sons")
        print("  --gen-wav      Générer les fichiers WAV")
        print("  --sound NOM    Jouer un son spécifique")
        print(f"\nTypes disponibles : {[t.value for t in AlertType]}")
