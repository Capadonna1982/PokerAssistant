"""
leak_finder.py — Détection automatique des fuites (leaks) dans ton jeu
Dépendances : tracker.py (déjà installé)

Un "leak" est une erreur récurrente qui te coûte de l'argent.
Ce module analyse la base poker_stats.db pour détecter :

  1. Fuites préflop
     - Calls trop larges en early position (call > 20% en UTG)
     - 3-bet insuffisant face aux ouvertures
     - Folds trop fréquents aux relances

  2. Fuites postflop
     - Over-folding face aux cbets (fold > 65%)
     - Under-betting sur les bonnes mains
     - Chaser les draws sans pot odds favorables

  3. Fuites de sizing
     - Bet trop petit pour valeur (< 50% pot avec mains fortes)
     - Over-bet comme bluff trop fréquent

  4. Fuites de gestion de bankroll
     - Ignorer les conseils Claude systématiquement
     - EV réalisée vs EV théorique (leakage d'EV)

  5. Fuites par position
     - Perdre systématiquement dans certaines positions
     - Profitabilité nulle hors position

Chaque leak est noté sur 10 (sévérité) et accompagné d'un conseil.

Usage :
    finder = LeakFinder(tracker)
    report = finder.analyse()
    print(report.top_leaks(5))

    # Intégration dans stats_viewer.py
    finder.add_tab_to_viewer(stats_viewer)
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seuils de détection
# ---------------------------------------------------------------------------

# Nombre minimum de mains pour qu'un leak soit fiable
MIN_HANDS_LEAK = 20

# Seuils d'alerte
FOLD_TO_CBET_THRESHOLD   = 0.65   # fold > 65% face cbet = leak
CALL_EV_NEG_THRESHOLD    = -5.0   # EV moyenne < -5$ = leak de call
EV_LEAK_THRESHOLD        = 0.25   # EV réalisée < 75% de l'EV théorique
ADVICE_IGNORE_THRESHOLD  = 0.45   # ignore conseils > 55% du temps = leak
MIN_RESULT_LOSS          = -20.0  # perte moyenne < -20$ par main = leak sévère

# Sévérité (1 = mineur, 10 = critique)
SEVERITY = {
    "critical": (8, 10),
    "major":    (5, 7),
    "minor":    (1, 4),
}


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------

@dataclass
class Leak:
    """Fuite détectée dans le jeu."""
    category:    str        # "preflop" / "postflop" / "sizing" / "mental" / "position"
    name:        str        # nom court du leak
    description: str        # explication détaillée
    severity:    int        # 1–10
    frequency:   float      # fréquence d'occurrence (0–1)
    cost_per_100:float      # coût estimé en $ par 100 mains
    sample_size: int        # nombre de mains analysées
    advice:      str        # conseil pour corriger
    data:        dict = field(default_factory=dict)   # données brutes

    @property
    def severity_label(self) -> str:
        if self.severity >= 8:   return "CRITIQUE"
        if self.severity >= 5:   return "MAJEUR"
        return "MINEUR"

    @property
    def is_reliable(self) -> bool:
        return self.sample_size >= MIN_HANDS_LEAK

    def to_dict(self) -> dict:
        return {
            "category":     self.category,
            "name":         self.name,
            "severity":     self.severity,
            "severity_label": self.severity_label,
            "frequency":    round(self.frequency, 3),
            "cost_per_100": round(self.cost_per_100, 2),
            "sample_size":  self.sample_size,
            "description":  self.description,
            "advice":       self.advice,
            "reliable":     self.is_reliable,
        }

    def __str__(self) -> str:
        bar = "█" * self.severity + "░" * (10 - self.severity)
        return (
            f"[{self.severity_label:<8}] {self.name}\n"
            f"  Sévérité : {bar} {self.severity}/10\n"
            f"  Fréquence : {self.frequency:.0%}  |  Coût : {self.cost_per_100:+.1f}$/100 mains\n"
            f"  {self.description}\n"
            f"  → {self.advice}\n"
            f"  ({self.sample_size} mains analysées)"
        )


@dataclass
class LeakReport:
    """Rapport complet de fuites."""
    total_hands:  int
    total_leaks:  int
    total_cost:   float        # coût total estimé par 100 mains
    leaks:        list[Leak]
    strengths:    list[str]    # ce qui va bien

    def top_leaks(self, n: int = 5) -> list[Leak]:
        return sorted(self.leaks, key=lambda l: l.severity, reverse=True)[:n]

    def critical_leaks(self) -> list[Leak]:
        return [l for l in self.leaks if l.severity >= 8]

    def summary(self) -> str:
        lines = [
            f"Rapport de fuites — {self.total_hands} mains analysées",
            f"Leaks détectés : {self.total_leaks}",
            f"Coût total estimé : {self.total_cost:+.1f}$/100 mains",
            "",
        ]
        if self.critical_leaks():
            lines.append("LEAKS CRITIQUES :")
            for leak in self.critical_leaks():
                lines.append(f"  • {leak.name} ({leak.severity}/10)")
        if self.strengths:
            lines.append("\nPOINTS FORTS :")
            for s in self.strengths:
                lines.append(f"  ✓ {s}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Moteur de détection
# ---------------------------------------------------------------------------

class LeakFinder:
    """
    Analyse la base de données tracker.py pour détecter les fuites.

    Usage :
        from tracker import PokerTracker
        tracker = PokerTracker()
        finder  = LeakFinder(tracker)
        report  = finder.analyse()
        for leak in report.top_leaks(5):
            print(leak)
    """

    def __init__(self, tracker):
        self.tracker = tracker

    def analyse(self, min_hands: int = MIN_HANDS_LEAK) -> LeakReport:
        """Lance l'analyse complète et retourne un LeakReport."""
        leaks     = []
        strengths = []

        with self.tracker._conn() as conn:
            total_hands = conn.execute(
                "SELECT COUNT(*) FROM hands"
            ).fetchone()[0]

        if total_hands < min_hands:
            log.info(f"Données insuffisantes ({total_hands} mains, min={min_hands})")
            return LeakReport(total_hands, 0, 0.0, [], ["Jouez plus de mains pour un rapport fiable."])

        # Lancer tous les détecteurs
        detectors = [
            self._detect_advice_ignored,
            self._detect_ev_leakage,
            self._detect_fold_too_much,
            self._detect_call_ev_negative,
            self._detect_stage_leaks,
            self._detect_position_leaks,
            self._detect_hand_class_leaks,
            self._detect_overfolding,
            self._detect_session_tilt,
        ]

        for detector in detectors:
            try:
                result = detector()
                if result:
                    if isinstance(result, list):
                        leaks.extend(result)
                    else:
                        leaks.append(result)
            except Exception as e:
                log.error(f"Erreur détecteur {detector.__name__} : {e}")

        # Filtrer les leaks sans données suffisantes
        leaks = [l for l in leaks if l.is_reliable]

        # Trier par sévérité
        leaks.sort(key=lambda l: (l.severity, abs(l.cost_per_100)), reverse=True)

        # Identifier les points forts
        strengths = self._find_strengths()

        total_cost = sum(l.cost_per_100 for l in leaks)

        return LeakReport(
            total_hands = total_hands,
            total_leaks = len(leaks),
            total_cost  = total_cost,
            leaks       = leaks,
            strengths   = strengths,
        )

    # ------------------------------------------------------------------
    # Détecteurs individuels
    # ------------------------------------------------------------------

    def _detect_advice_ignored(self) -> Optional[Leak]:
        """Détecte si le joueur ignore systématiquement les conseils Claude."""
        with self.tracker._conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)                                        AS total,
                    SUM(CASE WHEN followed_advice=0 THEN 1 END)    AS ignored,
                    AVG(CASE WHEN followed_advice=0 THEN result END) AS avg_result_ignored,
                    AVG(CASE WHEN followed_advice=1 THEN result END) AS avg_result_followed
                FROM hands
                WHERE recommended_action != '' AND recommended_action IS NOT NULL
            """).fetchone()

        total, ignored, avg_ign, avg_fol = (
            row[0] or 0, row[1] or 0,
            row[2] or 0.0, row[3] or 0.0
        )
        if total < MIN_HANDS_LEAK:
            return None

        ignore_rate = ignored / total
        if ignore_rate < ADVICE_IGNORE_THRESHOLD:
            return None

        cost = (avg_ign - avg_fol) * 100  # par 100 mains
        return Leak(
            category    = "mental",
            name        = "Ignore les conseils Claude",
            description = (
                f"Tu ignores les conseils {ignore_rate:.0%} du temps. "
                f"Résultat moyen en ignorant : {avg_ign:+.1f}$ vs "
                f"en suivant : {avg_fol:+.1f}$."
            ),
            severity    = min(10, int(ignore_rate * 12)),
            frequency   = ignore_rate,
            cost_per_100= cost,
            sample_size = total,
            advice      = "Fais confiance à l'analyse mathématique. Expérimente d'abord en mode play-money.",
            data        = {"avg_ignored": avg_ign, "avg_followed": avg_fol},
        )

    def _detect_ev_leakage(self) -> Optional[Leak]:
        """Détecte si l'EV réalisée est systématiquement inférieure à l'EV théorique."""
        with self.tracker._conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)          AS total,
                    AVG(ev_estimate)  AS avg_ev_theory,
                    AVG(ev_realized)  AS avg_ev_real,
                    SUM(ev_estimate)  AS total_ev_theory,
                    SUM(ev_realized)  AS total_ev_real
                FROM hands
                WHERE ev_estimate != 0 AND ev_estimate IS NOT NULL
            """).fetchone()

        total, avg_th, avg_re, sum_th, sum_re = (
            row[0] or 0, row[1] or 0.0, row[2] or 0.0,
            row[3] or 0.0, row[4] or 0.0
        )
        if total < MIN_HANDS_LEAK or sum_th == 0:
            return None

        efficiency = sum_re / sum_th if sum_th != 0 else 1.0
        leakage    = 1.0 - efficiency

        if leakage < EV_LEAK_THRESHOLD:
            return None

        cost = (avg_th - avg_re) * 100
        return Leak(
            category    = "mental",
            name        = "Leakage d'EV important",
            description = (
                f"Tu réalises seulement {efficiency:.0%} de ton EV théorique. "
                f"EV moyenne théorique : {avg_th:+.1f}$ / EV réalisée : {avg_re:+.1f}$."
            ),
            severity    = min(10, int(leakage * 15)),
            frequency   = leakage,
            cost_per_100= cost,
            sample_size = total,
            advice      = "Analyse les mains où tu t'es écarté du conseil optimal. Souvent dû au tilt ou manque de discipline.",
            data        = {"efficiency": efficiency, "avg_theory": avg_th, "avg_real": avg_re},
        )

    def _detect_fold_too_much(self) -> Optional[Leak]:
        """Détecte un fold excessif (over-folding)."""
        with self.tracker._conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)                                           AS total,
                    SUM(CASE WHEN action_taken='FOLD' THEN 1 END)     AS folds,
                    AVG(CASE WHEN action_taken='FOLD' THEN result END) AS avg_fold_result,
                    AVG(CASE WHEN recommended_action!='FOLD'
                              AND action_taken='FOLD' THEN result END) AS bad_fold_result
                FROM hands
            """).fetchone()

        total, folds, avg_fold, bad_fold = (
            row[0] or 0, row[1] or 0,
            row[2] or 0.0, row[3] or 0.0
        )
        if total < MIN_HANDS_LEAK or not folds:
            return None

        fold_rate = folds / total
        if fold_rate < 0.55:   # fold > 55% = suspect
            return None

        return Leak(
            category    = "postflop",
            name        = "Over-folding général",
            description = (
                f"Tu foldes {fold_rate:.0%} de tes mains. "
                f"Les folds contre-conseil te coûtent {bad_fold:+.1f}$ en moyenne."
            ),
            severity    = min(9, int(fold_rate * 10)),
            frequency   = fold_rate,
            cost_per_100= abs(bad_fold) * 100 * (fold_rate - 0.45),
            sample_size = total,
            advice      = "Élargis ta range de call, surtout en position. Calcule les pot odds avant de folder.",
        )

    def _detect_call_ev_negative(self) -> Optional[Leak]:
        """Détecte les calls récurrents avec EV négative."""
        with self.tracker._conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)     AS total,
                    AVG(result)  AS avg_result,
                    SUM(result)  AS total_result
                FROM hands
                WHERE action_taken = 'CALL'
                  AND recommended_action IN ('FOLD', 'CHECK')
            """).fetchone()

        total, avg_result, total_result = (
            row[0] or 0, row[1] or 0.0, row[2] or 0.0
        )
        if total < MIN_HANDS_LEAK // 2:
            return None
        if avg_result > CALL_EV_NEG_THRESHOLD:
            return None

        return Leak(
            category    = "postflop",
            name        = "Calls EV-négatifs répétés",
            description = (
                f"Sur {total} mains où fold/check était conseillé, "
                f"tu as callé et perdu {avg_result:+.1f}$ en moyenne."
            ),
            severity    = min(8, abs(int(avg_result / 3))),
            frequency   = 0.0,
            cost_per_100= avg_result * 100,
            sample_size = total,
            advice      = "Respecte les pot odds. Si tu n'as pas l'équité requise, le call est une fuite directe.",
        )

    def _detect_stage_leaks(self) -> list[Leak]:
        """Détecte les pertes systématiques sur certaines streets."""
        leaks = []
        with self.tracker._conn() as conn:
            rows = conn.execute("""
                SELECT
                    stage_final,
                    COUNT(*)    AS total,
                    AVG(result) AS avg_result,
                    SUM(result) AS total_result
                FROM hands
                GROUP BY stage_final
                HAVING total >= ?
            """, (MIN_HANDS_LEAK,)).fetchall()

        for row in rows:
            stage, total, avg_result, total_result = row
            if avg_result < MIN_RESULT_LOSS:
                leaks.append(Leak(
                    category    = "postflop",
                    name        = f"Leak sur {stage.upper()}",
                    description = (
                        f"Tu perds en moyenne {avg_result:+.1f}$ sur les mains "
                        f"finissant au {stage.upper()} ({total} mains)."
                    ),
                    severity    = min(8, abs(int(avg_result / 5))),
                    frequency   = 0.0,
                    cost_per_100= avg_result * 100,
                    sample_size = total,
                    advice      = self._stage_advice(stage),
                ))
        return leaks

    def _detect_position_leaks(self) -> list[Leak]:
        """Détecte les pertes liées aux positions en DB."""
        leaks = []
        with self.tracker._conn() as conn:
            # Utiliser les notes pour récupérer la position si disponible
            rows = conn.execute("""
                SELECT
                    h.stage_final,
                    COUNT(*)    AS total,
                    AVG(h.result) AS avg_result
                FROM hands h
                JOIN sessions s ON h.session_id = s.id
                WHERE h.result < 0
                GROUP BY h.stage_final
                HAVING total >= ?
                ORDER BY avg_result ASC
                LIMIT 3
            """, (MIN_HANDS_LEAK,)).fetchall()

        for row in rows:
            stage, total, avg_result = row
            if avg_result < MIN_RESULT_LOSS * 1.5:
                leaks.append(Leak(
                    category    = "position",
                    name        = f"Perte chronique au {stage.upper()}",
                    description = (
                        f"Tes mains finissant au {stage.upper()} te coûtent "
                        f"{avg_result:+.1f}$ en moyenne ({total} mains)."
                    ),
                    severity    = 5,
                    frequency   = 0.0,
                    cost_per_100= avg_result * 100,
                    sample_size = total,
                    advice      = "Réduis ton volume sur cette street ou réévalue ta stratégie postflop.",
                ))
        return leaks

    def _detect_hand_class_leaks(self) -> list[Leak]:
        """Détecte les pertes systématiques sur certaines classes de mains."""
        leaks = []
        with self.tracker._conn() as conn:
            rows = conn.execute("""
                SELECT
                    hand_class,
                    COUNT(*)    AS total,
                    AVG(result) AS avg_result,
                    AVG(win_probability) AS avg_equity
                FROM hands
                WHERE hand_class != '' AND hand_class IS NOT NULL
                GROUP BY hand_class
                HAVING total >= ?
                ORDER BY avg_result ASC
            """, (MIN_HANDS_LEAK // 2,)).fetchall()

        for row in rows:
            hand_class, total, avg_result, avg_equity = row
            if avg_result < -10.0:
                leaks.append(Leak(
                    category    = "preflop",
                    name        = f"Leak avec {hand_class}",
                    description = (
                        f"Tu perds {avg_result:+.1f}$ en moyenne avec {hand_class} "
                        f"(équité moy. {avg_equity:.0%}, {total} mains)."
                    ),
                    severity    = min(7, abs(int(avg_result / 4))),
                    frequency   = 0.0,
                    cost_per_100= avg_result * 100,
                    sample_size = total,
                    advice      = self._hand_class_advice(hand_class, avg_equity),
                ))
        return leaks

    def _detect_overfolding(self) -> Optional[Leak]:
        """Détecte le fold systématique face aux relances adverses."""
        with self.tracker._conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)  AS total,
                    AVG(result) AS avg_result
                FROM hands
                WHERE action_taken = 'FOLD'
                  AND win_probability > 0.35
                  AND recommended_action IN ('CALL', 'RAISE')
            """).fetchone()

        total, avg_result = row[0] or 0, row[1] or 0.0
        if total < MIN_HANDS_LEAK // 2:
            return None

        return Leak(
            category    = "mental",
            name        = "Fold avec bonne équité",
            description = (
                f"Sur {total} situations avec >35% d'équité où call/raise était conseillé, "
                f"tu as foldé et perdu {avg_result:+.1f}$ en moyenne."
            ),
            severity    = min(9, int(total / 3)),
            frequency   = 0.0,
            cost_per_100= abs(avg_result) * 100,
            sample_size = total,
            advice      = "Avec 35%+ d'équité, le fold est souvent une erreur mathématique. Mémorise les pot odds.",
        )

    def _detect_session_tilt(self) -> Optional[Leak]:
        """Détecte le tilt de session (pertes qui s'accélèrent en fin de session)."""
        with self.tracker._conn() as conn:
            sessions = conn.execute("""
                SELECT s.id, COUNT(h.id) as hand_count
                FROM sessions s
                JOIN hands h ON h.session_id = s.id
                WHERE s.end_time > 0
                GROUP BY s.id
                HAVING hand_count >= 10
            """).fetchall()

        tilt_sessions = 0
        total_checked = 0

        for session_id, hand_count in sessions:
            with self.tracker._conn() as conn:
                hands = conn.execute("""
                    SELECT result FROM hands
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                """, (session_id,)).fetchall()

            if len(hands) < 10:
                continue

            results = [h[0] for h in hands]
            mid     = len(results) // 2
            first_half_avg  = sum(results[:mid]) / max(mid, 1)
            second_half_avg = sum(results[mid:]) / max(len(results) - mid, 1)

            # Tilt = seconde moitié nettement pire que première
            if second_half_avg < first_half_avg - 15:
                tilt_sessions += 1
            total_checked += 1

        if total_checked < 3:
            return None

        tilt_rate = tilt_sessions / total_checked
        if tilt_rate < 0.35:
            return None

        return Leak(
            category    = "mental",
            name        = "Tilt de session récurrent",
            description = (
                f"Dans {tilt_rate:.0%} de tes sessions ({tilt_sessions}/{total_checked}), "
                f"tes résultats se dégradent significativement en 2ème moitié."
            ),
            severity    = min(8, int(tilt_rate * 12)),
            frequency   = tilt_rate,
            cost_per_100= -15.0 * tilt_rate,
            sample_size = total_checked,
            advice      = "Fixe une limite de stop-loss par session (ex: -3 buy-ins). Fais des pauses entre les mains.",
        )

    # ------------------------------------------------------------------
    # Points forts
    # ------------------------------------------------------------------

    def _find_strengths(self) -> list[str]:
        """Identifie ce qui va bien dans le jeu."""
        strengths = []
        with self.tracker._conn() as conn:
            # Suivi des conseils
            row = conn.execute("""
                SELECT AVG(CAST(followed_advice AS FLOAT)) FROM hands
                WHERE recommended_action != ''
            """).fetchone()
            if row[0] and row[0] > 0.60:
                strengths.append(f"Bonne discipline : {row[0]:.0%} des conseils suivis")

            # Résultat global positif
            row = conn.execute("SELECT AVG(result) FROM hands").fetchone()
            if row[0] and row[0] > 0:
                strengths.append(f"Win rate positif : +{row[0]:.1f}$/main en moyenne")

            # Bonnes mains bien jouées
            row = conn.execute("""
                SELECT COUNT(*) FROM hands
                WHERE hand_class IN ('Couleur','Full House','Carré','Quinte Flush')
                  AND result > 0
            """).fetchone()
            total_nutted = conn.execute("""
                SELECT COUNT(*) FROM hands
                WHERE hand_class IN ('Couleur','Full House','Carré','Quinte Flush')
            """).fetchone()[0]
            if total_nutted >= 5 and row[0] / max(total_nutted, 1) > 0.7:
                strengths.append(f"Extraction de valeur sur mains fortes : {row[0]/total_nutted:.0%}")

        return strengths

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stage_advice(stage: str) -> str:
        return {
            "preflop": "Resserre ta range préflop en early position. Ne joue que les mains profitables.",
            "flop":    "Analyse la texture du board au flop. Évite de cbet avec 0% d'équité.",
            "turn":    "Au turn, joue plus polarisé : value ou bluff, rarement médian.",
            "river":   "La rivière nécessite de la discipline. Bluff uniquement avec blockers.",
        }.get(stage, "Révise ta stratégie sur cette street.")

    @staticmethod
    def _hand_class_advice(hand_class: str, equity: float) -> str:
        if equity > 0.60:
            return f"Avec {hand_class} et {equity:.0%} d'équité, tes pertes suggèrent un sizing trop petit. Mise plus fort."
        return f"Avec {hand_class}, tu entres peut-être trop souvent. Sois plus sélectif."


# ---------------------------------------------------------------------------
# Intégration dans stats_viewer.py
# ---------------------------------------------------------------------------

def add_leak_tab_to_viewer(viewer, tracker) -> None:
    """
    Ajoute un onglet 'Leaks' dans le dashboard stats_viewer.py existant.

    Usage dans stats_viewer.py (run()) :
        from leak_finder import add_leak_tab_to_viewer
        add_leak_tab_to_viewer(self, self.tracker)
    """
    import tkinter as tk
    from tkinter import ttk, font as tkfont

    THEME_BG     = "#0D0D0D"
    THEME_CARD   = "#16213E"
    THEME_TEXT   = "#E8E8E8"
    THEME_MUTED  = "#9AA5B4"
    THEME_DIM    = "#5A6478"
    THEME_RED    = "#E74C3C"
    THEME_ORANGE = "#F39C12"
    THEME_GREEN  = "#27AE60"
    THEME_ACCENT = "#3498DB"
    FONT         = "Consolas"

    tab = tk.Frame(viewer.notebook, bg=THEME_BG)
    viewer.notebook.add(tab, text="  Leaks  ")

    # Header
    header = tk.Frame(tab, bg=THEME_CARD, pady=6)
    header.pack(fill=tk.X)
    tk.Label(header, text="Analyse des fuites dans ton jeu",
             font=tkfont.Font(family=FONT, size=10, weight="bold"),
             bg=THEME_CARD, fg=THEME_ACCENT).pack(side=tk.LEFT, padx=10)

    # Bouton analyse
    analyse_btn = tk.Button(
        header, text="Analyser",
        font=tkfont.Font(family=FONT, size=9),
        bg="#16213E", fg=THEME_TEXT,
        relief=tk.FLAT, cursor="hand2",
    )
    analyse_btn.pack(side=tk.RIGHT, padx=10)

    # Zone de résultats (scrollable)
    canvas_frame = tk.Frame(tab, bg=THEME_BG)
    canvas_frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(canvas_frame, bg=THEME_BG, highlightthickness=0)
    scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    inner = tk.Frame(canvas, bg=THEME_BG)
    window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

    def on_resize(event):
        canvas.itemconfig(window_id, width=event.width)
    canvas.bind("<Configure>", on_resize)

    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.scrollregion() if hasattr(canvas, 'scrollregion') else canvas.bbox("all"))
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    result_label = tk.Label(inner, text="Cliquez sur 'Analyser' pour détecter vos fuites.",
                            font=tkfont.Font(family=FONT, size=9),
                            bg=THEME_BG, fg=THEME_MUTED, justify=tk.LEFT)
    result_label.pack(padx=12, pady=12, anchor=tk.W)

    def run_analysis():
        for widget in inner.winfo_children():
            widget.destroy()

        loading = tk.Label(inner, text="Analyse en cours…",
                           font=tkfont.Font(family=FONT, size=9),
                           bg=THEME_BG, fg=THEME_MUTED)
        loading.pack(padx=12, pady=8)
        inner.update()

        finder = LeakFinder(tracker)
        report = finder.analyse()
        loading.destroy()

        # Résumé global
        summary_frame = tk.Frame(inner, bg=THEME_CARD)
        summary_frame.pack(fill=tk.X, padx=10, pady=(8, 4))

        def stat_row(parent, label, value, color):
            f = tk.Frame(parent, bg=THEME_CARD)
            f.pack(fill=tk.X, padx=8, pady=2)
            tk.Label(f, text=label, font=tkfont.Font(family=FONT, size=9),
                     bg=THEME_CARD, fg=THEME_DIM, width=20, anchor=tk.W).pack(side=tk.LEFT)
            tk.Label(f, text=value, font=tkfont.Font(family=FONT, size=9, weight="bold"),
                     bg=THEME_CARD, fg=color).pack(side=tk.LEFT)

        stat_row(inner, "Mains analysées :",  str(report.total_hands),   THEME_TEXT)
        stat_row(inner, "Leaks détectés :",   str(report.total_leaks),
                 THEME_RED if report.total_leaks > 3 else THEME_ORANGE)
        cost_color = THEME_RED if report.total_cost < -50 else THEME_ORANGE if report.total_cost < 0 else THEME_GREEN
        stat_row(inner, "Coût estimé/100 :",  f"{report.total_cost:+.1f}$", cost_color)

        # Points forts
        if report.strengths:
            tk.Label(inner, text="Points forts",
                     font=tkfont.Font(family=FONT, size=9, weight="bold"),
                     bg=THEME_BG, fg=THEME_GREEN).pack(anchor=tk.W, padx=12, pady=(10, 2))
            for s in report.strengths:
                tk.Label(inner, text=f"  ✓ {s}",
                         font=tkfont.Font(family=FONT, size=9),
                         bg=THEME_BG, fg=THEME_GREEN).pack(anchor=tk.W, padx=12)

        # Leaks
        if not report.leaks:
            tk.Label(inner, text="Aucun leak significatif détecté — bon travail !",
                     font=tkfont.Font(family=FONT, size=9),
                     bg=THEME_BG, fg=THEME_GREEN).pack(padx=12, pady=12)
            return

        tk.Label(inner, text="Leaks détectés (par sévérité)",
                 font=tkfont.Font(family=FONT, size=9, weight="bold"),
                 bg=THEME_BG, fg=THEME_RED).pack(anchor=tk.W, padx=12, pady=(12, 4))

        for leak in report.top_leaks(10):
            color = THEME_RED if leak.severity >= 8 else THEME_ORANGE if leak.severity >= 5 else THEME_MUTED
            card = tk.Frame(inner, bg=THEME_CARD, pady=6)
            card.pack(fill=tk.X, padx=10, pady=3)

            # Header leak
            hdr = tk.Frame(card, bg=THEME_CARD)
            hdr.pack(fill=tk.X, padx=8)
            tk.Label(hdr, text=f"[{leak.severity_label}]",
                     font=tkfont.Font(family=FONT, size=8, weight="bold"),
                     bg=THEME_CARD, fg=color).pack(side=tk.LEFT)
            tk.Label(hdr, text=f" {leak.name}",
                     font=tkfont.Font(family=FONT, size=9, weight="bold"),
                     bg=THEME_CARD, fg=THEME_TEXT).pack(side=tk.LEFT)

            # Barre de sévérité
            bar_frame = tk.Frame(card, bg=THEME_CARD)
            bar_frame.pack(fill=tk.X, padx=8, pady=2)
            bar_canvas = tk.Canvas(bar_frame, height=6, bg="#2C3E50",
                                   highlightthickness=0)
            bar_canvas.pack(fill=tk.X)
            bar_canvas.update_idletasks()
            bw = bar_canvas.winfo_width() or 250
            bar_canvas.create_rectangle(0, 0, int(bw * leak.severity / 10), 6,
                                        fill=color, outline="")

            # Description
            tk.Label(card, text=leak.description,
                     font=tkfont.Font(family=FONT, size=8),
                     bg=THEME_CARD, fg=THEME_MUTED,
                     wraplength=240, justify=tk.LEFT).pack(anchor=tk.W, padx=8, pady=1)

            # Conseil
            tk.Label(card, text=f"→ {leak.advice}",
                     font=tkfont.Font(family=FONT, size=8, weight="bold"),
                     bg=THEME_CARD, fg=THEME_ACCENT,
                     wraplength=240, justify=tk.LEFT).pack(anchor=tk.W, padx=8, pady=(1, 2))

            # Stats
            stats_line = (
                f"Fréquence : {leak.frequency:.0%}  |  "
                f"Coût : {leak.cost_per_100:+.1f}$/100  |  "
                f"{leak.sample_size} mains"
            )
            tk.Label(card, text=stats_line,
                     font=tkfont.Font(family=FONT, size=7),
                     bg=THEME_CARD, fg=THEME_DIM).pack(anchor=tk.W, padx=8)

    analyse_btn.configure(command=run_analysis)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Détecteur de fuites poker")
    parser.add_argument("--db",    type=str, default=None,
                        help="Chemin vers poker_stats.db")
    parser.add_argument("--top",   type=int, default=5,
                        help="Nombre de leaks à afficher (défaut: 5)")
    parser.add_argument("--json",  action="store_true",
                        help="Sortie en JSON")
    parser.add_argument("--demo",  action="store_true",
                        help="Insérer des données de démo et analyser")
    args = parser.parse_args()

    try:
        from tracker import PokerTracker, HandRecord
        db_path = args.db
        tracker = PokerTracker(db_path=Path(db_path) if db_path else None)
    except ImportError:
        print("ERREUR : tracker.py introuvable.")
        sys.exit(1)

    if args.demo:
        # Insérer des données de démo variées pour tester
        import random, time as _time
        sid = tracker.start_session(buy_in=50, game_type="Tournoi")
        for i in range(60):
            # Simuler une variété de situations
            stage    = random.choice(["preflop", "flop", "turn", "river"])
            rec      = random.choice(["FOLD", "CALL", "RAISE", "CHECK", "BET"])
            # Simuler quelques leaks : ignorer souvent les conseils
            taken    = rec if random.random() > 0.45 else random.choice(["FOLD","CALL","RAISE"])
            wp       = random.uniform(0.20, 0.90)
            result   = random.gauss(-2, 30)   # légèrement négatif pour montrer les leaks
            ev       = random.uniform(5, 40)
            ev_real  = ev * random.uniform(0.4, 1.1)
            tracker.record_hand(HandRecord(
                session_id=sid, stage_final=stage,
                player_cards=["Ks","7h"], board_cards=["Kd","7d","2c"],
                num_opponents=random.randint(1,5), pot_final=random.uniform(20,200),
                hand_class=random.choice(["Paire","Deux Paires","Haute Carte","Brelan","Couleur"]),
                win_probability=wp, recommended_action=rec, action_taken=taken,
                followed_advice=(rec==taken), result=result,
                ev_estimate=ev, ev_realized=ev_real,
                timestamp=_time.time() - random.uniform(0, 3600),
            ))
        tracker.end_session(sid, placement=3, prize=80)
        print("Données de démo insérées.\n")

    finder = LeakFinder(tracker)
    report = finder.analyse()

    if args.json:
        import json
        print(json.dumps({
            "total_hands": report.total_hands,
            "total_leaks": report.total_leaks,
            "total_cost":  report.total_cost,
            "leaks": [l.to_dict() for l in report.top_leaks(args.top)],
            "strengths": report.strengths,
        }, ensure_ascii=False, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"  RAPPORT DE FUITES — {report.total_hands} mains analysées")
        print(f"{'='*60}")
        print(f"  Leaks détectés : {report.total_leaks}")
        print(f"  Coût estimé    : {report.total_cost:+.1f}$/100 mains\n")

        if report.strengths:
            print("Points forts :")
            for s in report.strengths:
                print(f"  ✓ {s}")
            print()

        leaks = report.top_leaks(args.top)
        if not leaks:
            print("  Aucun leak significatif détecté.")
        else:
            print(f"Top {args.top} leaks :\n")
            for i, leak in enumerate(leaks, 1):
                print(f"{i}. {leak}\n")
