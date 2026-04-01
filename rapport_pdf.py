"""
rapport_pdf.py — Génération automatique de rapport PDF de session poker
Dépendances : reportlab  (pip install reportlab)
              tracker.py, leak_finder.py (déjà installés)

Génère un rapport PDF professionnel après chaque session contenant :
  Page 1 — Résumé de session
    • Titre, date, type de jeu, buy-in / prize / profit
    • KPIs : win rate, equity moyenne, taux de suivi des conseils
    • Graphique sparkline du profit par main

  Page 2 — Statistiques détaillées
    • Tableau de performance par classe de main
    • Distribution des actions (FOLD/CALL/RAISE/etc.)
    • Comparaison EV théorique vs EV réalisée

  Page 3 — Analyse des fuites (leak_finder)
    • Top 5 leaks détectés avec sévérité
    • Points forts
    • Conseils d'amélioration

Usage :
    generator = PDFReportGenerator(tracker)
    path = generator.generate_session_report(session_id=3)
    print(f"Rapport sauvegardé : {path}")

    # Automatique en fin de session (depuis main.py)
    path = generator.generate_latest_session()
"""

import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent / "rapports"

# ---------------------------------------------------------------------------
# Couleurs et styles (thème sombre poker → version papier sobre)
# ---------------------------------------------------------------------------

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm, mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak, KeepTogether,
    )
    from reportlab.graphics.shapes import Drawing, Rect, Line, String, Polygon
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics import renderPDF
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False
    log.warning("reportlab non installé — pip install reportlab")

# Palette
C_DARK      = colors.HexColor("#1A1A2E")
C_ACCENT    = colors.HexColor("#3498DB")
C_GREEN     = colors.HexColor("#27AE60")
C_RED       = colors.HexColor("#E74C3C")
C_ORANGE    = colors.HexColor("#F39C12")
C_LIGHT_BG  = colors.HexColor("#F4F6F9")
C_BORDER    = colors.HexColor("#BDC3C7")
C_TEXT      = colors.HexColor("#2C3E50")
C_MUTED     = colors.HexColor("#7F8C8D")
C_WHITE     = colors.white


# ---------------------------------------------------------------------------
# Générateur de rapport
# ---------------------------------------------------------------------------

class PDFReportGenerator:
    """
    Génère des rapports PDF de session poker.

    Usage :
        from tracker import PokerTracker
        tracker   = PokerTracker()
        generator = PDFReportGenerator(tracker)

        # Rapport d'une session spécifique
        path = generator.generate_session_report(session_id=5)

        # Rapport de la dernière session
        path = generator.generate_latest_session()

        # Rapport global (toutes sessions)
        path = generator.generate_global_report()
    """

    def __init__(self, tracker, reports_dir: Path = REPORTS_DIR):
        if not _HAS_REPORTLAB:
            raise ImportError("reportlab requis : pip install reportlab")
        self.tracker     = tracker
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(exist_ok=True)
        self._styles     = self._build_styles()

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def generate_session_report(self, session_id: int) -> Path:
        """Génère le rapport PDF d'une session spécifique."""
        stats = self.tracker.get_session_stats(session_id)
        if not stats:
            raise ValueError(f"Session #{session_id} introuvable.")

        hands = self._get_session_hands(session_id)
        leaks = self._get_leaks()

        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rapport_session_{session_id}_{date_str}.pdf"
        path     = self.reports_dir / filename

        self._build_pdf(path, stats, hands, leaks, mode="session")
        log.info(f"Rapport PDF généré : {path}")
        return path

    def generate_latest_session(self) -> Optional[Path]:
        """Génère le rapport de la dernière session clôturée."""
        sessions = self.tracker.get_recent_sessions(limit=1)
        if not sessions:
            log.warning("Aucune session trouvée.")
            return None
        return self.generate_session_report(sessions[0]["id"])

    def generate_global_report(self) -> Path:
        """Génère un rapport global de toutes les sessions."""
        stats  = self.tracker.get_global_stats()
        hands  = self._get_all_hands(limit=200)
        leaks  = self._get_leaks()

        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        path     = self.reports_dir / f"rapport_global_{date_str}.pdf"

        self._build_pdf(path, stats, hands, leaks, mode="global")
        log.info(f"Rapport global PDF généré : {path}")
        return path

    # ------------------------------------------------------------------
    # Construction du PDF
    # ------------------------------------------------------------------

    def _build_pdf(
        self,
        path:  Path,
        stats: dict,
        hands: list,
        leaks: object,
        mode:  str = "session",
    ) -> None:
        """Construit le document PDF complet."""
        doc = SimpleDocTemplate(
            str(path),
            pagesize    = A4,
            rightMargin = 2*cm,
            leftMargin  = 2*cm,
            topMargin   = 2.5*cm,
            bottomMargin= 2*cm,
            title       = "Rapport de Session Poker",
            author      = "Poker HUD Assistant",
        )

        story = []

        # ── Page 1 : Résumé ───────────────────────────────────────────
        story += self._page_summary(stats, hands, mode)
        story.append(PageBreak())

        # ── Page 2 : Statistiques détaillées ─────────────────────────
        story += self._page_stats(stats, hands)
        story.append(PageBreak())

        # ── Page 3 : Analyse des fuites ───────────────────────────────
        story += self._page_leaks(leaks)

        doc.build(story, onFirstPage=self._header_footer,
                         onLaterPages=self._header_footer)

    # ------------------------------------------------------------------
    # Page 1 — Résumé
    # ------------------------------------------------------------------

    def _page_summary(self, stats: dict, hands: list, mode: str) -> list:
        s = self._styles
        story = []

        # Titre principal
        title_text = (
            "Rapport de Session" if mode == "session"
            else "Rapport Global"
        )
        story.append(Paragraph(f"<font color='#3498DB'>♠</font> {title_text}", s["Title"]))
        story.append(Spacer(1, 4*mm))

        # Date et type
        date_now = datetime.now().strftime("%d %B %Y à %H:%M")
        game_type = stats.get("game_type", "Tournoi")
        story.append(Paragraph(f"{date_now}  —  {game_type}", s["Subtitle"]))
        story.append(HRFlowable(width="100%", thickness=1,
                                color=C_ACCENT, spaceAfter=6*mm))

        # KPIs en grille 2×3
        profit      = stats.get("profit", 0.0)
        win_rate    = stats.get("win_rate", stats.get("win_rate_hands", 0.0))
        follow_rate = stats.get("advice_follow_rate", 0.0)
        total_hands = stats.get("total_hands", 0)
        buy_in      = stats.get("buy_in", 0.0)
        prize       = stats.get("prize", 0.0)
        placement   = stats.get("placement", 0)
        duration    = stats.get("duration_minutes", 0)

        profit_color  = "#27AE60" if profit >= 0 else "#E74C3C"
        profit_sign   = "+" if profit >= 0 else ""

        kpi_data = [
            ["Buy-in", "Prize", "Profit"],
            [
                f"${buy_in:.0f}",
                f"${prize:.0f}" if prize else "—",
                f"<font color='{profit_color}'>{profit_sign}{profit:.2f}$</font>",
            ],
            ["Mains jouées", "Win rate", "Conseils suivis"],
            [
                str(total_hands),
                f"{win_rate:.1%}",
                f"{follow_rate:.1%}",
            ],
        ]

        if placement:
            kpi_data[0].append("Placement")
            kpi_data[1].append(f"#{placement}")
            kpi_data[2].append("Durée")
            kpi_data[3].append(f"{duration:.0f} min")

        kpi_table = Table(
            [[Paragraph(str(cell), s["KPILabel"] if i % 2 == 0 else s["KPIValue"])
              for cell in row]
             for i, row in enumerate(kpi_data)],
            colWidths=[4.2*cm] * (len(kpi_data[0])),
        )
        kpi_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, -1), C_LIGHT_BG),
            ("BACKGROUND",  (0, 0), (-1, 0),  C_DARK),
            ("BACKGROUND",  (0, 2), (-1, 2),  C_DARK),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  C_WHITE),
            ("TEXTCOLOR",   (0, 2), (-1, 2),  C_WHITE),
            ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_LIGHT_BG, colors.white]),
            ("GRID",        (0, 0), (-1, -1), 0.5, C_BORDER),
            ("TOPPADDING",  (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0,0), (-1, -1), 5),
        ]))
        story.append(kpi_table)
        story.append(Spacer(1, 8*mm))

        # Graphique profit par main (sparkline)
        if hands:
            story.append(Paragraph("Évolution du profit (main par main)", s["SectionTitle"]))
            story.append(Spacer(1, 2*mm))
            chart = self._build_profit_sparkline(hands)
            if chart:
                story.append(chart)
                story.append(Spacer(1, 4*mm))

        # EV théorique vs réalisée
        ev_theory  = stats.get("total_ev_estimate", 0.0)
        ev_real    = stats.get("total_ev_realized", profit)
        if ev_theory != 0:
            efficiency = ev_real / ev_theory * 100 if ev_theory else 0
            eff_color  = "#27AE60" if efficiency >= 75 else "#E74C3C"
            story.append(Paragraph(
                f"EV théorique : <b>{ev_theory:+.1f}$</b>  —  "
                f"EV réalisée : <b>{ev_real:+.1f}$</b>  —  "
                f"Efficacité : <font color='{eff_color}'><b>{efficiency:.0f}%</b></font>",
                s["BodyCenter"]
            ))

        return story

    # ------------------------------------------------------------------
    # Page 2 — Statistiques détaillées
    # ------------------------------------------------------------------

    def _page_stats(self, stats: dict, hands: list) -> list:
        s = self._styles
        story = []

        story.append(Paragraph("Statistiques détaillées", s["PageTitle"]))
        story.append(HRFlowable(width="100%", thickness=1,
                                color=C_ACCENT, spaceAfter=5*mm))

        # Performance par classe de main
        hand_perf = self._compute_hand_class_perf(hands)
        if hand_perf:
            story.append(Paragraph("Performance par classe de main", s["SectionTitle"]))
            story.append(Spacer(1, 2*mm))

            table_data = [["Classe de main", "Mains", "Profit total", "Moy./main", "Win rate"]]
            for hc, data in sorted(hand_perf.items(), key=lambda x: -x[1]["profit"]):
                profit  = data["profit"]
                count   = data["count"]
                wins    = data["wins"]
                color   = "#27AE60" if profit >= 0 else "#E74C3C"
                sign    = "+" if profit >= 0 else ""
                table_data.append([
                    hc,
                    str(count),
                    Paragraph(f"<font color='{color}'>{sign}{profit:.1f}$</font>", s["TableCell"]),
                    Paragraph(f"<font color='{color}'>{sign}{profit/max(count,1):.1f}$</font>", s["TableCell"]),
                    f"{wins/max(count,1):.0%}",
                ])

            t = Table(table_data, colWidths=[5*cm, 2*cm, 3.5*cm, 3*cm, 3*cm])
            t.setStyle(self._base_table_style())
            story.append(t)
            story.append(Spacer(1, 6*mm))

        # Distribution des actions
        action_dist = self._compute_action_distribution(hands)
        if action_dist:
            story.append(Paragraph("Distribution des actions jouées", s["SectionTitle"]))
            story.append(Spacer(1, 2*mm))

            action_data = [["Action", "Occurrences", "% du total", "Résultat moy."]]
            total_actions = sum(d["count"] for d in action_dist.values())
            for action, data in sorted(action_dist.items(), key=lambda x: -x[1]["count"]):
                pct    = data["count"] / max(total_actions, 1)
                avg_r  = data["avg_result"]
                color  = "#27AE60" if avg_r >= 0 else "#E74C3C"
                sign   = "+" if avg_r >= 0 else ""
                action_data.append([
                    action,
                    str(data["count"]),
                    f"{pct:.1%}",
                    Paragraph(f"<font color='{color}'>{sign}{avg_r:.1f}$</font>", s["TableCell"]),
                ])

            t2 = Table(action_data, colWidths=[4*cm, 3*cm, 4*cm, 4.6*cm])
            t2.setStyle(self._base_table_style())
            story.append(t2)
            story.append(Spacer(1, 6*mm))

        # Graphique en barres des actions
        if action_dist:
            story.append(Paragraph("Profit moyen par action", s["SectionTitle"]))
            story.append(Spacer(1, 2*mm))
            bar_chart = self._build_action_bar_chart(action_dist)
            if bar_chart:
                story.append(bar_chart)

        return story

    # ------------------------------------------------------------------
    # Page 3 — Leaks
    # ------------------------------------------------------------------

    def _page_leaks(self, leaks) -> list:
        s = self._styles
        story = []

        story.append(Paragraph("Analyse des fuites (Leaks)", s["PageTitle"]))
        story.append(HRFlowable(width="100%", thickness=1,
                                color=C_ACCENT, spaceAfter=5*mm))

        if leaks is None or leaks.total_hands < 10:
            story.append(Paragraph(
                "Données insuffisantes pour l'analyse des fuites. "
                "Jouez au moins 20 mains pour obtenir un rapport complet.",
                s["Body"]
            ))
            return story

        # Résumé leaks
        cost_color = "#E74C3C" if leaks.total_cost < -20 else "#F39C12" if leaks.total_cost < 0 else "#27AE60"
        story.append(Paragraph(
            f"<b>{leaks.total_leaks}</b> fuites détectées sur "
            f"<b>{leaks.total_hands}</b> mains — "
            f"Coût estimé : "
            f"<font color='{cost_color}'><b>{leaks.total_cost:+.1f}$/100 mains</b></font>",
            s["BodyCenter"]
        ))
        story.append(Spacer(1, 4*mm))

        # Points forts
        if leaks.strengths:
            story.append(Paragraph("Points forts", s["SectionTitle"]))
            for strength in leaks.strengths:
                story.append(Paragraph(f"✓  {strength}", s["StrengthItem"]))
            story.append(Spacer(1, 4*mm))

        # Top leaks
        top = leaks.top_leaks(5)
        if not top:
            story.append(Paragraph(
                "Aucun leak significatif détecté — excellent travail !",
                s["Body"]
            ))
            return story

        story.append(Paragraph("Top 5 fuites à corriger", s["SectionTitle"]))
        story.append(Spacer(1, 2*mm))

        for i, leak in enumerate(top, 1):
            sev_color = (
                "#E74C3C" if leak.severity >= 8 else
                "#F39C12" if leak.severity >= 5 else
                "#7F8C8D"
            )
            bar_filled = "█" * leak.severity
            bar_empty  = "░" * (10 - leak.severity)

            block = [
                Paragraph(
                    f"<b>{i}. {leak.name}</b> "
                    f"<font color='{sev_color}'>[{leak.severity_label} — {leak.severity}/10]</font>",
                    s["LeakTitle"]
                ),
                Paragraph(
                    f"<font color='{sev_color}'>{bar_filled}</font>{bar_empty}  "
                    f"Fréquence : {leak.frequency:.0%}  |  "
                    f"Coût : {leak.cost_per_100:+.1f}$/100 mains  |  "
                    f"{leak.sample_size} mains",
                    s["LeakMeta"]
                ),
                Paragraph(leak.description, s["LeakDescription"]),
                Paragraph(f"→  {leak.advice}", s["LeakAdvice"]),
                Spacer(1, 3*mm),
            ]
            story.append(KeepTogether(block))

        return story

    # ------------------------------------------------------------------
    # Graphiques
    # ------------------------------------------------------------------

    def _build_profit_sparkline(self, hands: list):
        """Construit un graphique linéaire du profit cumulé main par main."""
        if not hands:
            return None

        results  = [h["result"] for h in hands]
        cumul    = []
        running  = 0.0
        for r in results:
            running += r
            cumul.append(running)

        if not cumul:
            return None

        W, H = 16*cm, 5*cm
        drawing = Drawing(W, H)

        pad_l, pad_r, pad_t, pad_b = 40, 10, 10, 30
        chart_w = W - pad_l - pad_r
        chart_h = H - pad_t - pad_b

        min_v = min(cumul + [0])
        max_v = max(cumul + [0])
        span  = max(max_v - min_v, 1)

        def xp(i):
            return pad_l + i * chart_w / max(len(cumul) - 1, 1)
        def yp(v):
            return pad_b + (v - min_v) / span * chart_h

        # Ligne zéro
        y0 = yp(0)
        drawing.add(Line(pad_l, y0, pad_l + chart_w, y0,
                         strokeColor=C_BORDER, strokeWidth=0.5,
                         strokeDashArray=[3, 3]))

        # Axe Y labels
        for v in [min_v, 0, max_v]:
            y = yp(v)
            sign = "+" if v > 0 else ""
            lbl = String(pad_l - 4, y - 3, f"{sign}{v:.0f}$",
                         fontSize=6, fillColor=C_MUTED,
                         textAnchor="end")
            drawing.add(lbl)

        # Ligne profit cumulé
        points = [(xp(i), yp(v)) for i, v in enumerate(cumul)]
        for i in range(len(points) - 1):
            color = C_GREEN if cumul[i+1] >= 0 else C_RED
            drawing.add(Line(
                points[i][0], points[i][1],
                points[i+1][0], points[i+1][1],
                strokeColor=color, strokeWidth=1.5,
            ))

        # Valeur finale
        if points:
            lx, ly = points[-1]
            final  = cumul[-1]
            fc     = C_GREEN if final >= 0 else C_RED
            sign   = "+" if final >= 0 else ""
            drawing.add(String(lx + 3, ly - 3, f"{sign}{final:.1f}$",
                               fontSize=7, fillColor=fc))

        return drawing

    def _build_action_bar_chart(self, action_dist: dict):
        """Graphique en barres du profit moyen par action."""
        if not action_dist:
            return None

        W, H = 14*cm, 5*cm
        drawing = Drawing(W, H)

        actions = list(action_dist.keys())
        values  = [action_dist[a]["avg_result"] for a in actions]
        n       = len(actions)

        if n == 0:
            return None

        pad_l, pad_r, pad_t, pad_b = 45, 10, 10, 25
        chart_w = W - pad_l - pad_r
        chart_h = H - pad_t - pad_b
        bar_w   = chart_w / n * 0.6
        bar_gap = chart_w / n

        max_abs = max(abs(v) for v in values) if values else 1
        span    = max_abs * 2

        def yp(v):
            return pad_b + (v + max_abs) / span * chart_h

        y0 = yp(0)
        drawing.add(Line(pad_l, y0, pad_l + chart_w, y0,
                         strokeColor=C_BORDER, strokeWidth=0.5))

        for i, (action, value) in enumerate(zip(actions, values)):
            cx    = pad_l + i * bar_gap + bar_gap / 2
            color = C_GREEN if value >= 0 else C_RED
            y_bar = yp(value)
            y_bot = y0
            if value < 0:
                y_bar, y_bot = y0, yp(value)

            drawing.add(Rect(
                cx - bar_w/2, min(y_bar, y_bot),
                bar_w, abs(y_bar - y_bot),
                fillColor=color, strokeColor=None,
            ))
            # Label action
            drawing.add(String(cx, pad_b - 12, action[:4],
                               fontSize=6, fillColor=C_TEXT,
                               textAnchor="middle"))
            # Valeur
            sign = "+" if value >= 0 else ""
            drawing.add(String(cx, max(y_bar, y_bot) + 2,
                               f"{sign}{value:.1f}",
                               fontSize=5, fillColor=color,
                               textAnchor="middle"))

        return drawing

    # ------------------------------------------------------------------
    # Helpers données
    # ------------------------------------------------------------------

    def _get_session_hands(self, session_id: int) -> list:
        with self.tracker._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM hands WHERE session_id=? ORDER BY timestamp ASC",
                (session_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def _get_all_hands(self, limit: int = 200) -> list:
        with self.tracker._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM hands ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def _get_leaks(self):
        try:
            from leak_finder import LeakFinder
            finder = LeakFinder(self.tracker)
            return finder.analyse()
        except Exception as e:
            log.warning(f"LeakFinder indisponible : {e}")
            return None

    def _compute_hand_class_perf(self, hands: list) -> dict:
        perf = {}
        for h in hands:
            hc = h.get("hand_class") or "Inconnue"
            if hc not in perf:
                perf[hc] = {"count": 0, "profit": 0.0, "wins": 0}
            perf[hc]["count"]  += 1
            perf[hc]["profit"] += h.get("result", 0.0)
            if h.get("result", 0) > 0:
                perf[hc]["wins"] += 1
        return perf

    def _compute_action_distribution(self, hands: list) -> dict:
        dist = {}
        for h in hands:
            action = h.get("action_taken") or "UNKNOWN"
            if action not in dist:
                dist[action] = {"count": 0, "total": 0.0, "avg_result": 0.0}
            dist[action]["count"] += 1
            dist[action]["total"] += h.get("result", 0.0)
        for action in dist:
            dist[action]["avg_result"] = (
                dist[action]["total"] / dist[action]["count"]
            )
        return dist

    # ------------------------------------------------------------------
    # Styles
    # ------------------------------------------------------------------

    def _build_styles(self) -> dict:
        base = getSampleStyleSheet()
        return {
            "Title": ParagraphStyle("Title",
                fontSize=22, textColor=C_DARK,
                fontName="Helvetica-Bold", spaceAfter=2*mm,
                alignment=TA_LEFT),
            "PageTitle": ParagraphStyle("PageTitle",
                fontSize=16, textColor=C_DARK,
                fontName="Helvetica-Bold", spaceAfter=2*mm),
            "Subtitle": ParagraphStyle("Subtitle",
                fontSize=10, textColor=C_MUTED,
                fontName="Helvetica", spaceAfter=2*mm),
            "SectionTitle": ParagraphStyle("SectionTitle",
                fontSize=11, textColor=C_ACCENT,
                fontName="Helvetica-Bold", spaceAfter=1*mm,
                spaceBefore=2*mm),
            "Body": ParagraphStyle("Body",
                fontSize=9, textColor=C_TEXT,
                fontName="Helvetica", spaceAfter=2*mm),
            "BodyCenter": ParagraphStyle("BodyCenter",
                fontSize=9, textColor=C_TEXT,
                fontName="Helvetica", spaceAfter=2*mm,
                alignment=TA_CENTER),
            "KPILabel": ParagraphStyle("KPILabel",
                fontSize=8, textColor=C_WHITE,
                fontName="Helvetica-Bold", alignment=TA_CENTER),
            "KPIValue": ParagraphStyle("KPIValue",
                fontSize=12, textColor=C_TEXT,
                fontName="Helvetica-Bold", alignment=TA_CENTER),
            "TableCell": ParagraphStyle("TableCell",
                fontSize=8, textColor=C_TEXT,
                fontName="Helvetica", alignment=TA_CENTER),
            "LeakTitle": ParagraphStyle("LeakTitle",
                fontSize=10, textColor=C_DARK,
                fontName="Helvetica-Bold", spaceAfter=1*mm),
            "LeakMeta": ParagraphStyle("LeakMeta",
                fontSize=7, textColor=C_MUTED,
                fontName="Helvetica", spaceAfter=1*mm),
            "LeakDescription": ParagraphStyle("LeakDescription",
                fontSize=8, textColor=C_TEXT,
                fontName="Helvetica", spaceAfter=1*mm),
            "LeakAdvice": ParagraphStyle("LeakAdvice",
                fontSize=8, textColor=C_ACCENT,
                fontName="Helvetica-Bold", spaceAfter=1*mm),
            "StrengthItem": ParagraphStyle("StrengthItem",
                fontSize=9, textColor=C_GREEN,
                fontName="Helvetica", spaceAfter=1*mm),
        }

    def _base_table_style(self) -> TableStyle:
        return TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0),  C_DARK),
            ("TEXTCOLOR",    (0, 0), (-1, 0),  C_WHITE),
            ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 8),
            ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS",(0,1), (-1,-1),  [colors.white, C_LIGHT_BG]),
            ("GRID",         (0, 0), (-1, -1), 0.5, C_BORDER),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ])

    # ------------------------------------------------------------------
    # En-tête / pied de page
    # ------------------------------------------------------------------

    def _header_footer(self, canvas, doc):
        canvas.saveState()
        W, H = A4

        # Header — bande bleue
        canvas.setFillColor(C_DARK)
        canvas.rect(0, H - 1.5*cm, W, 1.5*cm, fill=True, stroke=False)
        canvas.setFillColor(C_WHITE)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(2*cm, H - 1*cm, "♠ Poker HUD Assistant")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(W - 2*cm, H - 1*cm,
                               datetime.now().strftime("%d/%m/%Y"))

        # Footer
        canvas.setFillColor(C_MUTED)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(2*cm, 0.7*cm,
                          "Rapport généré automatiquement — Usage personnel")
        canvas.drawRightString(W - 2*cm, 0.7*cm,
                               f"Page {doc.page}")
        canvas.restoreState()


# ---------------------------------------------------------------------------
# Intégration main.py
# ---------------------------------------------------------------------------

def generate_session_report_async(tracker, session_id: int) -> Optional[Path]:
    """
    Génère le rapport en arrière-plan depuis main.py.

    Usage dans _close_session() de PokerAssistant :
        from rapport_pdf import generate_session_report_async
        path = generate_session_report_async(self.tracker, self.session_id)
        if path:
            print(f"Rapport PDF : {path}")
    """
    try:
        gen  = PDFReportGenerator(tracker)
        path = gen.generate_session_report(session_id)
        return path
    except Exception as e:
        log.error(f"Génération rapport PDF échouée : {e}")
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if not _HAS_REPORTLAB:
        print("ERREUR : installez reportlab → pip install reportlab")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Générateur de rapport PDF poker")
    parser.add_argument("--session", type=int, default=None,
                        help="ID de session spécifique")
    parser.add_argument("--latest",  action="store_true",
                        help="Rapport de la dernière session")
    parser.add_argument("--global",  dest="global_report", action="store_true",
                        help="Rapport global toutes sessions")
    parser.add_argument("--demo",    action="store_true",
                        help="Créer des données de démo puis générer le rapport")
    args = parser.parse_args()

    try:
        from tracker import PokerTracker, HandRecord
        tracker = PokerTracker()
    except ImportError:
        print("ERREUR : tracker.py introuvable.")
        sys.exit(1)

    if args.demo:
        import random, time as _t
        print("Insertion de données de démo…")
        sid = tracker.start_session(buy_in=50.0, game_type="Tournoi", num_players=8)
        for i in range(40):
            rec   = random.choice(["FOLD","CALL","RAISE","CHECK","BET","ALL-IN"])
            taken = rec if random.random() > 0.35 else random.choice(["FOLD","CALL","RAISE"])
            tracker.record_hand(HandRecord(
                session_id=sid, stage_final=random.choice(["preflop","flop","turn","river"]),
                player_cards=["Ks","7h"], board_cards=["Kd","7d","2c"],
                num_opponents=random.randint(1,5),
                pot_final=random.uniform(20,200),
                hand_class=random.choice(["Paire","Deux Paires","Haute Carte","Brelan","Couleur"]),
                win_probability=random.uniform(0.25,0.90),
                recommended_action=rec, action_taken=taken,
                followed_advice=(rec==taken),
                result=random.gauss(2, 25),
                ev_estimate=random.uniform(5,40),
                ev_realized=random.gauss(2, 25),
            ))
        tracker.end_session(sid, placement=2, prize=120.0)
        args.session = sid
        print(f"Session #{sid} créée.\n")

    gen = PDFReportGenerator(tracker)

    if args.session:
        path = gen.generate_session_report(args.session)
    elif args.latest:
        path = gen.generate_latest_session()
    elif args.global_report:
        path = gen.generate_global_report()
    else:
        path = gen.generate_latest_session()

    if path:
        print(f"\nRapport PDF généré : {path}")
