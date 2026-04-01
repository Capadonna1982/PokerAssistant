"""
stats_viewer.py — Dashboard de statistiques poker
Dépendances : tkinter (inclus Python), tracker.py

Lance une fenêtre indépendante avec graphiques et tableaux de stats.
Peut être lancé séparément ou depuis main.py.

Usage :
    python stats_viewer.py
    python stats_viewer.py --session 3   (stats d'une session spécifique)
"""

import argparse
import json
import logging
import tkinter as tk
from tkinter import ttk, font as tkfont
from datetime import datetime
from pathlib import Path
from typing import Optional

from tracker     import PokerTracker, DB_PATH
try:
    from leak_finder import add_leak_tab_to_viewer
    _HAS_LEAK_FINDER = True
except ImportError:
    _HAS_LEAK_FINDER = False

try:
    from hand_replay import add_replay_tab_to_viewer
    _HAS_REPLAY = True
except ImportError:
    _HAS_REPLAY = False

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thème (cohérent avec overlay.py)
# ---------------------------------------------------------------------------

THEME = {
    "bg":           "#0D0D0D",
    "bg_panel":     "#1A1A2E",
    "bg_card":      "#16213E",
    "bg_header":    "#0F3460",
    "text":         "#E8E8E8",
    "text_muted":   "#9AA5B4",
    "text_dim":     "#5A6478",
    "accent":       "#3498DB",
    "green":        "#27AE60",
    "red":          "#E74C3C",
    "orange":       "#F39C12",
    "border":       "#2C3E50",
    "font":         "Consolas",
    "win_w":        900,
    "win_h":        680,
}


# ---------------------------------------------------------------------------
# Composants UI réutilisables
# ---------------------------------------------------------------------------

def make_label(parent, text, size=10, color=None, bold=False, **kwargs):
    f = tkfont.Font(
        family=THEME["font"],
        size=size,
        weight="bold" if bold else "normal",
    )
    return tk.Label(
        parent, text=text,
        font=f,
        bg=kwargs.pop("bg", THEME["bg_card"]),
        fg=color or THEME["text"],
        **kwargs,
    )


def make_stat_card(parent, title: str, value: str, color: str = None, width: int = 160):
    """Carte métrique : titre + grande valeur."""
    frame = tk.Frame(parent, bg=THEME["bg_card"],
                     highlightbackground=THEME["border"],
                     highlightthickness=1,
                     width=width)
    frame.pack_propagate(False)

    make_label(frame, title, size=8, color=THEME["text_dim"], bg=THEME["bg_card"]).pack(pady=(8, 2))
    make_label(frame, value, size=14, color=color or THEME["accent"], bold=True,
               bg=THEME["bg_card"]).pack(pady=(0, 8))
    return frame


def canvas_bar(canvas: tk.Canvas, x, y, w, h, pct: float, color: str, bg: str = THEME["border"]):
    """Dessine une barre de progression sur un Canvas."""
    canvas.create_rectangle(x, y, x + w, y + h, fill=bg, outline="")
    fill_w = int(w * min(max(pct, 0), 1))
    if fill_w > 0:
        canvas.create_rectangle(x, y, x + fill_w, y + h, fill=color, outline="")


# ---------------------------------------------------------------------------
# Dashboard principal
# ---------------------------------------------------------------------------

class StatsViewer:
    """
    Fenêtre tkinter affichant toutes les statistiques poker.

    Onglets :
      1. Vue d'ensemble    — KPIs globaux + graphique profit cumulé
      2. Sessions          — tableau des dernières sessions
      3. Performance       — suivi des conseils, win rate par main
      4. Saisie manuelle   — enregistrer une nouvelle session/main
    """

    def __init__(self, tracker: PokerTracker, session_id: Optional[int] = None):
        self.tracker    = tracker
        self.session_id = session_id
        self.root: Optional[tk.Tk] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.root = tk.Tk()
        self.root.title("♠ Poker Stats Dashboard")
        self.root.configure(bg=THEME["bg"])
        self.root.geometry(f"{THEME['win_w']}x{THEME['win_h']}")
        self.root.resizable(True, True)

        self._build_header()
        self._build_tabs()
        self._load_all_data()

        log.info("Stats Dashboard ouvert.")
        self.root.mainloop()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _build_header(self) -> None:
        header = tk.Frame(self.root, bg=THEME["bg_header"], height=50)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(
            header,
            text="♠ POKER STATS DASHBOARD",
            font=tkfont.Font(family=THEME["font"], size=14, weight="bold"),
            bg=THEME["bg_header"],
            fg=THEME["accent"],
        ).pack(side=tk.LEFT, padx=16, pady=10)

        tk.Button(
            header,
            text="↻ Actualiser",
            font=tkfont.Font(family=THEME["font"], size=9),
            bg=THEME["bg_panel"],
            fg=THEME["text"],
            relief=tk.FLAT,
            cursor="hand2",
            command=self._load_all_data,
        ).pack(side=tk.RIGHT, padx=16, pady=10)

    # ------------------------------------------------------------------
    # Onglets
    # ------------------------------------------------------------------

    def _build_tabs(self) -> None:
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook",
                         background=THEME["bg"],
                         borderwidth=0)
        style.configure("TNotebook.Tab",
                         background=THEME["bg_panel"],
                         foreground=THEME["text_muted"],
                         font=(THEME["font"], 9),
                         padding=[12, 6])
        style.map("TNotebook.Tab",
                  background=[("selected", THEME["bg_card"])],
                  foreground=[("selected", THEME["accent"])])

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.tab_overview  = tk.Frame(self.notebook, bg=THEME["bg"])
        self.tab_sessions  = tk.Frame(self.notebook, bg=THEME["bg"])
        self.tab_perf      = tk.Frame(self.notebook, bg=THEME["bg"])
        self.tab_entry     = tk.Frame(self.notebook, bg=THEME["bg"])

        self.notebook.add(self.tab_overview, text="  Vue d'ensemble  ")
        self.notebook.add(self.tab_sessions, text="  Sessions  ")
        self.notebook.add(self.tab_perf,     text="  Performance  ")
        self.notebook.add(self.tab_entry,    text="  Saisie  ")

        self._build_overview()
        self._build_sessions()
        self._build_performance()
        self._build_entry()

        # Onglet Leaks (si disponible)
        if _HAS_LEAK_FINDER:
            add_leak_tab_to_viewer(self, self.tracker)

        # Onglet Replay (si disponible)
        if _HAS_REPLAY:
            add_replay_tab_to_viewer(self, self.tracker)

    # ------------------------------------------------------------------
    # Onglet 1 — Vue d'ensemble
    # ------------------------------------------------------------------

    def _build_overview(self) -> None:
        p = self.tab_overview

        # KPI cards row
        self.kpi_frame = tk.Frame(p, bg=THEME["bg"])
        self.kpi_frame.pack(fill=tk.X, padx=12, pady=12)

        self.kpi_cards = {}
        kpis = [
            ("Sessions",    "—", None),
            ("Profit total","—", None),
            ("ROI %",       "—", None),
            ("Win rate",    "—", None),
            ("Mains jouées","—", None),
        ]
        for title, val, color in kpis:
            card = make_stat_card(self.kpi_frame, title, val, color, width=158)
            card.pack(side=tk.LEFT, padx=4, fill=tk.Y)
            self.kpi_cards[title] = card

        # Graphique profit cumulé
        graph_frame = tk.Frame(p, bg=THEME["bg_card"],
                               highlightbackground=THEME["border"],
                               highlightthickness=1)
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        tk.Label(
            graph_frame,
            text="Profit cumulé par session",
            font=tkfont.Font(family=THEME["font"], size=9),
            bg=THEME["bg_card"],
            fg=THEME["text_muted"],
        ).pack(anchor=tk.W, padx=10, pady=(8, 0))

        self.profit_canvas = tk.Canvas(
            graph_frame,
            bg=THEME["bg_card"],
            highlightthickness=0,
            height=280,
        )
        self.profit_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _draw_profit_chart(self, data: list[dict]) -> None:
        """Dessine le graphique ligne du profit cumulé."""
        canvas = self.profit_canvas
        canvas.delete("all")

        if not data:
            canvas.create_text(
                canvas.winfo_width() // 2 or 400,
                130,
                text="Aucune donnée — jouez votre première session !",
                fill=THEME["text_dim"],
                font=(THEME["font"], 10),
            )
            return

        canvas.update_idletasks()
        W = canvas.winfo_width()  or 820
        H = canvas.winfo_height() or 260

        pad_l, pad_r, pad_t, pad_b = 50, 20, 20, 40

        values = [d["cumulative"] for d in data]
        min_v  = min(values + [0])
        max_v  = max(values + [0])
        span   = max(max_v - min_v, 1)

        def x_pos(i):
            return pad_l + i * (W - pad_l - pad_r) / max(len(data) - 1, 1)

        def y_pos(v):
            return pad_t + (1 - (v - min_v) / span) * (H - pad_t - pad_b)

        # Ligne zéro
        y0 = y_pos(0)
        canvas.create_line(pad_l, y0, W - pad_r, y0,
                           fill=THEME["border"], dash=(4, 4), width=1)
        canvas.create_text(pad_l - 4, y0, text="0", fill=THEME["text_dim"],
                           font=(THEME["font"], 8), anchor=tk.E)

        # Axes
        canvas.create_line(pad_l, pad_t, pad_l, H - pad_b,
                           fill=THEME["border"], width=1)
        canvas.create_line(pad_l, H - pad_b, W - pad_r, H - pad_b,
                           fill=THEME["border"], width=1)

        # Lignes du graphique
        points = [(x_pos(i), y_pos(v)) for i, v in enumerate(values)]

        for i in range(len(points) - 1):
            color = THEME["green"] if values[i + 1] >= 0 else THEME["red"]
            canvas.create_line(
                points[i][0], points[i][1],
                points[i + 1][0], points[i + 1][1],
                fill=color, width=2, smooth=True,
            )

        # Points
        for i, (px, py) in enumerate(points):
            color = THEME["green"] if values[i] >= 0 else THEME["red"]
            canvas.create_oval(px - 4, py - 4, px + 4, py + 4,
                               fill=color, outline=THEME["bg_card"])

        # Labels dates (premier, milieu, dernier)
        indices = [0, len(data) // 2, len(data) - 1] if len(data) > 2 else list(range(len(data)))
        for i in indices:
            px = x_pos(i)
            canvas.create_text(
                px, H - pad_b + 12,
                text=data[i]["date"][:10],
                fill=THEME["text_dim"],
                font=(THEME["font"], 7),
            )

        # Valeur finale
        if points:
            last_x, last_y = points[-1]
            color = THEME["green"] if values[-1] >= 0 else THEME["red"]
            canvas.create_text(
                last_x, last_y - 14,
                text=f"{values[-1]:+.1f}$",
                fill=color,
                font=(THEME["font"], 9, "bold"),
            )

    # ------------------------------------------------------------------
    # Onglet 2 — Sessions
    # ------------------------------------------------------------------

    def _build_sessions(self) -> None:
        p = self.tab_sessions

        cols = ("Date", "Type", "Buy-in", "Prix", "Profit", "Place", "Mains", "Suivi %", "Durée")

        tree_frame = tk.Frame(p, bg=THEME["bg"])
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        style = ttk.Style()
        style.configure("Poker.Treeview",
                         background=THEME["bg_card"],
                         foreground=THEME["text"],
                         fieldbackground=THEME["bg_card"],
                         rowheight=28,
                         font=(THEME["font"], 9))
        style.configure("Poker.Treeview.Heading",
                         background=THEME["bg_panel"],
                         foreground=THEME["accent"],
                         font=(THEME["font"], 9, "bold"))
        style.map("Poker.Treeview",
                  background=[("selected", THEME["bg_header"])])

        self.sessions_tree = ttk.Treeview(
            tree_frame,
            columns=cols,
            show="headings",
            style="Poker.Treeview",
        )

        col_widths = [130, 80, 70, 70, 80, 55, 60, 70, 70]
        for col, w in zip(cols, col_widths):
            self.sessions_tree.heading(col, text=col)
            self.sessions_tree.column(col, width=w, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL,
                                   command=self.sessions_tree.yview)
        self.sessions_tree.configure(yscrollcommand=scrollbar.set)

        self.sessions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Tags couleur pour profit
        self.sessions_tree.tag_configure("profit",  foreground=THEME["green"])
        self.sessions_tree.tag_configure("loss",    foreground=THEME["red"])
        self.sessions_tree.tag_configure("neutral", foreground=THEME["text_muted"])

    # ------------------------------------------------------------------
    # Onglet 3 — Performance
    # ------------------------------------------------------------------

    def _build_performance(self) -> None:
        p = self.tab_perf

        # Suivi des conseils
        advice_frame = tk.LabelFrame(
            p, text="  Suivi des conseils Claude  ",
            bg=THEME["bg"], fg=THEME["accent"],
            font=(THEME["font"], 9),
            bd=1, relief=tk.GROOVE,
        )
        advice_frame.pack(fill=tk.X, padx=12, pady=(12, 6))

        self.advice_canvas = tk.Canvas(
            advice_frame, bg=THEME["bg"], height=70, highlightthickness=0
        )
        self.advice_canvas.pack(fill=tk.X, padx=12, pady=8)

        # Performance par classe de main
        hand_frame = tk.LabelFrame(
            p, text="  Profit par classe de main  ",
            bg=THEME["bg"], fg=THEME["accent"],
            font=(THEME["font"], 9),
            bd=1, relief=tk.GROOVE,
        )
        hand_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        self.hand_canvas = tk.Canvas(
            hand_frame, bg=THEME["bg"], highlightthickness=0
        )
        self.hand_canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

    def _draw_advice_chart(self, perf: dict) -> None:
        canvas = self.advice_canvas
        canvas.delete("all")

        followed = perf.get("followed", {})
        ignored  = perf.get("ignored",  {})

        f_count = followed.get("count", 0)
        i_count = ignored.get("count",  0)
        total   = f_count + i_count or 1
        f_pct   = f_count / total

        canvas.update_idletasks()
        W = canvas.winfo_width() or 820

        # Barre suivi
        canvas.create_text(12, 16, text="Conseils suivis", fill=THEME["text_muted"],
                           font=(THEME["font"], 9), anchor=tk.W)
        canvas_bar(canvas, 140, 8, W - 280, 20, f_pct, THEME["green"])
        canvas.create_text(W - 130, 18,
                           text=f"{f_count} mains | moy. {followed.get('avg_result', 0):+.2f}$",
                           fill=THEME["green"], font=(THEME["font"], 9), anchor=tk.W)

        # Barre ignoré
        canvas.create_text(12, 50, text="Conseils ignorés", fill=THEME["text_muted"],
                           font=(THEME["font"], 9), anchor=tk.W)
        canvas_bar(canvas, 140, 42, W - 280, 20, 1 - f_pct, THEME["red"])
        canvas.create_text(W - 130, 52,
                           text=f"{i_count} mains | moy. {ignored.get('avg_result', 0):+.2f}$",
                           fill=THEME["red"], font=(THEME["font"], 9), anchor=tk.W)

    def _draw_hand_chart(self, breakdown: dict) -> None:
        canvas = self.hand_canvas
        canvas.delete("all")

        if not breakdown:
            canvas.create_text(400, 80, text="Aucune donnée",
                               fill=THEME["text_dim"], font=(THEME["font"], 10))
            return

        canvas.update_idletasks()
        W = canvas.winfo_width()  or 820
        H = canvas.winfo_height() or 200

        sorted_hands = sorted(breakdown.items(), key=lambda x: x[1]["profit"], reverse=True)
        n    = len(sorted_hands)
        bar_h = min(24, (H - 20) // max(n, 1))
        pad_l = 150

        max_abs = max(abs(v["profit"]) for _, v in sorted_hands) or 1

        for i, (hand_class, data) in enumerate(sorted_hands):
            y = 10 + i * (bar_h + 4)
            profit = data["profit"]
            pct    = abs(profit) / max_abs
            color  = THEME["green"] if profit >= 0 else THEME["red"]

            canvas.create_text(pad_l - 8, y + bar_h // 2,
                               text=hand_class, fill=THEME["text_muted"],
                               font=(THEME["font"], 8), anchor=tk.E)
            canvas_bar(canvas, pad_l, y, W - pad_l - 120, bar_h, pct, color)
            canvas.create_text(W - 110, y + bar_h // 2,
                               text=f"{profit:+.1f}$ ({data['count']} mains)",
                               fill=color, font=(THEME["font"], 8), anchor=tk.W)

    # ------------------------------------------------------------------
    # Onglet 4 — Saisie manuelle
    # ------------------------------------------------------------------

    def _build_entry(self) -> None:
        p = self.tab_entry

        # ── Nouvelle session ────────────────────────────────────────────
        session_frame = tk.LabelFrame(
            p, text="  Nouvelle session  ",
            bg=THEME["bg"], fg=THEME["accent"],
            font=(THEME["font"], 9), bd=1, relief=tk.GROOVE,
        )
        session_frame.pack(fill=tk.X, padx=12, pady=(12, 6))

        fields_s = tk.Frame(session_frame, bg=THEME["bg"])
        fields_s.pack(fill=tk.X, padx=12, pady=8)

        self._sv = {}
        row_data = [
            ("Type de jeu",  "game_type",   "Tournoi"),
            ("Buy-in ($)",   "buy_in",      "50"),
            ("Nb joueurs",   "num_players", "8"),
            ("Notes",        "notes",       ""),
        ]
        for i, (label, key, default) in enumerate(row_data):
            tk.Label(fields_s, text=label, bg=THEME["bg"], fg=THEME["text_muted"],
                     font=(THEME["font"], 9), width=14, anchor=tk.W).grid(
                row=i, column=0, padx=4, pady=3, sticky=tk.W)
            var = tk.StringVar(value=default)
            self._sv[key] = var
            tk.Entry(fields_s, textvariable=var, bg=THEME["bg_card"], fg=THEME["text"],
                     font=(THEME["font"], 9), relief=tk.FLAT, width=20).grid(
                row=i, column=1, padx=4, pady=3, sticky=tk.W)

        tk.Button(
            session_frame, text="▶ Démarrer la session",
            bg=THEME["accent"], fg=THEME["bg"],
            font=(THEME["font"], 9, "bold"),
            relief=tk.FLAT, cursor="hand2",
            command=self._start_session,
        ).pack(pady=(0, 8))

        # ── Clôturer session ────────────────────────────────────────────
        end_frame = tk.LabelFrame(
            p, text="  Clôturer la session en cours  ",
            bg=THEME["bg"], fg=THEME["accent"],
            font=(THEME["font"], 9), bd=1, relief=tk.GROOVE,
        )
        end_frame.pack(fill=tk.X, padx=12, pady=6)

        fields_e = tk.Frame(end_frame, bg=THEME["bg"])
        fields_e.pack(fill=tk.X, padx=12, pady=8)

        self._ev = {}
        row_end = [
            ("Session ID",  "session_id",  ""),
            ("Placement",   "placement",   "1"),
            ("Prix gagné ($)", "prize",    "0"),
            ("Stack final ($)", "final_stack", "0"),
        ]
        for i, (label, key, default) in enumerate(row_end):
            tk.Label(fields_e, text=label, bg=THEME["bg"], fg=THEME["text_muted"],
                     font=(THEME["font"], 9), width=14, anchor=tk.W).grid(
                row=i, column=0, padx=4, pady=3, sticky=tk.W)
            var = tk.StringVar(value=default)
            self._ev[key] = var
            tk.Entry(fields_e, textvariable=var, bg=THEME["bg_card"], fg=THEME["text"],
                     font=(THEME["font"], 9), relief=tk.FLAT, width=20).grid(
                row=i, column=1, padx=4, pady=3, sticky=tk.W)

        tk.Button(
            end_frame, text="■ Clôturer la session",
            bg=THEME["orange"], fg=THEME["bg"],
            font=(THEME["font"], 9, "bold"),
            relief=tk.FLAT, cursor="hand2",
            command=self._end_session,
        ).pack(pady=(0, 8))

        # Status
        self._status_var = tk.StringVar(value="")
        tk.Label(p, textvariable=self._status_var, bg=THEME["bg"],
                 fg=THEME["green"], font=(THEME["font"], 9)).pack(pady=4)

    # ------------------------------------------------------------------
    # Actions saisie
    # ------------------------------------------------------------------

    def _start_session(self) -> None:
        try:
            sid = self.tracker.start_session(
                buy_in      = float(self._sv["buy_in"].get() or 0),
                game_type   = self._sv["game_type"].get(),
                num_players = int(self._sv["num_players"].get() or 8),
                notes       = self._sv["notes"].get(),
            )
            self._status_var.set(f"✓ Session #{sid} démarrée. Notez cet ID pour la clôturer.")
            self._ev["session_id"].set(str(sid))
            self._load_all_data()
        except Exception as e:
            self._status_var.set(f"⚠ Erreur : {e}")

    def _end_session(self) -> None:
        try:
            self.tracker.end_session(
                session_id  = int(self._ev["session_id"].get()),
                placement   = int(self._ev["placement"].get() or 0),
                prize       = float(self._ev["prize"].get() or 0),
                final_stack = float(self._ev["final_stack"].get() or 0),
            )
            self._status_var.set("✓ Session clôturée avec succès.")
            self._load_all_data()
        except Exception as e:
            self._status_var.set(f"⚠ Erreur : {e}")

    # ------------------------------------------------------------------
    # Chargement des données
    # ------------------------------------------------------------------

    def _load_all_data(self) -> None:
        """Recharge toutes les données depuis la DB et met à jour l'UI."""
        try:
            global_stats = self.tracker.get_global_stats()
            sessions     = self.tracker.get_recent_sessions(50)
            advice_perf  = self.tracker.get_advice_performance()
            profit_data  = self.tracker.get_profit_over_time(50)

            self._update_kpis(global_stats)
            self._update_sessions_table(sessions)
            self._draw_advice_chart(advice_perf)
            self._draw_hand_chart(global_stats.get("hand_class_breakdown", {}))

            self.root.after(100, lambda: self._draw_profit_chart(profit_data))

            log.debug("Données rechargées.")
        except Exception as e:
            log.error(f"Erreur chargement données : {e}")

    def _update_kpis(self, stats: dict) -> None:
        kpi_map = {
            "Sessions":     (str(stats.get("total_sessions", 0)), None),
            "Profit total": (
                f"{stats.get('total_profit', 0):+.2f}$",
                THEME["green"] if stats.get("total_profit", 0) >= 0 else THEME["red"],
            ),
            "ROI %": (
                f"{stats.get('roi_pct', 0):+.1f}%",
                THEME["green"] if stats.get("roi_pct", 0) >= 0 else THEME["red"],
            ),
            "Win rate":     (f"{stats.get('win_rate_hands', 0):.1%}", None),
            "Mains jouées": (str(stats.get("total_hands", 0)), None),
        }

        for title, (value, color) in kpi_map.items():
            card = self.kpi_cards.get(title)
            if card:
                for widget in card.winfo_children():
                    if isinstance(widget, tk.Label):
                        txt = widget.cget("text")
                        # Mettre à jour la valeur (2e label = valeur)
                        if txt == title:
                            continue
                        widget.configure(text=value, fg=color or THEME["accent"])
                        break

    def _update_sessions_table(self, sessions: list[dict]) -> None:
        tree = self.sessions_tree
        for item in tree.get_children():
            tree.delete(item)

        for s in sessions:
            profit = s["profit"]
            tag    = "profit" if profit > 0 else ("loss" if profit < 0 else "neutral")
            tree.insert("", tk.END, values=(
                s["date"],
                s["game_type"],
                f"{s['buy_in']:.0f}$",
                f"{s['prize']:.0f}$" if s["prize"] else "—",
                f"{profit:+.2f}$",
                s["placement"] if s["placement"] else "—",
                s["hand_count"],
                f"{s['follow_rate']:.0f}%",
                f"{s['duration_min']:.0f} min",
            ), tags=(tag,))


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="♠ Poker Stats Dashboard")
    parser.add_argument("--session", type=int, default=None,
                        help="Afficher les stats d'une session spécifique")
    parser.add_argument("--db", type=str, default=str(DB_PATH),
                        help="Chemin vers la base de données SQLite")
    args = parser.parse_args()

    tracker = PokerTracker(db_path=Path(args.db))

    # Insérer des données de démonstration si la DB est vide
    stats = tracker.get_global_stats()
    if stats["total_sessions"] == 0:
        _insert_demo_data(tracker)

    viewer = StatsViewer(tracker, session_id=args.session)
    viewer.run()


def _insert_demo_data(tracker: PokerTracker) -> None:
    """Insère des données de démo pour visualiser le dashboard."""
    import random
    log.info("Insertion de données de démonstration…")

    scenarios = [
        (50,  1,   200, "Tournoi"), (50,  3,   0,   "Tournoi"),
        (50,  2,   100, "Tournoi"), (50,  5,   0,   "Tournoi"),
        (100, 1,   500, "Tournoi"), (50,  4,   0,   "Tournoi"),
        (50,  2,   80,  "Tournoi"), (100, 1,   350, "Tournoi"),
    ]
    hand_classes = ["Paire", "Deux Paires", "Brelan", "Quinte",
                    "Couleur", "Full House", "Haute Carte"]
    actions = ["FOLD", "CALL", "RAISE", "CHECK", "BET"]

    for buy_in, placement, prize, gtype in scenarios:
        sid = tracker.start_session(buy_in=buy_in, game_type=gtype)
        num_hands = random.randint(8, 20)
        for _ in range(num_hands):
            hc    = random.choice(hand_classes)
            rec   = random.choice(actions)
            taken = rec if random.random() > 0.25 else random.choice(actions)
            res   = random.uniform(-30, 80) if random.random() > 0.4 else random.uniform(-50, -5)
            from tracker import HandRecord
            tracker.record_hand(HandRecord(
                session_id=sid, stage_final=random.choice(["flop","turn","river"]),
                player_cards=["As","Kh"], board_cards=["Ah","3d","9c"],
                num_opponents=random.randint(1,5), pot_final=random.uniform(20,300),
                hand_class=hc, win_probability=random.uniform(0.3,0.9),
                recommended_action=rec, action_taken=taken,
                followed_advice=(rec==taken), result=res,
                ev_estimate=random.uniform(5,60), ev_realized=res,
            ))
        tracker.end_session(sid, placement=placement, prize=prize)

    log.info("Données de démonstration insérées.")


if __name__ == "__main__":
    main()
