"""
overlay.py — HUD flottant temps réel pour l'assistant poker
Dépendances : tkinter (inclus dans Python standard)
              Optionnel : Pillow (pip install Pillow) pour les icônes

Crée une fenêtre transparente, toujours au premier plan, sans bordure,
déplaçable par clic-glisser, qui affiche les conseils de Claude en temps réel.
"""

import json
import logging
import queue
import threading
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import font as tkfont
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thème visuel
# ---------------------------------------------------------------------------

THEME = {
    # Fond principal (noir semi-transparent simulé)
    "bg":            "#0D0D0D",
    "bg_header":     "#1A1A2E",
    "bg_card":       "#16213E",
    "bg_card_hover": "#1F2D4A",

    # Textes
    "text_primary":  "#E8E8E8",
    "text_secondary":"#9AA5B4",
    "text_muted":    "#5A6478",

    # Actions — couleurs sémantiques
    "action_fold":   "#E74C3C",   # rouge
    "action_check":  "#95A5A6",   # gris
    "action_call":   "#27AE60",   # vert
    "action_bet":    "#F39C12",   # orange
    "action_raise":  "#E67E22",   # orange vif
    "action_allin":  "#E91E63",   # rose électrique

    # Equity bar
    "equity_low":    "#E74C3C",
    "equity_mid":    "#F39C12",
    "equity_high":   "#27AE60",

    # Bordures
    "border":        "#2C3E50",
    "border_accent": "#3498DB",

    # Dimensions
    "width":         280,
    "font_family":   "Consolas",   # monospace lisible
    "corner_radius": 8,
}

ACTION_COLORS = {
    "FOLD":   THEME["action_fold"],
    "CHECK":  THEME["action_check"],
    "CALL":   THEME["action_call"],
    "BET":    THEME["action_bet"],
    "RAISE":  THEME["action_raise"],
    "ALL-IN": THEME["action_allin"],
}

ACTION_ICONS = {
    "FOLD":   "✗",
    "CHECK":  "—",
    "CALL":   "✓",
    "BET":    "↑",
    "RAISE":  "↑↑",
    "ALL-IN": "⚡",
}

# ---------------------------------------------------------------------------
# Données à afficher
# ---------------------------------------------------------------------------

@dataclass
class DisplayData:
    """Snapshot à afficher dans le HUD."""
    action:           str   = "—"
    action_color:     str   = THEME["text_secondary"]
    sizing:           str   = ""
    win_probability:  float = 0.0
    hand_class:       str   = ""
    stage:            str   = ""
    player_cards:     list  = field(default_factory=list)
    board_cards:      list  = field(default_factory=list)
    num_opponents:    int   = 0
    hands_beating:    int   = 0
    ev_estimate:      float = 0.0
    summary:          str   = ""
    explanation:      str   = ""
    latency_ms:       float = 0.0
    is_loading:       bool  = False
    error:            str   = ""

    @classmethod
    def from_advice(cls, advice, game_state: Optional[dict] = None) -> "DisplayData":
        """Construit un DisplayData depuis un PokerAdvice (claude_client.py)."""
        action = advice.recommended_action or "—"
        d = cls(
            action          = action,
            action_color    = ACTION_COLORS.get(action, THEME["text_primary"]),
            sizing          = advice.recommended_sizing,
            win_probability = advice.win_probability,
            hand_class      = advice.hand_class,
            hands_beating   = advice.hands_beating_us,
            summary         = advice.summary,
            explanation     = advice.action_explanation,
            latency_ms      = advice.latency_ms,
            error           = advice.error,
        )
        if game_state:
            d.stage         = game_state.get("stage", "")
            d.player_cards  = game_state.get("player_cards", [])
            d.board_cards   = game_state.get("board_cards", [])
            d.num_opponents = game_state.get("num_players", 0)
        return d

    @classmethod
    def loading(cls) -> "DisplayData":
        d = cls(is_loading=True)
        d.summary = "Analyse en cours…"
        return d


# ---------------------------------------------------------------------------
# Widget Canvas arrondi (helper)
# ---------------------------------------------------------------------------

def _rounded_rect(canvas: tk.Canvas, x1, y1, x2, y2, r=8, **kwargs):
    """Dessine un rectangle arrondi sur un Canvas tkinter."""
    canvas.create_arc(x1,     y1,     x1+2*r, y1+2*r, start=90,  extent=90,  style=tk.PIESLICE, **kwargs)
    canvas.create_arc(x2-2*r, y1,     x2,     y1+2*r, start=0,   extent=90,  style=tk.PIESLICE, **kwargs)
    canvas.create_arc(x1,     y2-2*r, x1+2*r, y2,     start=180, extent=90,  style=tk.PIESLICE, **kwargs)
    canvas.create_arc(x2-2*r, y2-2*r, x2,     y2,     start=270, extent=90,  style=tk.PIESLICE, **kwargs)
    canvas.create_rectangle(x1+r, y1, x2-r, y2, **kwargs)
    canvas.create_rectangle(x1, y1+r, x2, y2-r, **kwargs)


# ---------------------------------------------------------------------------
# HUD Principal
# ---------------------------------------------------------------------------

class PokerHUD:
    """
    Fenêtre overlay transparente et flottante affichant les conseils poker.

    Usage :
        hud = PokerHUD()
        hud.update(DisplayData.from_advice(advice, state.to_dict()))
        hud.run()        # bloquant — à appeler dans le thread principal

    Depuis un autre thread :
        hud.update_async(display_data)
    """

    def __init__(
        self,
        x: int = 40,
        y: int = 200,
        opacity: float = 0.92,
        always_on_top: bool = True,
    ):
        self._x       = x
        self._y       = y
        self._opacity = opacity
        self._always_on_top = always_on_top
        self._queue: queue.Queue = queue.Queue()
        self._data  = DisplayData()

        self.root: Optional[tk.Tk] = None
        self._widgets: dict = {}
        self._drag_x = 0
        self._drag_y = 0

    # ------------------------------------------------------------------
    # Thread-safe update
    # ------------------------------------------------------------------

    def update_async(self, data: DisplayData) -> None:
        """Envoie une mise à jour depuis n'importe quel thread."""
        self._queue.put(data)

    def update(self, data: DisplayData) -> None:
        """Met à jour depuis le thread tkinter (sinon utiliser update_async)."""
        self._data = data
        if self.root:
            self._refresh_ui()

    # ------------------------------------------------------------------
    # Construction de la fenêtre
    # ------------------------------------------------------------------

    def _build(self) -> None:
        self.root = tk.Tk()
        self.root.title("Poker HUD")
        self.root.overrideredirect(True)          # supprime la barre de titre
        self.root.attributes("-topmost", self._always_on_top)
        self.root.attributes("-alpha",   self._opacity)
        self.root.configure(bg=THEME["bg"])
        self.root.geometry(f"{THEME['width']}x20+{self._x}+{self._y}")

        # Polices
        ff = THEME["font_family"]
        self._f_title    = tkfont.Font(family=ff, size=11, weight="bold")
        self._f_action   = tkfont.Font(family=ff, size=20, weight="bold")
        self._f_label    = tkfont.Font(family=ff, size=9)
        self._f_value    = tkfont.Font(family=ff, size=10, weight="bold")
        self._f_small    = tkfont.Font(family=ff, size=8)
        self._f_summary  = tkfont.Font(family=ff, size=9)

        self._build_ui()
        self._bind_drag()
        self._bind_shortcuts()
        self._schedule_queue_poll()

    def _build_ui(self) -> None:
        w = THEME["width"]
        root = self.root

        # ── Header ──────────────────────────────────────────────────────
        header = tk.Frame(root, bg=THEME["bg_header"], height=28)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(
            header, text="♠ POKER HUD",
            font=self._f_title,
            bg=THEME["bg_header"],
            fg=THEME["border_accent"],
        ).pack(side=tk.LEFT, padx=8, pady=4)

        # Bouton fermer
        btn_close = tk.Label(
            header, text="✕",
            font=self._f_label,
            bg=THEME["bg_header"],
            fg=THEME["text_muted"],
            cursor="hand2",
        )
        btn_close.pack(side=tk.RIGHT, padx=8, pady=4)
        btn_close.bind("<Button-1>", lambda e: self.root.destroy())

        # Bouton minimize
        btn_min = tk.Label(
            header, text="—",
            font=self._f_label,
            bg=THEME["bg_header"],
            fg=THEME["text_muted"],
            cursor="hand2",
        )
        btn_min.pack(side=tk.RIGHT, padx=2, pady=4)
        btn_min.bind("<Button-1>", self._toggle_minimize)

        self._widgets["header"] = header
        self._minimized = False
        self._content_frame = None

        # ── Content ─────────────────────────────────────────────────────
        self._content_frame = tk.Frame(root, bg=THEME["bg"])
        self._content_frame.pack(fill=tk.BOTH, padx=6, pady=(0, 6))

        self._build_content(self._content_frame)

    def _build_content(self, parent: tk.Frame) -> None:

        # ── Cartes ──────────────────────────────────────────────────────
        cards_frame = tk.Frame(parent, bg=THEME["bg_card"], pady=4)
        cards_frame.pack(fill=tk.X, pady=(4, 2))

        tk.Label(
            cards_frame, text="MAIN",
            font=self._f_label,
            bg=THEME["bg_card"],
            fg=THEME["text_muted"],
        ).pack(side=tk.LEFT, padx=8)

        self._widgets["cards_var"] = tk.StringVar(value="— —")
        self._widgets["cards_lbl"] = tk.Label(
            cards_frame,
            textvariable=self._widgets["cards_var"],
            font=self._f_value,
            bg=THEME["bg_card"],
            fg=THEME["text_primary"],
        )
        self._widgets["cards_lbl"].pack(side=tk.LEFT, padx=4)

        self._widgets["stage_var"] = tk.StringVar(value="")
        tk.Label(
            cards_frame,
            textvariable=self._widgets["stage_var"],
            font=self._f_small,
            bg=THEME["bg_card"],
            fg=THEME["text_muted"],
        ).pack(side=tk.RIGHT, padx=8)

        # ── Board ────────────────────────────────────────────────────────
        board_frame = tk.Frame(parent, bg=THEME["bg_card"], pady=3)
        board_frame.pack(fill=tk.X, pady=1)

        tk.Label(
            board_frame, text="BOARD",
            font=self._f_label,
            bg=THEME["bg_card"],
            fg=THEME["text_muted"],
        ).pack(side=tk.LEFT, padx=8)

        self._widgets["board_var"] = tk.StringVar(value="—")
        tk.Label(
            board_frame,
            textvariable=self._widgets["board_var"],
            font=self._f_value,
            bg=THEME["bg_card"],
            fg=THEME["text_primary"],
        ).pack(side=tk.LEFT, padx=4)

        # ── Séparateur ───────────────────────────────────────────────────
        sep = tk.Frame(parent, bg=THEME["border"], height=1)
        sep.pack(fill=tk.X, pady=4)

        # ── Action principale ────────────────────────────────────────────
        action_frame = tk.Frame(parent, bg=THEME["bg"])
        action_frame.pack(fill=tk.X)

        self._widgets["action_var"]   = tk.StringVar(value="—")
        self._widgets["action_color"] = THEME["text_primary"]

        self._widgets["action_lbl"] = tk.Label(
            action_frame,
            textvariable=self._widgets["action_var"],
            font=self._f_action,
            bg=THEME["bg"],
            fg=THEME["text_primary"],
        )
        self._widgets["action_lbl"].pack(pady=(2, 0))

        self._widgets["sizing_var"] = tk.StringVar(value="")
        tk.Label(
            action_frame,
            textvariable=self._widgets["sizing_var"],
            font=self._f_small,
            bg=THEME["bg"],
            fg=THEME["text_secondary"],
            wraplength=THEME["width"] - 20,
        ).pack()

        # ── Barre d'équité ───────────────────────────────────────────────
        equity_frame = tk.Frame(parent, bg=THEME["bg"])
        equity_frame.pack(fill=tk.X, padx=8, pady=(4, 2))

        eq_header = tk.Frame(equity_frame, bg=THEME["bg"])
        eq_header.pack(fill=tk.X)

        tk.Label(
            eq_header, text="ÉQUITÉ",
            font=self._f_label,
            bg=THEME["bg"],
            fg=THEME["text_muted"],
        ).pack(side=tk.LEFT)

        self._widgets["equity_pct_var"] = tk.StringVar(value="0%")
        tk.Label(
            eq_header,
            textvariable=self._widgets["equity_pct_var"],
            font=self._f_value,
            bg=THEME["bg"],
            fg=THEME["text_primary"],
        ).pack(side=tk.RIGHT)

        # Canvas barre
        bar_h = 10
        bar_w = THEME["width"] - 32
        self._widgets["equity_canvas"] = tk.Canvas(
            equity_frame,
            width=bar_w, height=bar_h,
            bg=THEME["border"], bd=0, highlightthickness=0,
        )
        self._widgets["equity_canvas"].pack(fill=tk.X, pady=2)
        self._widgets["equity_bar_w"] = bar_w
        # Fond de la barre
        self._widgets["equity_canvas"].create_rectangle(
            0, 0, bar_w, bar_h, fill=THEME["border"], outline=""
        )
        # Barre de remplissage (mise à jour dynamiquement)
        self._widgets["equity_fill"] = self._widgets["equity_canvas"].create_rectangle(
            0, 0, 0, bar_h, fill=THEME["equity_low"], outline=""
        )

        # ── Stats ────────────────────────────────────────────────────────
        stats_frame = tk.Frame(parent, bg=THEME["bg_card"], pady=4)
        stats_frame.pack(fill=tk.X, pady=2)

        self._widgets["hand_var"] = tk.StringVar(value="—")
        self._add_stat_row(stats_frame, "MAIN    :", "hand_var",  row=0)

        self._widgets["combos_var"] = tk.StringVar(value="—")
        self._add_stat_row(stats_frame, "BATTUS  :", "combos_var", row=1)

        self._widgets["ev_var"] = tk.StringVar(value="—")
        self._add_stat_row(stats_frame, "EV      :", "ev_var",    row=2)

        self._widgets["opp_var"] = tk.StringVar(value="—")
        self._add_stat_row(stats_frame, "ADVERS. :", "opp_var",  row=3)

        # ── Résumé Claude ────────────────────────────────────────────────
        sep2 = tk.Frame(parent, bg=THEME["border"], height=1)
        sep2.pack(fill=tk.X, pady=3)

        self._widgets["summary_var"] = tk.StringVar(value="En attente…")
        summary_lbl = tk.Label(
            parent,
            textvariable=self._widgets["summary_var"],
            font=self._f_summary,
            bg=THEME["bg"],
            fg=THEME["text_secondary"],
            wraplength=THEME["width"] - 20,
            justify=tk.LEFT,
        )
        summary_lbl.pack(fill=tk.X, padx=8, pady=(0, 4))

        # ── Latence ──────────────────────────────────────────────────────
        self._widgets["latency_var"] = tk.StringVar(value="")
        tk.Label(
            parent,
            textvariable=self._widgets["latency_var"],
            font=self._f_small,
            bg=THEME["bg"],
            fg=THEME["text_muted"],
        ).pack(anchor=tk.E, padx=8)

    def _add_stat_row(self, parent, label: str, widget_key: str, row: int) -> None:
        frame = tk.Frame(parent, bg=THEME["bg_card"])
        frame.pack(fill=tk.X, padx=8)
        tk.Label(
            frame, text=label,
            font=self._f_label,
            bg=THEME["bg_card"],
            fg=THEME["text_muted"],
            width=8, anchor=tk.W,
        ).pack(side=tk.LEFT)
        tk.Label(
            frame,
            textvariable=self._widgets[widget_key],
            font=self._f_label,
            bg=THEME["bg_card"],
            fg=THEME["text_primary"],
        ).pack(side=tk.LEFT, padx=4)

    # ------------------------------------------------------------------
    # Rafraîchissement des données
    # ------------------------------------------------------------------

    def _refresh_ui(self) -> None:
        d = self._data

        # Cartes
        cards_str = " ".join(d.player_cards) if d.player_cards else "— —"
        self._widgets["cards_var"].set(cards_str)

        board_str = " ".join(d.board_cards) if d.board_cards else "—"
        self._widgets["board_var"].set(board_str)

        stage_label = d.stage.upper() if d.stage else ""
        self._widgets["stage_var"].set(stage_label)

        # Action
        if d.is_loading:
            action_text  = "⏳"
            action_color = THEME["text_muted"]
        elif d.error:
            action_text  = "ERR"
            action_color = THEME["action_fold"]
        else:
            icon         = ACTION_ICONS.get(d.action, "")
            action_text  = f"{icon} {d.action}" if icon else d.action
            action_color = ACTION_COLORS.get(d.action, THEME["text_primary"])

        self._widgets["action_var"].set(action_text)
        self._widgets["action_lbl"].configure(fg=action_color)
        self._widgets["sizing_var"].set(d.sizing)

        # Barre d'équité
        pct = d.win_probability
        self._widgets["equity_pct_var"].set(f"{pct:.0%}")
        bar_w   = self._widgets["equity_bar_w"]
        fill_w  = int(bar_w * pct)
        color   = (
            THEME["equity_high"] if pct >= 0.6 else
            THEME["equity_mid"]  if pct >= 0.35 else
            THEME["equity_low"]
        )
        canvas = self._widgets["equity_canvas"]
        canvas.coords(self._widgets["equity_fill"], 0, 0, fill_w, 10)
        canvas.itemconfig(self._widgets["equity_fill"], fill=color)

        # Stats
        self._widgets["hand_var"].set(d.hand_class or "—")
        self._widgets["combos_var"].set(
            f"{d.hands_beating:,} combos" if d.hands_beating else "—"
        )
        ev_str = (
            f"+{d.ev_estimate:.1f}$" if d.ev_estimate > 0
            else f"{d.ev_estimate:.1f}$" if d.ev_estimate != 0
            else "—"
        )
        self._widgets["ev_var"].set(ev_str)
        self._widgets["opp_var"].set(
            f"{d.num_opponents}" if d.num_opponents else "—"
        )

        # Résumé
        summary = d.error if d.error else (d.summary or d.explanation or "—")
        self._widgets["summary_var"].set(summary[:160])

        # Latence
        if d.latency_ms > 0:
            self._widgets["latency_var"].set(f"{d.latency_ms:.0f}ms")

    # ------------------------------------------------------------------
    # Drag & drop
    # ------------------------------------------------------------------

    def _bind_drag(self) -> None:
        header = self._widgets["header"]
        header.bind("<ButtonPress-1>",   self._on_drag_start)
        header.bind("<B1-Motion>",       self._on_drag_motion)
        for child in header.winfo_children():
            child.bind("<ButtonPress-1>", self._on_drag_start)
            child.bind("<B1-Motion>",     self._on_drag_motion)

    def _on_drag_start(self, event) -> None:
        self._drag_x = event.x_root - self.root.winfo_x()
        self._drag_y = event.y_root - self.root.winfo_y()

    def _on_drag_motion(self, event) -> None:
        x = event.x_root - self._drag_x
        y = event.y_root - self._drag_y
        self.root.geometry(f"+{x}+{y}")

    # ------------------------------------------------------------------
    # Minimize
    # ------------------------------------------------------------------

    def _toggle_minimize(self, event=None) -> None:
        if self._minimized:
            self._content_frame.pack(fill=tk.BOTH, padx=6, pady=(0, 6))
            self._minimized = False
        else:
            self._content_frame.pack_forget()
            self._minimized = True

    # ------------------------------------------------------------------
    # Raccourcis clavier
    # ------------------------------------------------------------------

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Escape>",     lambda e: self.root.destroy())
        self.root.bind("<Control-h>",  self._toggle_minimize)
        self.root.bind("<Control-o>",  lambda e: self._cycle_opacity())

    def _cycle_opacity(self) -> None:
        levels = [0.95, 0.80, 0.60, 0.40]
        try:
            idx = levels.index(self._opacity)
            self._opacity = levels[(idx + 1) % len(levels)]
        except ValueError:
            self._opacity = 0.92
        self.root.attributes("-alpha", self._opacity)

    # ------------------------------------------------------------------
    # Polling de la queue (thread-safe)
    # ------------------------------------------------------------------

    def _schedule_queue_poll(self) -> None:
        self._poll_queue()

    def _poll_queue(self) -> None:
        try:
            while True:
                data = self._queue.get_nowait()
                self._data = data
                self._refresh_ui()
        except queue.Empty:
            pass
        if self.root:
            self.root.after(100, self._poll_queue)   # poll toutes les 100ms

    # ------------------------------------------------------------------
    # Démarrage
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Lance la boucle tkinter (bloquant — appeler dans le thread principal)."""
        self._build()
        self._refresh_ui()
        log.info("HUD démarré. Raccourcis : Esc=quitter, Ctrl+H=minimize, Ctrl+O=opacité")
        self.root.mainloop()

    def stop(self) -> None:
        """Ferme le HUD proprement depuis un autre thread."""
        if self.root:
            self.root.after(0, self.root.destroy)


# ---------------------------------------------------------------------------
# Exécution dans un thread séparé (usage recommandé)
# ---------------------------------------------------------------------------

class HUDThread(threading.Thread):
    """
    Lance le HUD dans un thread dédié pour ne pas bloquer la boucle principale.

    Usage :
        hud_thread = HUDThread()
        hud_thread.start()

        # Depuis n'importe quel thread :
        hud_thread.hud.update_async(DisplayData.from_advice(advice, state))

        # Arrêt propre :
        hud_thread.hud.stop()
    """

    def __init__(self, x: int = 40, y: int = 200, **kwargs):
        super().__init__(daemon=True)
        self.hud = PokerHUD(x=x, y=y, **kwargs)
        self._ready = threading.Event()

    def run(self) -> None:
        self._ready.set()
        self.hud.run()

    def wait_ready(self, timeout: float = 5.0) -> bool:
        return self._ready.wait(timeout)


# ---------------------------------------------------------------------------
# Test standalone avec données simulées
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    import random

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    hud_thread = HUDThread(x=60, y=150)
    hud_thread.start()
    hud_thread.wait_ready()

    hud = hud_thread.hud

    scenarios = [
        DisplayData(
            action="FOLD", action_color=ACTION_COLORS["FOLD"],
            win_probability=0.22, hand_class="Haute Carte",
            stage="preflop", player_cards=["2h", "7d"],
            num_opponents=6, hands_beating=180,
            summary="Main trop faible en early position. Fold standard.",
            latency_ms=320,
        ),
        DisplayData(is_loading=True, summary="Analyse en cours…"),
        DisplayData(
            action="CALL", action_color=ACTION_COLORS["CALL"],
            win_probability=0.51, hand_class="Paire",
            sizing="80$",
            stage="flop", player_cards=["As", "Kh"], board_cards=["Ah", "7d", "2c"],
            num_opponents=3, hands_beating=42, ev_estimate=12.4,
            summary="Paire d'As en tête. Call pour contrôler le pot.",
            latency_ms=410,
        ),
        DisplayData(is_loading=True, summary="Analyse en cours…"),
        DisplayData(
            action="RAISE", action_color=ACTION_COLORS["RAISE"],
            win_probability=0.87, hand_class="Deux Paires",
            sizing="420$ (2.5× pot)",
            stage="flop", player_cards=["Ks", "7h"], board_cards=["Kd", "7d", "2c"],
            num_opponents=2, hands_beating=18, ev_estimate=68.4,
            summary="Top two pair. Raise pour valeur contre la calling station.",
            explanation="Adversaire VPIP 55% — mise max pour extraire de la valeur.",
            latency_ms=380,
        ),
        DisplayData(
            action="ALL-IN", action_color=ACTION_COLORS["ALL-IN"],
            win_probability=0.94, hand_class="Full House",
            sizing="1200$ (tapis)",
            stage="river", player_cards=["Ks", "7h"], board_cards=["Kd", "7d", "2c", "Kc", "7c"],
            num_opponents=1, hands_beating=2, ev_estimate=940.0,
            summary="Full House K over 7. Shove le tapis — quasi imbattable.",
            latency_ms=295,
        ),
    ]

    print("Démo HUD en cours (fermer la fenêtre ou Ctrl+C pour arrêter)...")

    for i, scenario in enumerate(scenarios):
        time.sleep(2.5)
        hud.update_async(scenario)
        print(f"Scénario {i+1}/{len(scenarios)} : {scenario.action} | {scenario.win_probability:.0%}")

    # Garder le HUD ouvert
    try:
        hud_thread.join()
    except KeyboardInterrupt:
        hud.stop()
        print("HUD fermé.")
