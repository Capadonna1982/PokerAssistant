"""
hand_replay.py — Replay interactif de mains poker post-session
Dépendances : tkinter (inclus Python), tracker.py, engine.py

Permet de rejouer chaque main enregistrée étape par étape :
  - Affichage progressif des cartes (préflop → flop → turn → river)
  - Recalcul de l'équité à chaque street via engine.py
  - Comparaison conseil recommandé vs action prise
  - Navigation entre les mains (précédent / suivant)
  - Filtrage par session, classe de main, résultat
  - Export de la main analysée en texte

Usage :
    viewer = HandReplayViewer(tracker)
    viewer.run()

    # Intégration dans stats_viewer.py
    from hand_replay import add_replay_tab_to_viewer
    add_replay_tab_to_viewer(viewer, tracker)
"""

import json
import logging
import tkinter as tk
from dataclasses import dataclass, field
from datetime import datetime
from tkinter import font as tkfont, ttk
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thème (cohérent avec overlay.py)
# ---------------------------------------------------------------------------

THEME = {
    "bg":          "#0D0D0D",
    "bg_panel":    "#1A1A2E",
    "bg_card":     "#16213E",
    "text":        "#E8E8E8",
    "text_muted":  "#9AA5B4",
    "text_dim":    "#5A6478",
    "accent":      "#3498DB",
    "green":       "#27AE60",
    "red":         "#E74C3C",
    "orange":      "#F39C12",
    "border":      "#2C3E50",
    "font":        "Consolas",
}

ACTION_COLORS = {
    "FOLD":   "#E74C3C",
    "CHECK":  "#95A5A6",
    "CALL":   "#27AE60",
    "BET":    "#F39C12",
    "RAISE":  "#E67E22",
    "ALL-IN": "#E91E63",
}

SUIT_SYMBOLS = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
SUIT_COLORS  = {"s": "#E8E8E8", "h": "#E74C3C", "d": "#E74C3C", "c": "#E8E8E8"}

# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------

@dataclass
class ReplayHand:
    """Main chargée depuis la base pour le replay."""
    hand_id:           int
    session_id:        int
    timestamp:         float
    stage_final:       str
    player_cards:      list[str]
    board_cards:       list[str]
    hand_class:        str
    win_probability:   float
    recommended_action:str
    action_taken:      str
    followed_advice:   bool
    result:            float
    ev_estimate:       float
    pot_final:         float
    num_opponents:     int
    notes:             str

    @property
    def date_str(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%d/%m %H:%M")

    @property
    def profit_color(self) -> str:
        return THEME["green"] if self.result > 0 else THEME["red"] if self.result < 0 else THEME["text_muted"]

    @property
    def cards_at_stage(self) -> dict:
        """Retourne les cartes disponibles par street."""
        board = self.board_cards
        return {
            "preflop": [],
            "flop":    board[:3]  if len(board) >= 3 else board,
            "turn":    board[:4]  if len(board) >= 4 else board,
            "river":   board[:5]  if len(board) >= 5 else board,
        }


# ---------------------------------------------------------------------------
# Chargement depuis la BD
# ---------------------------------------------------------------------------

def load_hands_from_db(tracker, session_id: Optional[int] = None,
                       limit: int = 100) -> list[ReplayHand]:
    """Charge les mains depuis tracker.py pour le replay."""
    with tracker._conn() as conn:
        if session_id:
            rows = conn.execute(
                "SELECT * FROM hands WHERE session_id=? ORDER BY timestamp DESC LIMIT ?",
                (session_id, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM hands ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()

    hands = []
    for r in rows:
        try:
            hands.append(ReplayHand(
                hand_id           = r["id"],
                session_id        = r["session_id"],
                timestamp         = r["timestamp"],
                stage_final       = r["stage_final"] or "preflop",
                player_cards      = json.loads(r["player_cards"] or "[]"),
                board_cards       = json.loads(r["board_cards"]  or "[]"),
                hand_class        = r["hand_class"]         or "",
                win_probability   = r["win_probability"]    or 0.0,
                recommended_action= r["recommended_action"] or "",
                action_taken      = r["action_taken"]       or "",
                followed_advice   = bool(r["followed_advice"]),
                result            = r["result"]             or 0.0,
                ev_estimate       = r["ev_estimate"]        or 0.0,
                pot_final         = r["pot_final"]          or 0.0,
                num_opponents     = r["num_opponents"]       or 1,
                notes             = r["notes"]              or "",
            ))
        except Exception as e:
            log.debug(f"Erreur chargement main #{r['id']} : {e}")
    return hands


# ---------------------------------------------------------------------------
# Viewer principal
# ---------------------------------------------------------------------------

class HandReplayViewer:
    """
    Fenêtre interactive de replay de mains.

    Usage standalone :
        tracker = PokerTracker()
        viewer  = HandReplayViewer(tracker)
        viewer.run()
    """

    STAGES = ["preflop", "flop", "turn", "river"]

    def __init__(self, tracker, session_id: Optional[int] = None):
        self.tracker    = tracker
        self.session_id = session_id
        self.hands:     list[ReplayHand] = []
        self.hand_idx   = 0      # index dans self.hands
        self.stage_idx  = 0      # index dans STAGES
        self.root: Optional[tk.Tk] = None
        self._widgets = {}
        self._equity_cache: dict = {}

    # ------------------------------------------------------------------
    # Démarrage
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.root = tk.Tk()
        self.root.title("♠ Replay de Mains")
        self.root.configure(bg=THEME["bg"])
        self.root.geometry("780x580")
        self.root.resizable(True, True)

        self._build_ui()
        self._load_hands()
        self._show_hand()
        self.root.mainloop()

    def _build_ui(self) -> None:
        ff = THEME["font"]
        self._f_title  = tkfont.Font(family=ff, size=13, weight="bold")
        self._f_card   = tkfont.Font(family=ff, size=22, weight="bold")
        self._f_action = tkfont.Font(family=ff, size=14, weight="bold")
        self._f_label  = tkfont.Font(family=ff, size=9)
        self._f_small  = tkfont.Font(family=ff, size=8)
        self._f_value  = tkfont.Font(family=ff, size=10, weight="bold")

        # ── Header ─────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=THEME["bg_panel"], height=40)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(header, text="♠ REPLAY DE MAINS",
                 font=self._f_title, bg=THEME["bg_panel"],
                 fg=THEME["accent"]).pack(side=tk.LEFT, padx=12, pady=6)

        # Filtre session
        self._widgets["session_var"] = tk.StringVar(value="Toutes")
        tk.Label(header, text="Session :", font=self._f_label,
                 bg=THEME["bg_panel"], fg=THEME["text_muted"]).pack(side=tk.LEFT, padx=(20,2))
        self._widgets["session_combo"] = ttk.Combobox(
            header, textvariable=self._widgets["session_var"],
            width=10, font=self._f_small, state="readonly"
        )
        self._widgets["session_combo"].pack(side=tk.LEFT, padx=2)
        self._widgets["session_combo"].bind("<<ComboboxSelected>>", self._on_filter_change)

        # Filtre résultat
        self._widgets["filter_var"] = tk.StringVar(value="Toutes")
        tk.Label(header, text="Filtre :", font=self._f_label,
                 bg=THEME["bg_panel"], fg=THEME["text_muted"]).pack(side=tk.LEFT, padx=(12,2))
        filter_combo = ttk.Combobox(
            header, textvariable=self._widgets["filter_var"],
            values=["Toutes", "Gains", "Pertes", "Conseil ignoré", "ALL-IN", "RAISE"],
            width=14, font=self._f_small, state="readonly"
        )
        filter_combo.pack(side=tk.LEFT, padx=2)
        filter_combo.bind("<<ComboboxSelected>>", self._on_filter_change)

        # Compteur
        self._widgets["counter_var"] = tk.StringVar(value="0 / 0")
        tk.Label(header, textvariable=self._widgets["counter_var"],
                 font=self._f_label, bg=THEME["bg_panel"],
                 fg=THEME["text_muted"]).pack(side=tk.RIGHT, padx=12)

        # ── Corps principal ─────────────────────────────────────────────
        body = tk.Frame(self.root, bg=THEME["bg"])
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Colonne gauche : liste des mains
        left = tk.Frame(body, bg=THEME["bg_card"], width=200)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left.pack_propagate(False)

        tk.Label(left, text="Mains",
                 font=self._f_label, bg=THEME["bg_card"],
                 fg=THEME["text_muted"]).pack(anchor=tk.W, padx=6, pady=(6,2))

        list_frame = tk.Frame(left, bg=THEME["bg_card"])
        list_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self._widgets["hand_list"] = tk.Listbox(
            list_frame,
            font=self._f_small,
            bg=THEME["bg_card"],
            fg=THEME["text"],
            selectbackground=THEME["accent"],
            selectforeground=THEME["text"],
            relief=tk.FLAT,
            borderwidth=0,
            activestyle="none",
            yscrollcommand=scrollbar.set,
        )
        scrollbar.config(command=self._widgets["hand_list"].yview)
        self._widgets["hand_list"].pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._widgets["hand_list"].bind("<<ListboxSelect>>", self._on_hand_select)

        # Colonne droite : replay
        right = tk.Frame(body, bg=THEME["bg"])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_replay_panel(right)

        # ── Contrôles navigation ────────────────────────────────────────
        nav = tk.Frame(self.root, bg=THEME["bg_panel"], height=44)
        nav.pack(fill=tk.X, side=tk.BOTTOM)
        nav.pack_propagate(False)

        btn_style = {"font": self._f_label, "bg": THEME["bg_card"],
                     "fg": THEME["text"], "relief": tk.FLAT,
                     "cursor": "hand2", "padx": 10}

        tk.Button(nav, text="◀◀ Préc.",
                  command=self._prev_hand, **btn_style).pack(side=tk.LEFT, padx=4, pady=6)
        tk.Button(nav, text="◀ Street",
                  command=self._prev_stage, **btn_style).pack(side=tk.LEFT, padx=2, pady=6)

        # Indicateur de street
        self._widgets["stage_var"] = tk.StringVar(value="PRÉFLOP")
        tk.Label(nav, textvariable=self._widgets["stage_var"],
                 font=self._f_value, bg=THEME["bg_panel"],
                 fg=THEME["accent"]).pack(side=tk.LEFT, padx=16)

        tk.Button(nav, text="Street ▶",
                  command=self._next_stage, **btn_style).pack(side=tk.LEFT, padx=2, pady=6)
        tk.Button(nav, text="Suiv. ▶▶",
                  command=self._next_hand, **btn_style).pack(side=tk.LEFT, padx=4, pady=6)

        # Bouton export
        tk.Button(nav, text="Exporter TXT",
                  command=self._export_hand,
                  font=self._f_small, bg=THEME["bg_panel"],
                  fg=THEME["text_muted"], relief=tk.FLAT,
                  cursor="hand2").pack(side=tk.RIGHT, padx=12, pady=8)

        # Bouton analyser avec Claude
        tk.Button(nav, text="Ré-analyser",
                  command=self._reanalyse,
                  font=self._f_small, bg=THEME["accent"],
                  fg=THEME["bg"], relief=tk.FLAT,
                  cursor="hand2").pack(side=tk.RIGHT, padx=4, pady=8)

    def _build_replay_panel(self, parent: tk.Frame) -> None:
        """Construit le panneau de replay central."""

        # ── Cartes du joueur ─────────────────────────────────────────────
        cards_top = tk.Frame(parent, bg=THEME["bg"])
        cards_top.pack(fill=tk.X, pady=(0, 6))

        tk.Label(cards_top, text="VOS CARTES",
                 font=self._f_small, bg=THEME["bg"],
                 fg=THEME["text_dim"]).pack(anchor=tk.W, padx=4)

        self._widgets["player_cards_frame"] = tk.Frame(cards_top, bg=THEME["bg"])
        self._widgets["player_cards_frame"].pack(anchor=tk.W, padx=4)

        # ── Board ─────────────────────────────────────────────────────────
        board_section = tk.Frame(parent, bg=THEME["bg"])
        board_section.pack(fill=tk.X, pady=(0, 8))

        tk.Label(board_section, text="BOARD",
                 font=self._f_small, bg=THEME["bg"],
                 fg=THEME["text_dim"]).pack(anchor=tk.W, padx=4)

        self._widgets["board_frame"] = tk.Frame(board_section, bg=THEME["bg"])
        self._widgets["board_frame"].pack(anchor=tk.W, padx=4)

        sep = tk.Frame(parent, bg=THEME["border"], height=1)
        sep.pack(fill=tk.X, pady=4)

        # ── Stats de la main ──────────────────────────────────────────────
        stats = tk.Frame(parent, bg=THEME["bg_card"])
        stats.pack(fill=tk.X, pady=(0, 6))

        def stat_row(label, key, pady=2):
            row = tk.Frame(stats, bg=THEME["bg_card"])
            row.pack(fill=tk.X, padx=8, pady=pady)
            tk.Label(row, text=label, font=self._f_label,
                     bg=THEME["bg_card"], fg=THEME["text_dim"],
                     width=18, anchor=tk.W).pack(side=tk.LEFT)
            var = tk.StringVar(value="—")
            self._widgets[key] = var
            tk.Label(row, textvariable=var, font=self._f_label,
                     bg=THEME["bg_card"], fg=THEME["text"]).pack(side=tk.LEFT)
            return var

        stat_row("Classe de main :",   "stat_hand_class")
        stat_row("Équité à ce stade :", "stat_equity")
        stat_row("Pot :",               "stat_pot")
        stat_row("Adversaires :",       "stat_opponents")
        stat_row("EV estimée :",        "stat_ev")

        sep2 = tk.Frame(parent, bg=THEME["border"], height=1)
        sep2.pack(fill=tk.X, pady=4)

        # ── Conseil vs Action ─────────────────────────────────────────────
        actions_frame = tk.Frame(parent, bg=THEME["bg"])
        actions_frame.pack(fill=tk.X, pady=(0, 4))

        # Conseil
        left_col = tk.Frame(actions_frame, bg=THEME["bg"])
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(left_col, text="CONSEIL",
                 font=self._f_small, bg=THEME["bg"],
                 fg=THEME["text_dim"]).pack()
        self._widgets["recommended_lbl"] = tk.Label(
            left_col, text="—", font=self._f_action,
            bg=THEME["bg"], fg=THEME["text"])
        self._widgets["recommended_lbl"].pack()

        # VS
        tk.Label(actions_frame, text="VS",
                 font=self._f_small, bg=THEME["bg"],
                 fg=THEME["text_dim"]).pack(side=tk.LEFT, padx=8)

        # Action prise
        right_col = tk.Frame(actions_frame, bg=THEME["bg"])
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(right_col, text="ACTION PRISE",
                 font=self._f_small, bg=THEME["bg"],
                 fg=THEME["text_dim"]).pack()
        self._widgets["taken_lbl"] = tk.Label(
            right_col, text="—", font=self._f_action,
            bg=THEME["bg"], fg=THEME["text"])
        self._widgets["taken_lbl"].pack()

        # Résultat
        sep3 = tk.Frame(parent, bg=THEME["border"], height=1)
        sep3.pack(fill=tk.X, pady=4)

        result_row = tk.Frame(parent, bg=THEME["bg"])
        result_row.pack(fill=tk.X, padx=8)
        tk.Label(result_row, text="RÉSULTAT :",
                 font=self._f_label, bg=THEME["bg"],
                 fg=THEME["text_dim"]).pack(side=tk.LEFT)
        self._widgets["result_var"] = tk.StringVar(value="—")
        self._widgets["result_lbl"] = tk.Label(
            result_row, textvariable=self._widgets["result_var"],
            font=self._f_value, bg=THEME["bg"], fg=THEME["text"])
        self._widgets["result_lbl"].pack(side=tk.LEFT, padx=8)

        # Suivi du conseil
        self._widgets["followed_var"] = tk.StringVar(value="")
        tk.Label(result_row, textvariable=self._widgets["followed_var"],
                 font=self._f_small, bg=THEME["bg"],
                 fg=THEME["text_muted"]).pack(side=tk.LEFT)

        # Notes
        notes_frame = tk.Frame(parent, bg=THEME["bg"])
        notes_frame.pack(fill=tk.X, padx=8, pady=(4, 0))
        self._widgets["notes_var"] = tk.StringVar(value="")
        tk.Label(notes_frame, textvariable=self._widgets["notes_var"],
                 font=self._f_small, bg=THEME["bg"],
                 fg=THEME["text_dim"], wraplength=520,
                 justify=tk.LEFT).pack(anchor=tk.W)

    # ------------------------------------------------------------------
    # Chargement et filtrage
    # ------------------------------------------------------------------

    def _load_hands(self) -> None:
        """Charge les mains depuis la BD et peuple la liste."""
        all_hands = load_hands_from_db(self.tracker, limit=500)

        # Peupler le combo sessions
        session_ids = sorted(set(h.session_id for h in all_hands), reverse=True)
        sessions_labels = ["Toutes"] + [f"Session #{sid}" for sid in session_ids]
        self._widgets["session_combo"]["values"] = sessions_labels
        self._session_ids = [None] + session_ids

        self._all_hands = all_hands
        self._apply_filter()

    def _apply_filter(self, event=None) -> None:
        """Filtre les mains selon les critères sélectionnés."""
        # Filtre session
        session_sel = self._widgets["session_var"].get()
        try:
            idx = self._widgets["session_combo"]["values"].index(session_sel)
            session_filter = self._session_ids[idx]
        except (ValueError, IndexError, AttributeError):
            session_filter = None

        # Filtre résultat
        result_filter = self._widgets["filter_var"].get()

        hands = self._all_hands
        if session_filter:
            hands = [h for h in hands if h.session_id == session_filter]

        if result_filter == "Gains":
            hands = [h for h in hands if h.result > 0]
        elif result_filter == "Pertes":
            hands = [h for h in hands if h.result < 0]
        elif result_filter == "Conseil ignoré":
            hands = [h for h in hands if not h.followed_advice]
        elif result_filter == "ALL-IN":
            hands = [h for h in hands if h.action_taken == "ALL-IN"
                     or h.recommended_action == "ALL-IN"]
        elif result_filter == "RAISE":
            hands = [h for h in hands if h.action_taken == "RAISE"
                     or h.recommended_action == "RAISE"]

        self.hands    = hands
        self.hand_idx = 0
        self.stage_idx= 0

        # Peupler la listbox
        lb = self._widgets["hand_list"]
        lb.delete(0, tk.END)
        for h in self.hands:
            sign   = "+" if h.result > 0 else ""
            status = "✓" if h.followed_advice else "✗"
            label  = f"{status} {h.date_str} {sign}{h.result:.0f}$ [{h.hand_class[:8]}]"
            lb.insert(tk.END, label)
            color = THEME["green"] if h.result > 0 else THEME["red"] if h.result < 0 else THEME["text_muted"]
            lb.itemconfig(tk.END, fg=color)

        self._update_counter()
        if self.hands:
            lb.select_set(0)
            self._show_hand()

    def _on_filter_change(self, event=None) -> None:
        self._apply_filter()

    def _on_hand_select(self, event=None) -> None:
        selection = self._widgets["hand_list"].curselection()
        if selection:
            self.hand_idx  = selection[0]
            self.stage_idx = 0
            self._show_hand()

    # ------------------------------------------------------------------
    # Affichage
    # ------------------------------------------------------------------

    def _show_hand(self) -> None:
        """Affiche la main courante au stage courant."""
        if not self.hands:
            return

        hand  = self.hands[self.hand_idx]
        stage = self.STAGES[min(self.stage_idx, len(self.STAGES) - 1)]

        # Limiter au stage final de la main
        max_stage_idx = self.STAGES.index(hand.stage_final) if hand.stage_final in self.STAGES else 3
        if self.stage_idx > max_stage_idx:
            self.stage_idx = max_stage_idx
            stage = self.STAGES[self.stage_idx]

        # Mettre à jour l'indicateur de street
        self._widgets["stage_var"].set(stage.upper())

        # ── Cartes du joueur ─────────────────────────────────────────────
        self._render_cards(
            self._widgets["player_cards_frame"],
            hand.player_cards,
            size=26,
        )

        # ── Board progressif ──────────────────────────────────────────────
        board_visible = hand.cards_at_stage.get(stage, [])
        self._render_cards(
            self._widgets["board_frame"],
            board_visible,
            placeholder=5,
            size=22,
        )

        # ── Stats recalculées via engine ──────────────────────────────────
        equity = self._get_equity(hand, board_visible)

        self._widgets["stat_hand_class"].set(hand.hand_class or "—")
        self._widgets["stat_equity"].set(f"{equity:.1%}" if equity else f"{hand.win_probability:.1%}")
        self._widgets["stat_pot"].set(f"{hand.pot_final:.0f}$")
        self._widgets["stat_opponents"].set(str(hand.num_opponents))
        self._widgets["stat_ev"].set(f"{hand.ev_estimate:+.1f}$" if hand.ev_estimate else "—")

        # ── Conseil vs Action ─────────────────────────────────────────────
        rec   = hand.recommended_action or "—"
        taken = hand.action_taken or "—"
        rec_color   = ACTION_COLORS.get(rec,   THEME["text"])
        taken_color = ACTION_COLORS.get(taken, THEME["text"])

        self._widgets["recommended_lbl"].configure(text=rec,   fg=rec_color)
        self._widgets["taken_lbl"].configure(text=taken, fg=taken_color)

        # Résultat
        sign   = "+" if hand.result > 0 else ""
        r_color = THEME["green"] if hand.result > 0 else THEME["red"] if hand.result < 0 else THEME["text_muted"]
        self._widgets["result_var"].set(f"{sign}{hand.result:.2f}$")
        self._widgets["result_lbl"].configure(fg=r_color)

        # Suivi
        if hand.followed_advice:
            self._widgets["followed_var"].set("✓ Conseil suivi")
        else:
            self._widgets["followed_var"].set("✗ Conseil ignoré")

        # Notes HH
        if hand.notes:
            notes_short = hand.notes.replace("HH#", "Main #")[:100]
            self._widgets["notes_var"].set(notes_short)
        else:
            self._widgets["notes_var"].set("")

        self._update_counter()

        # Sélectionner la main dans la liste
        lb = self._widgets["hand_list"]
        lb.select_clear(0, tk.END)
        lb.select_set(self.hand_idx)
        lb.see(self.hand_idx)

    def _render_cards(
        self,
        frame: tk.Frame,
        cards: list[str],
        placeholder: int = 0,
        size: int = 22,
    ) -> None:
        """Affiche les cartes dans un frame tkinter."""
        for widget in frame.winfo_children():
            widget.destroy()

        for card in cards:
            if len(card) < 2:
                continue
            rank = card[0]
            suit = card[1].lower() if len(card) > 1 else "s"
            sym  = SUIT_SYMBOLS.get(suit, suit)
            color= SUIT_COLORS.get(suit, THEME["text"])

            card_frame = tk.Frame(frame, bg="#F0F0F0", relief=tk.RAISED,
                                  bd=1, padx=4, pady=2)
            card_frame.pack(side=tk.LEFT, padx=2)

            tk.Label(card_frame,
                     text=f"{rank}{sym}",
                     font=tkfont.Font(family=THEME["font"], size=size, weight="bold"),
                     bg="#F0F0F0", fg=color).pack()

        # Placeholders (cartes face-down)
        revealed = len(cards)
        for _ in range(max(0, placeholder - revealed)):
            ph = tk.Frame(frame, bg="#2C3E50", relief=tk.RAISED,
                          bd=1, padx=6, pady=4)
            ph.pack(side=tk.LEFT, padx=2)
            tk.Label(ph, text="?",
                     font=tkfont.Font(family=THEME["font"], size=size, weight="bold"),
                     bg="#2C3E50", fg="#5A6478").pack()

    def _get_equity(self, hand: ReplayHand, board: list[str]) -> float:
        """Calcule l'équité via engine.py pour le stage courant."""
        cache_key = f"{hand.hand_id}_{len(board)}"
        if cache_key in self._equity_cache:
            return self._equity_cache[cache_key]

        try:
            from engine import PokerEngine
            engine = PokerEngine(simulations=800)
            result = engine.analyse(
                hole_cards    = hand.player_cards,
                board         = board,
                num_opponents = max(1, hand.num_opponents),
            )
            equity = result.win_probability
            self._equity_cache[cache_key] = equity
            return equity
        except Exception as e:
            log.debug(f"Équité non calculable : {e}")
            return hand.win_probability

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _next_stage(self) -> None:
        if not self.hands:
            return
        hand = self.hands[self.hand_idx]
        max_idx = self.STAGES.index(hand.stage_final) if hand.stage_final in self.STAGES else 3
        if self.stage_idx < max_idx:
            self.stage_idx += 1
            self._show_hand()

    def _prev_stage(self) -> None:
        if not self.hands:
            return
        if self.stage_idx > 0:
            self.stage_idx -= 1
            self._show_hand()

    def _next_hand(self) -> None:
        if not self.hands:
            return
        self.hand_idx  = (self.hand_idx + 1) % len(self.hands)
        self.stage_idx = 0
        self._show_hand()

    def _prev_hand(self) -> None:
        if not self.hands:
            return
        self.hand_idx  = (self.hand_idx - 1) % len(self.hands)
        self.stage_idx = 0
        self._show_hand()

    def _update_counter(self) -> None:
        total = len(self.hands)
        idx   = self.hand_idx + 1 if self.hands else 0
        self._widgets["counter_var"].set(f"{idx} / {total}")

    # ------------------------------------------------------------------
    # Export et ré-analyse
    # ------------------------------------------------------------------

    def _export_hand(self) -> None:
        """Exporte la main courante en texte."""
        if not self.hands:
            return
        hand = self.hands[self.hand_idx]
        text = self._hand_to_text(hand)

        from pathlib import Path
        export_dir = Path(__file__).parent / "exports"
        export_dir.mkdir(exist_ok=True)
        path = export_dir / f"main_{hand.hand_id}.txt"
        path.write_text(text, encoding="utf-8")
        log.info(f"Main exportée : {path}")

        # Feedback visuel
        if self.root:
            self.root.title(f"♠ Replay — Exportée : {path.name}")
            self.root.after(3000, lambda: self.root.title("♠ Replay de Mains"))

    def _reanalyse(self) -> None:
        """Ré-analyse la main avec engine.py et met à jour l'affichage."""
        if not self.hands:
            return
        self._equity_cache.clear()
        self._show_hand()

    @staticmethod
    def _hand_to_text(hand: ReplayHand) -> str:
        board_str = " ".join(hand.board_cards) if hand.board_cards else "(aucun)"
        follow    = "OUI" if hand.followed_advice else "NON"
        sign      = "+" if hand.result > 0 else ""
        return (
            f"=== Main #{hand.hand_id} — {hand.date_str} ===\n"
            f"Cartes       : {' '.join(hand.player_cards)}\n"
            f"Board        : {board_str}\n"
            f"Stage final  : {hand.stage_final.upper()}\n"
            f"Classe       : {hand.hand_class}\n"
            f"Équité       : {hand.win_probability:.1%}\n"
            f"Pot          : {hand.pot_final:.0f}$\n"
            f"Adversaires  : {hand.num_opponents}\n"
            f"Conseil      : {hand.recommended_action}\n"
            f"Action prise : {hand.action_taken}\n"
            f"Conseil suivi: {follow}\n"
            f"Résultat     : {sign}{hand.result:.2f}$\n"
            f"EV estimée   : {hand.ev_estimate:+.1f}$\n"
            f"Notes        : {hand.notes}\n"
        )


# ---------------------------------------------------------------------------
# Intégration dans stats_viewer.py
# ---------------------------------------------------------------------------

def add_replay_tab_to_viewer(viewer, tracker) -> None:
    """
    Ajoute l'onglet 'Replay' dans le dashboard stats_viewer.py existant.

    Usage dans stats_viewer.py run() :
        from hand_replay import add_replay_tab_to_viewer
        add_replay_tab_to_viewer(self, self.tracker)
    """
    import tkinter as tk
    from tkinter import ttk

    tab = tk.Frame(viewer.notebook, bg=THEME["bg"])
    viewer.notebook.add(tab, text="  Replay  ")

    # Bouton d'ouverture du viewer complet
    center = tk.Frame(tab, bg=THEME["bg"])
    center.pack(expand=True)

    tk.Label(center,
             text="Replay de Mains",
             font=tkfont.Font(family=THEME["font"], size=14, weight="bold"),
             bg=THEME["bg"], fg=THEME["accent"]).pack(pady=(30, 8))

    tk.Label(center,
             text="Rejouez et analysez chaque main street par street",
             font=tkfont.Font(family=THEME["font"], size=9),
             bg=THEME["bg"], fg=THEME["text_muted"]).pack(pady=(0, 20))

    def open_replay():
        replay_viewer = HandReplayViewer(tracker)
        replay_viewer.run()

    tk.Button(center,
              text="▶  Ouvrir le Replay",
              font=tkfont.Font(family=THEME["font"], size=11, weight="bold"),
              bg=THEME["accent"], fg=THEME["bg"],
              relief=tk.FLAT, cursor="hand2",
              padx=20, pady=8,
              command=open_replay).pack()

    # Mini-stats replay
    tk.Label(center, text="", bg=THEME["bg"]).pack(pady=10)

    try:
        hands = load_hands_from_db(tracker, limit=500)
        n_total   = len(hands)
        n_ignored = sum(1 for h in hands if not h.followed_advice)
        n_profit  = sum(1 for h in hands if h.result > 0)

        stats_frame = tk.Frame(center, bg=THEME["bg_card"], padx=20, pady=12)
        stats_frame.pack()

        def stat_lbl(parent, label, value, color=None):
            row = tk.Frame(parent, bg=THEME["bg_card"])
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=f"{label:<25}",
                     font=tkfont.Font(family=THEME["font"], size=9),
                     bg=THEME["bg_card"], fg=THEME["text_muted"],
                     anchor=tk.W).pack(side=tk.LEFT)
            tk.Label(row, text=value,
                     font=tkfont.Font(family=THEME["font"], size=9, weight="bold"),
                     bg=THEME["bg_card"], fg=color or THEME["text"]).pack(side=tk.LEFT)

        stat_lbl(stats_frame, "Mains disponibles :", str(n_total))
        stat_lbl(stats_frame, "Conseils ignorés :",
                 f"{n_ignored} ({n_ignored/max(n_total,1):.0%})",
                 THEME["orange"] if n_ignored > n_total * 0.3 else THEME["text"])
        stat_lbl(stats_frame, "Mains gagnantes :",
                 f"{n_profit} ({n_profit/max(n_total,1):.0%})",
                 THEME["green"] if n_profit > n_total * 0.5 else THEME["red"])

    except Exception:
        pass


# ---------------------------------------------------------------------------
# Entrée standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Replay interactif de mains poker")
    parser.add_argument("--session", type=int, default=None,
                        help="ID de session à rejouer")
    parser.add_argument("--demo",    action="store_true",
                        help="Créer des données de démo et lancer le replay")
    args = parser.parse_args()

    try:
        from tracker import PokerTracker, HandRecord
        tracker = PokerTracker()
    except ImportError:
        print("ERREUR : tracker.py introuvable.")
        sys.exit(1)

    if args.demo:
        import random, time as _t
        sid = tracker.start_session(buy_in=50.0, game_type="Tournoi")
        stages = ["preflop","flop","turn","river"]
        for i in range(15):
            stage  = stages[i % 4]
            board  = ["Kd","7d","2c","As","3h"][:{"preflop":0,"flop":3,"turn":4,"river":5}[stage]]
            rec    = random.choice(["FOLD","CALL","RAISE","CHECK","BET"])
            taken  = rec if random.random() > 0.4 else random.choice(["FOLD","CALL","RAISE"])
            tracker.record_hand(HandRecord(
                session_id=sid, stage_final=stage,
                player_cards=["Ks","7h"], board_cards=board,
                num_opponents=random.randint(1,5),
                pot_final=random.uniform(20,200),
                hand_class=random.choice(["Paire","Deux Paires","Haute Carte","Brelan"]),
                win_probability=random.uniform(0.25,0.85),
                recommended_action=rec, action_taken=taken,
                followed_advice=(rec==taken),
                result=random.gauss(0, 30),
                ev_estimate=random.uniform(5,40),
                ev_realized=random.gauss(0, 30),
            ))
        tracker.end_session(sid, placement=2, prize=80.0)
        args.session = sid
        print(f"Session démo #{sid} créée — lancement du replay…\n")

    viewer = HandReplayViewer(tracker, session_id=args.session)
    viewer.run()
