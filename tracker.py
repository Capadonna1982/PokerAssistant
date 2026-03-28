"""
tracker.py — Enregistrement des gains, pertes et décisions poker
Dépendances : sqlite3 (inclus dans Python standard)

Stocke chaque main, chaque décision et chaque session dans une base
SQLite locale. Aucune dépendance externe requise.
"""

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chemin de la base de données
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).parent / "poker_stats.db"


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------

@dataclass
class HandRecord:
    """Enregistrement complet d'une main jouée."""
    session_id:          int
    stage_final:         str          # preflop / flop / turn / river
    player_cards:        list[str]
    board_cards:         list[str]
    num_opponents:       int
    pot_final:           float
    hand_class:          str          # Paire, Deux Paires, etc.
    win_probability:     float        # equity au moment de la décision
    recommended_action:  str          # conseil de Claude/engine
    action_taken:        str          # action réellement jouée (saisie manuelle)
    followed_advice:     bool         # a-t-on suivi le conseil ?
    result:              float        # gain (+) ou perte (-) en $
    ev_estimate:         float        # EV théorique
    ev_realized:         float        # EV réalisée (= result)
    notes:               str   = ""
    timestamp:           float = field(default_factory=time.time)

    @property
    def datetime_str(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class SessionRecord:
    """Enregistrement d'une session de jeu."""
    buy_in:        float
    game_type:     str    = "Tournoi"   # Tournoi / Cash Game
    num_players:   int    = 8
    notes:         str    = ""
    start_time:    float  = field(default_factory=time.time)
    end_time:      float  = 0.0
    final_stack:   float  = 0.0
    placement:     int    = 0           # place finale (tournoi)
    prize:         float  = 0.0         # gain du tournoi

    @property
    def profit(self) -> float:
        return self.prize - self.buy_in if self.prize else self.final_stack - self.buy_in

    @property
    def duration_minutes(self) -> float:
        end = self.end_time or time.time()
        return (end - self.start_time) / 60


# ---------------------------------------------------------------------------
# Gestionnaire de base de données
# ---------------------------------------------------------------------------

class PokerTracker:
    """
    Gère l'enregistrement de toutes les données de jeu dans SQLite.

    Usage :
        tracker = PokerTracker()

        # Début de session
        session_id = tracker.start_session(buy_in=50.0, game_type="Tournoi")

        # Enregistrer une main
        tracker.record_hand(HandRecord(
            session_id       = session_id,
            stage_final      = "flop",
            player_cards     = ["Ks", "7h"],
            board_cards      = ["Kd", "7d", "2c"],
            num_opponents    = 3,
            pot_final        = 120.0,
            hand_class       = "Deux Paires",
            win_probability  = 0.87,
            recommended_action = "RAISE",
            action_taken     = "RAISE",
            followed_advice  = True,
            result           = 120.0,
            ev_estimate      = 68.4,
            ev_realized      = 120.0,
        ))

        # Fin de session
        tracker.end_session(session_id, final_stack=0.0, placement=1, prize=200.0)
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()
        log.info(f"PokerTracker initialisé → {self.db_path}")

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Crée les tables si elles n'existent pas."""
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time   REAL    NOT NULL,
                    end_time     REAL    DEFAULT 0,
                    game_type    TEXT    DEFAULT 'Tournoi',
                    num_players  INTEGER DEFAULT 8,
                    buy_in       REAL    DEFAULT 0,
                    final_stack  REAL    DEFAULT 0,
                    placement    INTEGER DEFAULT 0,
                    prize        REAL    DEFAULT 0,
                    notes        TEXT    DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS hands (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id          INTEGER NOT NULL,
                    timestamp           REAL    NOT NULL,
                    stage_final         TEXT,
                    player_cards        TEXT,
                    board_cards         TEXT,
                    num_opponents       INTEGER DEFAULT 0,
                    pot_final           REAL    DEFAULT 0,
                    hand_class          TEXT,
                    win_probability     REAL    DEFAULT 0,
                    recommended_action  TEXT,
                    action_taken        TEXT,
                    followed_advice     INTEGER DEFAULT 0,
                    result              REAL    DEFAULT 0,
                    ev_estimate         REAL    DEFAULT 0,
                    ev_realized         REAL    DEFAULT 0,
                    notes               TEXT    DEFAULT '',
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS decisions (
                    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                    hand_id            INTEGER NOT NULL,
                    session_id         INTEGER NOT NULL,
                    timestamp          REAL    NOT NULL,
                    stage              TEXT,
                    player_cards       TEXT,
                    board_cards        TEXT,
                    win_probability    REAL    DEFAULT 0,
                    recommended_action TEXT,
                    action_taken       TEXT,
                    followed_advice    INTEGER DEFAULT 0,
                    pot_size           REAL    DEFAULT 0,
                    bet_size           REAL    DEFAULT 0,
                    ev_estimate        REAL    DEFAULT 0,
                    FOREIGN KEY (hand_id)    REFERENCES hands(id),
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );
            """)
        log.debug("Tables SQLite initialisées.")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            log.error(f"Erreur SQLite : {e}")
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def start_session(
        self,
        buy_in:      float = 0.0,
        game_type:   str   = "Tournoi",
        num_players: int   = 8,
        notes:       str   = "",
    ) -> int:
        """Démarre une nouvelle session. Retourne l'ID de session."""
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO sessions
                   (start_time, game_type, num_players, buy_in, notes)
                   VALUES (?, ?, ?, ?, ?)""",
                (time.time(), game_type, num_players, buy_in, notes),
            )
            session_id = cur.lastrowid
        log.info(f"Session #{session_id} démarrée — {game_type} | buy-in={buy_in}$")
        return session_id

    def end_session(
        self,
        session_id:  int,
        final_stack: float = 0.0,
        placement:   int   = 0,
        prize:       float = 0.0,
        notes:       str   = "",
    ) -> None:
        """Clôture une session avec le résultat final."""
        with self._conn() as conn:
            conn.execute(
                """UPDATE sessions
                   SET end_time=?, final_stack=?, placement=?, prize=?, notes=?
                   WHERE id=?""",
                (time.time(), final_stack, placement, prize, notes, session_id),
            )
        profit = prize - self._get_buy_in(session_id) if prize else final_stack
        log.info(
            f"Session #{session_id} terminée — "
            f"placement={placement} | prize={prize}$ | profit={profit:+.2f}$"
        )

    def _get_buy_in(self, session_id: int) -> float:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT buy_in FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
        return row["buy_in"] if row else 0.0

    # ------------------------------------------------------------------
    # Mains
    # ------------------------------------------------------------------

    def record_hand(self, hand: HandRecord) -> int:
        """Enregistre une main complète. Retourne l'ID de la main."""
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO hands
                   (session_id, timestamp, stage_final, player_cards, board_cards,
                    num_opponents, pot_final, hand_class, win_probability,
                    recommended_action, action_taken, followed_advice,
                    result, ev_estimate, ev_realized, notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    hand.session_id,
                    hand.timestamp,
                    hand.stage_final,
                    json.dumps(hand.player_cards),
                    json.dumps(hand.board_cards),
                    hand.num_opponents,
                    hand.pot_final,
                    hand.hand_class,
                    hand.win_probability,
                    hand.recommended_action,
                    hand.action_taken,
                    int(hand.followed_advice),
                    hand.result,
                    hand.ev_estimate,
                    hand.ev_realized,
                    hand.notes,
                ),
            )
            hand_id = cur.lastrowid
        log.info(
            f"Main #{hand_id} enregistrée — "
            f"{hand.hand_class} | {hand.action_taken} | résultat={hand.result:+.2f}$"
        )
        return hand_id

    def record_decision(
        self,
        hand_id:            int,
        session_id:         int,
        stage:              str,
        player_cards:       list[str],
        board_cards:        list[str],
        win_probability:    float,
        recommended_action: str,
        action_taken:       str,
        pot_size:           float = 0.0,
        bet_size:           float = 0.0,
        ev_estimate:        float = 0.0,
    ) -> None:
        """Enregistre une décision individuelle (une street)."""
        followed = action_taken.upper() == recommended_action.upper()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO decisions
                   (hand_id, session_id, timestamp, stage, player_cards, board_cards,
                    win_probability, recommended_action, action_taken, followed_advice,
                    pot_size, bet_size, ev_estimate)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    hand_id, session_id, time.time(), stage,
                    json.dumps(player_cards), json.dumps(board_cards),
                    win_probability, recommended_action, action_taken,
                    int(followed), pot_size, bet_size, ev_estimate,
                ),
            )

    # ------------------------------------------------------------------
    # Requêtes statistiques
    # ------------------------------------------------------------------

    def get_session_stats(self, session_id: int) -> dict:
        """Retourne les stats complètes d'une session."""
        with self._conn() as conn:
            session = conn.execute(
                "SELECT * FROM sessions WHERE id=?", (session_id,)
            ).fetchone()

            hands = conn.execute(
                "SELECT * FROM hands WHERE session_id=?", (session_id,)
            ).fetchall()

        if not session:
            return {}

        total_hands    = len(hands)
        total_profit   = sum(h["result"] for h in hands)
        wins           = sum(1 for h in hands if h["result"] > 0)
        followed       = sum(1 for h in hands if h["followed_advice"])
        ev_total       = sum(h["ev_estimate"] for h in hands)
        ev_realized    = sum(h["ev_realized"] for h in hands)

        return {
            "session_id":        session_id,
            "game_type":         session["game_type"],
            "buy_in":            session["buy_in"],
            "prize":             session["prize"],
            "profit":            session["prize"] - session["buy_in"] if session["prize"] else total_profit,
            "placement":         session["placement"],
            "duration_minutes":  (session["end_time"] - session["start_time"]) / 60 if session["end_time"] else 0,
            "total_hands":       total_hands,
            "win_rate":          wins / total_hands if total_hands else 0,
            "advice_follow_rate": followed / total_hands if total_hands else 0,
            "total_ev_estimate": round(ev_total, 2),
            "total_ev_realized": round(ev_realized, 2),
            "ev_efficiency":     round(ev_realized / ev_total, 2) if ev_total else 0,
        }

    def get_global_stats(self) -> dict:
        """Retourne les stats globales toutes sessions confondues."""
        with self._conn() as conn:
            sessions = conn.execute("SELECT * FROM sessions WHERE end_time > 0").fetchall()
            hands    = conn.execute("SELECT * FROM hands").fetchall()
            decisions = conn.execute("SELECT * FROM decisions").fetchall()

        total_sessions  = len(sessions)
        total_hands     = len(hands)
        total_invested  = sum(s["buy_in"] for s in sessions)
        total_prize     = sum(s["prize"] for s in sessions)
        total_profit    = total_prize - total_invested
        roi             = (total_profit / total_invested * 100) if total_invested else 0

        wins            = sum(1 for h in hands if h["result"] > 0)
        followed        = sum(1 for d in decisions if d["followed_advice"])
        profitable_when_followed = [
            h["result"] for h in hands if h["followed_advice"] and h["result"] > 0
        ]

        hand_classes = {}
        for h in hands:
            cls = h["hand_class"] or "Inconnue"
            if cls not in hand_classes:
                hand_classes[cls] = {"count": 0, "profit": 0.0}
            hand_classes[cls]["count"]  += 1
            hand_classes[cls]["profit"] += h["result"]

        best_hand = max(hand_classes.items(), key=lambda x: x[1]["profit"]) if hand_classes else None
        worst_hand = min(hand_classes.items(), key=lambda x: x[1]["profit"]) if hand_classes else None

        return {
            "total_sessions":     total_sessions,
            "total_hands":        total_hands,
            "total_invested":     round(total_invested, 2),
            "total_prize":        round(total_prize, 2),
            "total_profit":       round(total_profit, 2),
            "roi_pct":            round(roi, 2),
            "win_rate_hands":     round(wins / total_hands, 4) if total_hands else 0,
            "advice_follow_rate": round(followed / len(decisions), 4) if decisions else 0,
            "avg_profit_per_session": round(total_profit / total_sessions, 2) if total_sessions else 0,
            "best_hand_class":    best_hand[0] if best_hand else "—",
            "worst_hand_class":   worst_hand[0] if worst_hand else "—",
            "hand_class_breakdown": hand_classes,
        }

    def get_recent_sessions(self, limit: int = 20) -> list[dict]:
        """Retourne les N dernières sessions."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT s.*,
                          COUNT(h.id) as hand_count,
                          SUM(h.followed_advice) as followed_count
                   FROM sessions s
                   LEFT JOIN hands h ON h.session_id = s.id
                   WHERE s.end_time > 0
                   GROUP BY s.id
                   ORDER BY s.start_time DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()

        result = []
        for row in rows:
            profit = row["prize"] - row["buy_in"] if row["prize"] else row["final_stack"] - row["buy_in"]
            result.append({
                "id":           row["id"],
                "date":         datetime.fromtimestamp(row["start_time"]).strftime("%Y-%m-%d %H:%M"),
                "game_type":    row["game_type"],
                "buy_in":       row["buy_in"],
                "prize":        row["prize"],
                "profit":       round(profit, 2),
                "placement":    row["placement"],
                "hand_count":   row["hand_count"] or 0,
                "follow_rate":  round((row["followed_count"] or 0) / max(row["hand_count"] or 1, 1) * 100, 1),
                "duration_min": round((row["end_time"] - row["start_time"]) / 60, 1) if row["end_time"] else 0,
            })
        return result

    def get_advice_performance(self) -> dict:
        """Compare les résultats quand on suit vs ignore les conseils."""
        with self._conn() as conn:
            followed = conn.execute(
                "SELECT AVG(result) as avg, COUNT(*) as cnt FROM hands WHERE followed_advice=1"
            ).fetchone()
            ignored = conn.execute(
                "SELECT AVG(result) as avg, COUNT(*) as cnt FROM hands WHERE followed_advice=0"
            ).fetchone()

        return {
            "followed": {
                "avg_result": round(followed["avg"] or 0, 2),
                "count":      followed["cnt"] or 0,
            },
            "ignored": {
                "avg_result": round(ignored["avg"] or 0, 2),
                "count":      ignored["cnt"] or 0,
            },
        }

    def get_profit_over_time(self, limit: int = 50) -> list[dict]:
        """Retourne l'évolution du profit session par session."""
        sessions = self.get_recent_sessions(limit)
        sessions.reverse()
        cumulative = 0.0
        result = []
        for s in sessions:
            cumulative += s["profit"]
            result.append({
                "date":       s["date"],
                "profit":     s["profit"],
                "cumulative": round(cumulative, 2),
            })
        return result
