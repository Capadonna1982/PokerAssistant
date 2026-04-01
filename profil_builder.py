"""
profil_builder.py — Construction automatique des profils adverses (VPIP/PFR/AF)
Dépendances : tracker.py, hh_parser.py (déjà installés)

Sources de données :
  1. Hand History PokerStars (.txt) — le plus complet
  2. action_detector.py — actions détectées en temps réel
  3. Import manuel — stats SharkScope ou autres

Statistiques calculées :
  VPIP  (Voluntarily Put money In Pot)  — % de mains où l'adversaire entre volontairement
  PFR   (Pre-Flop Raise)               — % de mains où il relance préflop
  AF    (Aggression Factor)             — (bet+raise) / call (postflop)
  WTSD  (Went To ShowDown)             — % de mains allées au showdown
  W$SD  (Won money at ShowDown)        — % de showdowns gagnés
  3bet% — % de 3-bets face à une ouverture
  CBet% — % de continuation bets faits
  FoldCBet% — % de fois qu'il fold face au CBet adverse
  Tendance — profil synthétique (Nit/TAG/LAG/Loose Passive/Calling Station)

Usage :
    builder = ProfilBuilder(tracker)
    builder.build_from_hh_folder(Path("C:/...HandHistory/Pseudo/"))
    profile = builder.get_profile("PlayerName")
    print(profile.tendency)

    # Temps réel (depuis action_detector)
    builder.update_from_action(seat=3, action="RAISE", amount=80, stage="flop")
    builder.update_from_action(seat=3, action="FOLD",  amount=0,  stage="preflop")
"""

import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes de classification des profils
# ---------------------------------------------------------------------------

# Seuils VPIP/PFR/AF pour classification automatique
PROFILE_THRESHOLDS = {
    "Nit":             {"vpip_max": 15, "pfr_max": 12},
    "TAG":             {"vpip_min": 15, "vpip_max": 28, "pfr_min": 12, "pfr_max": 22},
    "LAG":             {"vpip_min": 28, "pfr_min": 20},
    "Calling Station": {"vpip_min": 28, "pfr_max": 12, "af_max": 1.0},
    "Loose Passive":   {"vpip_min": 20, "pfr_max": 10},
    "Maniac":          {"vpip_min": 40, "pfr_min": 30, "af_min": 3.0},
}

# Tendances → conseils stratégiques
TENDENCY_ADVICE = {
    "Nit":             "Joue uniquement les mains premium. Fold si résistance.",
    "TAG":             "Joueur solide. Respecte ses relances, évite de bluffer.",
    "LAG":             "Agressif et large. Contre-attaque avec de bonnes mains.",
    "Calling Station": "Ne bluffe pas. Mise pour valeur max avec tes mains fortes.",
    "Loose Passive":   "Mise pour valeur. Bluffs inutiles, il call trop souvent.",
    "Maniac":          "Laisse-le bluffer. Call plus large, trap avec tes bonnes mains.",
    "Inconnu":         "Profil insuffisant. Joue GTO en attendant plus de données.",
}

# Minimum de mains pour un profil fiable
MIN_HANDS_RELIABLE  = 30
MIN_HANDS_ESTIMATE  = 10


# ---------------------------------------------------------------------------
# Structure du profil
# ---------------------------------------------------------------------------

@dataclass
class OpponentProfile:
    """Profil statistique complet d'un adversaire."""
    name:           str

    # Compteurs bruts
    hands_total:    int   = 0
    hands_vpip:     int   = 0    # mains où il a volontairement misé/callé
    hands_pfr:      int   = 0    # mains où il a relancé préflop
    hands_3bet:     int   = 0    # 3-bets effectués
    hands_faced_3bet:int  = 0    # fois il a fait face à un 3-bet
    hands_fold_3bet:int   = 0    # folds face au 3-bet

    # Postflop
    postflop_bets:  int   = 0    # nb de bets postflop
    postflop_raises:int   = 0    # nb de raises postflop
    postflop_calls: int   = 0    # nb de calls postflop
    postflop_checks:int   = 0    # nb de checks postflop
    postflop_folds: int   = 0    # nb de folds postflop

    # CBet
    cbet_opps:      int   = 0    # fois qu'il pouvait cbet
    cbet_done:      int   = 0    # fois qu'il a cbetté
    faced_cbet:     int   = 0    # fois il a fait face au cbet
    folded_cbet:    int   = 0    # fois il a fold face au cbet

    # Showdown
    wtsd:           int   = 0    # went to showdown
    wsd:            int   = 0    # won at showdown

    # Montants
    total_won:      float = 0.0
    total_lost:     float = 0.0
    bb_per_100:     float = 0.0  # win rate en BB/100

    # Données table courante (session en cours)
    session_hands:  int   = 0
    last_seen:      float = field(default_factory=time.time)
    current_table:  str   = ""

    # ── Propriétés calculées ──────────────────────────────────────────────

    @property
    def vpip(self) -> float:
        """VPIP en % (0–100)."""
        return round(self.hands_vpip / max(self.hands_total, 1) * 100, 1)

    @property
    def pfr(self) -> float:
        """PFR en % (0–100)."""
        return round(self.hands_pfr / max(self.hands_total, 1) * 100, 1)

    @property
    def af(self) -> float:
        """Aggression Factor = (bet+raise) / call postflop."""
        passive = max(self.postflop_calls, 1)
        aggressive = self.postflop_bets + self.postflop_raises
        return round(aggressive / passive, 2)

    @property
    def three_bet_pct(self) -> float:
        return round(self.hands_3bet / max(self.hands_total, 1) * 100, 1)

    @property
    def fold_to_3bet_pct(self) -> float:
        return round(self.hands_fold_3bet / max(self.hands_faced_3bet, 1) * 100, 1)

    @property
    def cbet_pct(self) -> float:
        return round(self.cbet_done / max(self.cbet_opps, 1) * 100, 1)

    @property
    def fold_to_cbet_pct(self) -> float:
        return round(self.folded_cbet / max(self.faced_cbet, 1) * 100, 1)

    @property
    def wtsd_pct(self) -> float:
        return round(self.wtsd / max(self.hands_total, 1) * 100, 1)

    @property
    def wsd_pct(self) -> float:
        return round(self.wsd / max(self.wtsd, 1) * 100, 1)

    @property
    def is_reliable(self) -> bool:
        return self.hands_total >= MIN_HANDS_RELIABLE

    @property
    def is_estimate(self) -> bool:
        return MIN_HANDS_ESTIMATE <= self.hands_total < MIN_HANDS_RELIABLE

    @property
    def tendency(self) -> str:
        """Classification automatique du profil."""
        v = self.vpip
        p = self.pfr
        a = self.af

        if self.hands_total < MIN_HANDS_ESTIMATE:
            return "Inconnu"
        if v >= 40 and p >= 30 and a >= 3.0:
            return "Maniac"
        if v <= 15 and p <= 12:
            return "Nit"
        if 15 <= v <= 28 and 12 <= p <= 22:
            return "TAG"
        if v >= 28 and p >= 20:
            return "LAG"
        if v >= 28 and p <= 12 and a <= 1.0:
            return "Calling Station"
        if v >= 20 and p <= 10:
            return "Loose Passive"
        return "TAG"   # défaut raisonnable

    @property
    def strategic_advice(self) -> str:
        return TENDENCY_ADVICE.get(self.tendency, TENDENCY_ADVICE["Inconnu"])

    @property
    def reliability_label(self) -> str:
        if self.is_reliable:
            return f"Fiable ({self.hands_total} mains)"
        if self.is_estimate:
            return f"Estimé ({self.hands_total} mains)"
        return f"Insuffisant ({self.hands_total} mains)"

    def to_dict(self) -> dict:
        return {
            "name":              self.name,
            "hands_total":       self.hands_total,
            "vpip":              self.vpip,
            "pfr":               self.pfr,
            "af":                self.af,
            "3bet_pct":          self.three_bet_pct,
            "fold_to_3bet_pct":  self.fold_to_3bet_pct,
            "cbet_pct":          self.cbet_pct,
            "fold_to_cbet_pct":  self.fold_to_cbet_pct,
            "wtsd_pct":          self.wtsd_pct,
            "wsd_pct":           self.wsd_pct,
            "bb_per_100":        self.bb_per_100,
            "tendency":          self.tendency,
            "strategic_advice":  self.strategic_advice,
            "reliability":       self.reliability_label,
            "session_hands":     self.session_hands,
        }

    def to_claude_context(self) -> dict:
        """Format optimisé pour le prompt Claude."""
        return {
            "vpip":      self.vpip,
            "pfr":       self.pfr,
            "af":        self.af,
            "tendency":  self.tendency,
            "3bet":      self.three_bet_pct,
            "fold_cbet": self.fold_to_cbet_pct,
            "hands":     self.hands_total,
            "advice":    self.strategic_advice,
        }

    def hud_line(self) -> str:
        """Ligne compacte pour le HUD overlay."""
        reliability = "✓" if self.is_reliable else ("~" if self.is_estimate else "?")
        return (
            f"{self.name[:12]:<12} "
            f"V:{self.vpip:.0f}% P:{self.pfr:.0f}% AF:{self.af:.1f} "
            f"[{self.tendency[:4]}]{reliability}"
        )


# ---------------------------------------------------------------------------
# Base de données des profils
# ---------------------------------------------------------------------------

PROFILES_DB_PATH = Path(__file__).parent / "opponent_profiles.db"


class ProfileDatabase:
    """Stockage SQLite des profils adverses."""

    def __init__(self, db_path: Path = PROFILES_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS profiles (
                    name            TEXT PRIMARY KEY,
                    hands_total     INTEGER DEFAULT 0,
                    hands_vpip      INTEGER DEFAULT 0,
                    hands_pfr       INTEGER DEFAULT 0,
                    hands_3bet      INTEGER DEFAULT 0,
                    hands_faced_3bet INTEGER DEFAULT 0,
                    hands_fold_3bet INTEGER DEFAULT 0,
                    postflop_bets   INTEGER DEFAULT 0,
                    postflop_raises INTEGER DEFAULT 0,
                    postflop_calls  INTEGER DEFAULT 0,
                    postflop_checks INTEGER DEFAULT 0,
                    postflop_folds  INTEGER DEFAULT 0,
                    cbet_opps       INTEGER DEFAULT 0,
                    cbet_done       INTEGER DEFAULT 0,
                    faced_cbet      INTEGER DEFAULT 0,
                    folded_cbet     INTEGER DEFAULT 0,
                    wtsd            INTEGER DEFAULT 0,
                    wsd             INTEGER DEFAULT 0,
                    total_won       REAL    DEFAULT 0,
                    total_lost      REAL    DEFAULT 0,
                    bb_per_100      REAL    DEFAULT 0,
                    last_seen       REAL    DEFAULT 0,
                    current_table   TEXT    DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS profile_notes (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    name      TEXT NOT NULL,
                    note      TEXT NOT NULL,
                    timestamp REAL DEFAULT 0,
                    FOREIGN KEY (name) REFERENCES profiles(name)
                );
            """)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            log.error(f"DB erreur : {e}")
            raise
        finally:
            conn.close()

    def save(self, profile: OpponentProfile) -> None:
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO profiles (
                    name, hands_total, hands_vpip, hands_pfr, hands_3bet,
                    hands_faced_3bet, hands_fold_3bet,
                    postflop_bets, postflop_raises, postflop_calls,
                    postflop_checks, postflop_folds,
                    cbet_opps, cbet_done, faced_cbet, folded_cbet,
                    wtsd, wsd, total_won, total_lost, bb_per_100,
                    last_seen, current_table
                ) VALUES (
                    :name, :hands_total, :hands_vpip, :hands_pfr, :hands_3bet,
                    :hands_faced_3bet, :hands_fold_3bet,
                    :postflop_bets, :postflop_raises, :postflop_calls,
                    :postflop_checks, :postflop_folds,
                    :cbet_opps, :cbet_done, :faced_cbet, :folded_cbet,
                    :wtsd, :wsd, :total_won, :total_lost, :bb_per_100,
                    :last_seen, :current_table
                )
                ON CONFLICT(name) DO UPDATE SET
                    hands_total     = hands_total     + :hands_total,
                    hands_vpip      = hands_vpip      + :hands_vpip,
                    hands_pfr       = hands_pfr       + :hands_pfr,
                    hands_3bet      = hands_3bet      + :hands_3bet,
                    hands_faced_3bet= hands_faced_3bet+ :hands_faced_3bet,
                    hands_fold_3bet = hands_fold_3bet + :hands_fold_3bet,
                    postflop_bets   = postflop_bets   + :postflop_bets,
                    postflop_raises = postflop_raises + :postflop_raises,
                    postflop_calls  = postflop_calls  + :postflop_calls,
                    postflop_checks = postflop_checks + :postflop_checks,
                    postflop_folds  = postflop_folds  + :postflop_folds,
                    cbet_opps       = cbet_opps       + :cbet_opps,
                    cbet_done       = cbet_done       + :cbet_done,
                    faced_cbet      = faced_cbet       + :faced_cbet,
                    folded_cbet     = folded_cbet      + :folded_cbet,
                    wtsd            = wtsd             + :wtsd,
                    wsd             = wsd              + :wsd,
                    total_won       = total_won        + :total_won,
                    total_lost      = total_lost       + :total_lost,
                    last_seen       = :last_seen,
                    current_table   = :current_table
            """, {
                "name":            profile.name,
                "hands_total":     profile.hands_total,
                "hands_vpip":      profile.hands_vpip,
                "hands_pfr":       profile.hands_pfr,
                "hands_3bet":      profile.hands_3bet,
                "hands_faced_3bet":profile.hands_faced_3bet,
                "hands_fold_3bet": profile.hands_fold_3bet,
                "postflop_bets":   profile.postflop_bets,
                "postflop_raises": profile.postflop_raises,
                "postflop_calls":  profile.postflop_calls,
                "postflop_checks": profile.postflop_checks,
                "postflop_folds":  profile.postflop_folds,
                "cbet_opps":       profile.cbet_opps,
                "cbet_done":       profile.cbet_done,
                "faced_cbet":      profile.faced_cbet,
                "folded_cbet":     profile.folded_cbet,
                "wtsd":            profile.wtsd,
                "wsd":             profile.wsd,
                "total_won":       profile.total_won,
                "total_lost":      profile.total_lost,
                "bb_per_100":      profile.bb_per_100,
                "last_seen":       profile.last_seen,
                "current_table":   profile.current_table,
            })

    def load(self, name: str) -> Optional[OpponentProfile]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM profiles WHERE name=?", (name,)
            ).fetchone()
        if not row:
            return None
        p = OpponentProfile(name=name)
        for key in row.keys():
            if hasattr(p, key):
                setattr(p, key, row[key])
        return p

    def load_all(self) -> list[OpponentProfile]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM profiles ORDER BY hands_total DESC"
            ).fetchall()
        profiles = []
        for row in rows:
            p = OpponentProfile(name=row["name"])
            for key in row.keys():
                if hasattr(p, key):
                    setattr(p, key, row[key])
            profiles.append(p)
        return profiles

    def search(self, query: str) -> list[OpponentProfile]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM profiles WHERE name LIKE ? ORDER BY hands_total DESC",
                (f"%{query}%",)
            ).fetchall()
        profiles = []
        for row in rows:
            p = OpponentProfile(name=row["name"])
            for key in row.keys():
                if hasattr(p, key):
                    setattr(p, key, row[key])
            profiles.append(p)
        return profiles

    def add_note(self, name: str, note: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO profile_notes (name, note, timestamp) VALUES (?,?,?)",
                (name, note, time.time())
            )

    def get_notes(self, name: str) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT note FROM profile_notes WHERE name=? ORDER BY timestamp DESC",
                (name,)
            ).fetchall()
        return [row["note"] for row in rows]

    def top_players(self, limit: int = 20, min_hands: int = 10) -> list[OpponentProfile]:
        """Retourne les adversaires les plus fréquents."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM profiles WHERE hands_total >= ? ORDER BY hands_total DESC LIMIT ?",
                (min_hands, limit)
            ).fetchall()
        profiles = []
        for row in rows:
            p = OpponentProfile(name=row["name"])
            for key in row.keys():
                if hasattr(p, key):
                    setattr(p, key, row[key])
            profiles.append(p)
        return profiles


# ---------------------------------------------------------------------------
# Constructeur de profils
# ---------------------------------------------------------------------------

class ProfilBuilder:
    """
    Construit et maintient les profils VPIP/PFR/AF de tous les adversaires.

    Trois sources de données :
      1. Hand History (hh_parser.py) — analyse offline
      2. action_detector.py — mise à jour en temps réel
      3. Import manuel (SharkScope, etc.)

    Usage :
        db      = ProfileDatabase()
        builder = ProfilBuilder(db)

        # Depuis les Hand History
        builder.build_from_hh_folder(Path("C:/..."))

        # En temps réel depuis action_detector
        builder.update_realtime(seat=3, player_name="Villain42",
                                action="RAISE", amount=80, stage="flop")

        # Récupérer un profil
        p = builder.get_profile("Villain42")
        print(p.hud_line())   # Villain42     V:42% P:28% AF:2.1 [LAG]✓

        # Profils de la table courante (pour Claude)
        table_profiles = builder.get_table_profiles(["P1", "P2", "P3"])
    """

    def __init__(self, db: Optional[ProfileDatabase] = None):
        self.db = db or ProfileDatabase()
        # Cache en mémoire pour la session courante
        self._session_cache: dict[str, OpponentProfile] = {}
        # Contexte de la main en cours (pour déterminer PFR, 3bet, etc.)
        self._current_hand_context: dict = {}

    # ------------------------------------------------------------------
    # Construction depuis Hand History
    # ------------------------------------------------------------------

    def build_from_hh_folder(self, folder: Path, hero_name: str = "") -> int:
        """
        Parse tous les fichiers HH du dossier et construit les profils.
        Retourne le nombre de profils mis à jour.
        """
        try:
            from hh_parser import HandHistoryParser
        except ImportError:
            log.error("hh_parser.py introuvable.")
            return 0

        parser      = HandHistoryParser(hero_name=hero_name)
        updated     = set()
        total_hands = 0

        files = list(folder.rglob("*.txt"))
        log.info(f"Analyse de {len(files)} fichiers HH pour profils adverses…")

        for path in sorted(files):
            hands = parser.parse_file(path)
            for hand in hands:
                self._process_hand(hand, hero_name)
                total_hands += 1
                for p in hand.players:
                    updated.add(p.name)

        # Sauvegarder tous les profils en cache → DB
        for name, profile in self._session_cache.items():
            self.db.save(profile)

        log.info(
            f"Profils construits : {len(updated)} joueurs "
            f"depuis {total_hands} mains."
        )
        return len(updated)

    def build_from_hh_file(self, path: Path, hero_name: str = "") -> int:
        """Parse un fichier HH et met à jour les profils."""
        try:
            from hh_parser import HandHistoryParser
        except ImportError:
            return 0
        parser = HandHistoryParser(hero_name=hero_name)
        hands  = parser.parse_file(path)
        for hand in hands:
            self._process_hand(hand, hero_name)
        for profile in self._session_cache.values():
            self.db.save(profile)
        return len(hands)

    def _process_hand(self, hand, hero_name: str) -> None:
        """Extrait les stats de chaque adversaire depuis une main parsée."""
        from hh_parser import ParsedHand, ParsedAction

        # Ignorer le héros (on suit les adversaires)
        opp_names = [
            p.name for p in hand.players
            if p.name != hero_name and p.name
        ]

        # Identifier qui a misé/callé préflop (VPIP)
        preflop_aggressors: set = set()   # ont relancé préflop
        preflop_callers:    set = set()   # ont callé préflop
        preflop_raisers:    list = []     # dans l'ordre (pour 3bet)

        for action in hand.actions:
            if action.player == hero_name:
                continue
            if action.player not in opp_names:
                continue

            p = self._get_or_create(action.player)

            if action.street == "preflop":
                if action.action in ("raises", "is all-in", "raises all-in"):
                    preflop_aggressors.add(action.player)
                    preflop_raisers.append(action.player)
                elif action.action == "calls":
                    preflop_callers.add(action.player)
            else:
                # Actions postflop
                if action.action in ("bets",):
                    p.postflop_bets += 1
                elif action.action in ("raises", "raises all-in"):
                    p.postflop_raises += 1
                elif action.action == "calls":
                    p.postflop_calls += 1
                elif action.action == "checks":
                    p.postflop_checks += 1
                elif action.action == "folds":
                    p.postflop_folds += 1

        # Mettre à jour VPIP / PFR pour chaque adversaire
        for name in opp_names:
            p = self._get_or_create(name)
            p.hands_total  += 1
            p.session_hands += 1
            p.last_seen     = time.time()
            p.current_table = hand.table_name

            if name in preflop_aggressors or name in preflop_callers:
                p.hands_vpip += 1
            if name in preflop_aggressors:
                p.hands_pfr += 1

            # 3bet : a relancé alors qu'il y avait déjà une relance devant
            if len(preflop_raisers) >= 2 and name == preflop_raisers[-1]:
                p.hands_3bet += 1

            # Résultat financier
            player_obj = next((pl for pl in hand.players if pl.name == name), None)
            if player_obj:
                if player_obj.result > 0:
                    p.total_won  += player_obj.result
                else:
                    p.total_lost += abs(player_obj.result)
                if player_obj.went_to_sd:
                    p.wtsd += 1
                    if player_obj.result > 0:
                        p.wsd += 1

    # ------------------------------------------------------------------
    # Mise à jour temps réel (depuis action_detector)
    # ------------------------------------------------------------------

    def update_realtime(
        self,
        player_name: str,
        action:      str,       # "FOLD" / "CALL" / "RAISE" / "CHECK" / "BET" / "ALL-IN"
        amount:      float = 0.0,
        stage:       str   = "preflop",
        seat:        int   = 0,
    ) -> None:
        """
        Met à jour le profil d'un adversaire en temps réel.
        À appeler depuis action_detector.py à chaque action détectée.
        """
        p = self._get_or_create(player_name)
        p.last_seen = time.time()

        if stage == "preflop":
            if action in ("RAISE", "ALL-IN"):
                # Incrémentation partielle (sans mains_total, géré ailleurs)
                p.hands_vpip  += 1
                p.hands_pfr   += 1
            elif action == "CALL":
                p.hands_vpip  += 1
        else:
            if action == "BET":
                p.postflop_bets   += 1
            elif action == "RAISE":
                p.postflop_raises += 1
            elif action == "CALL":
                p.postflop_calls  += 1
            elif action == "CHECK":
                p.postflop_checks += 1
            elif action == "FOLD":
                p.postflop_folds  += 1

        # Sauvegarde différée (toutes les 5 actions pour éviter les I/O excessifs)
        if (p.postflop_bets + p.postflop_raises + p.postflop_calls) % 5 == 0:
            self.db.save(p)

    def new_hand_realtime(self, player_names: list[str]) -> None:
        """
        Signale le début d'une nouvelle main pour les joueurs visibles.
        Incrémente hands_total pour chacun.
        """
        for name in player_names:
            p = self._get_or_create(name)
            p.hands_total   += 1
            p.session_hands += 1

    def end_hand_realtime(self) -> None:
        """Sauvegarde tous les profils de la session en cours dans la DB."""
        for profile in self._session_cache.values():
            self.db.save(profile)
        log.debug(f"{len(self._session_cache)} profils sauvegardés.")

    # ------------------------------------------------------------------
    # Récupération
    # ------------------------------------------------------------------

    def get_profile(self, name: str) -> OpponentProfile:
        """Retourne le profil d'un adversaire (cache → DB → nouveau)."""
        if name in self._session_cache:
            return self._session_cache[name]
        db_profile = self.db.load(name)
        if db_profile:
            self._session_cache[name] = db_profile
            return db_profile
        return self._get_or_create(name)

    def get_table_profiles(self, player_names: list[str]) -> dict[str, dict]:
        """
        Retourne les profils de tous les joueurs à la table.
        Format optimisé pour le prompt Claude.
        """
        result = {}
        for name in player_names:
            p = self.get_profile(name)
            result[name] = p.to_claude_context()
        return result

    def get_hud_lines(self, player_names: list[str]) -> list[str]:
        """Retourne les lignes HUD pour l'overlay."""
        lines = []
        for name in player_names:
            p = self.get_profile(name)
            if p.hands_total >= MIN_HANDS_ESTIMATE:
                lines.append(p.hud_line())
        return lines

    def search_players(self, query: str) -> list[OpponentProfile]:
        return self.db.search(query)

    def top_players(self, limit: int = 20) -> list[OpponentProfile]:
        return self.db.top_players(limit)

    def add_note(self, name: str, note: str) -> None:
        self.db.add_note(name, note)

    def get_notes(self, name: str) -> list[str]:
        return self.db.get_notes(name)

    # ------------------------------------------------------------------
    # Import manuel (SharkScope / HUD externes)
    # ------------------------------------------------------------------

    def import_manual(
        self,
        name:    str,
        vpip:    float,
        pfr:     float,
        af:      float,
        hands:   int   = 100,
        notes:   str   = "",
    ) -> OpponentProfile:
        """
        Importe un profil manuellement (ex. depuis SharkScope).
        Rétro-calcule les compteurs depuis les pourcentages.
        """
        p = OpponentProfile(name=name)
        p.hands_total   = hands
        p.hands_vpip    = int(hands * vpip / 100)
        p.hands_pfr     = int(hands * pfr / 100)
        # AF approximé : (bet+raise)/call. On pose bet=raise=AF*n, call=n
        # → AF = 2*AF*n / n → impossible à déduire exactement
        # On utilise une approximation : ratio postflop
        n = max(hands // 3, 1)
        p.postflop_bets   = int(n * af * 0.5)
        p.postflop_raises = int(n * af * 0.5)
        p.postflop_calls  = max(n, 1)

        if notes:
            self.db.add_note(name, notes)
        self.db.save(p)
        self._session_cache[name] = p
        log.info(
            f"Profil manuel importé : {name} "
            f"VPIP={vpip}% PFR={pfr}% AF={af:.1f} ({hands} mains)"
        )
        return p

    # ------------------------------------------------------------------
    # Interne
    # ------------------------------------------------------------------

    def _get_or_create(self, name: str) -> OpponentProfile:
        if name not in self._session_cache:
            # Essayer de charger depuis la DB d'abord
            db_p = self.db.load(name)
            self._session_cache[name] = db_p or OpponentProfile(name=name)
        return self._session_cache[name]


# ---------------------------------------------------------------------------
# Intégration dans le pipeline principal
# ---------------------------------------------------------------------------

def build_opponent_profiles_context(
    builder:      ProfilBuilder,
    player_names: list[str],
    hero_name:    str = "",
) -> dict:
    """
    Construit le dictionnaire opponent_profiles prêt pour claude_client.py.

    Usage dans main.py :
        profiles = build_opponent_profiles_context(
            builder, state.player_names, hero_name="MonPseudo"
        )
        advice = claude.get_advice(state, equity, opponent_profiles=profiles)
    """
    names = [n for n in player_names if n and n != hero_name]
    return builder.get_table_profiles(names)


# ---------------------------------------------------------------------------
# Affichage console
# ---------------------------------------------------------------------------

def print_profile(p: OpponentProfile) -> None:
    bar_v = "█" * int(p.vpip / 5)
    bar_p = "█" * int(p.pfr  / 5)
    print(f"\n{'─'*55}")
    print(f"  {p.name}  [{p.tendency}]  {p.reliability_label}")
    print(f"{'─'*55}")
    print(f"  VPIP  {p.vpip:5.1f}%  {bar_v}")
    print(f"  PFR   {p.pfr:5.1f}%  {bar_p}")
    print(f"  AF    {p.af:5.2f}")
    print(f"  3Bet  {p.three_bet_pct:5.1f}%")
    print(f"  CBet  {p.cbet_pct:5.1f}%   Fold CBet: {p.fold_to_cbet_pct:.1f}%")
    print(f"  WTSD  {p.wtsd_pct:5.1f}%   W$SD:      {p.wsd_pct:.1f}%")
    print(f"  BB/100: {p.bb_per_100:+.1f}")
    print(f"\n  → {p.strategic_advice}")
    print(f"{'─'*55}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Constructeur de profils adverses VPIP/PFR/AF"
    )
    parser.add_argument("--build",   type=str, metavar="DOSSIER",
                        help="Construire profils depuis un dossier HH")
    parser.add_argument("--hero",    type=str, default="",
                        help="Pseudo du héros (ignoré dans les profils)")
    parser.add_argument("--search",  type=str, metavar="NOM",
                        help="Chercher un joueur dans la base")
    parser.add_argument("--top",     type=int, default=0,
                        help="Afficher les N adversaires les plus fréquents")
    parser.add_argument("--profile", type=str, metavar="NOM",
                        help="Afficher le profil détaillé d'un joueur")
    parser.add_argument("--import",  dest="import_manual", nargs=4,
                        metavar=("NOM","VPIP","PFR","AF"),
                        help="Importer manuellement : --import Villain 35 18 2.1")
    parser.add_argument("--note",    nargs=2, metavar=("NOM","NOTE"),
                        help="Ajouter une note sur un joueur")
    args = parser.parse_args()

    db      = ProfileDatabase()
    builder = ProfilBuilder(db)

    if args.build:
        folder = Path(args.build)
        if not folder.exists():
            print(f"ERREUR : dossier introuvable : {folder}")
        else:
            n = builder.build_from_hh_folder(folder, hero_name=args.hero)
            print(f"\n✓ {n} profils construits depuis {folder}")

    elif args.search:
        profiles = builder.search_players(args.search)
        if not profiles:
            print(f"Aucun joueur trouvé pour '{args.search}'")
        for p in profiles:
            print_profile(p)

    elif args.top:
        profiles = builder.top_players(limit=args.top)
        print(f"\nTop {args.top} adversaires :")
        print(f"{'Nom':<20} {'Mains':>6} {'VPIP':>6} {'PFR':>6} "
              f"{'AF':>5} {'Tendance':<16}")
        print("─" * 65)
        for p in profiles:
            print(f"{p.name:<20} {p.hands_total:>6} {p.vpip:>5.1f}% "
                  f"{p.pfr:>5.1f}% {p.af:>5.2f} {p.tendency:<16}")

    elif args.profile:
        p = builder.get_profile(args.profile)
        print_profile(p)
        notes = builder.get_notes(args.profile)
        if notes:
            print("\n  Notes :")
            for note in notes:
                print(f"    • {note}")

    elif args.import_manual:
        name, vpip, pfr, af = args.import_manual
        p = builder.import_manual(
            name=name, vpip=float(vpip),
            pfr=float(pfr), af=float(af)
        )
        print_profile(p)

    elif args.note:
        name, note = args.note
        builder.add_note(name, note)
        print(f"✓ Note ajoutée pour {name} : {note}")

    else:
        # Afficher un résumé global
        all_profiles = db.load_all()
        print(f"\nBase de profils : {len(all_profiles)} joueurs")
        if all_profiles:
            reliable = sum(1 for p in all_profiles if p.is_reliable)
            print(f"  Fiables (≥{MIN_HANDS_RELIABLE} mains) : {reliable}")
            print(f"\nCommandes disponibles :")
            print("  --build DOSSIER    Construire depuis HH")
            print("  --top 20           Meilleurs adversaires")
            print("  --search NOM       Chercher un joueur")
            print("  --profile NOM      Profil détaillé")
