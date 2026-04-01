"""
hh_parser.py — Parser de Hand History PokerStars avec import automatique dans tracker.py
Dépendances : watchdog (pip install watchdog) + tracker.py

PokerStars sauvegarde automatiquement chaque main jouée dans :
  Windows : C:\\Users\\<user>\\AppData\\Local\\PokerStars\\HandHistory\\<pseudo>\\
  macOS   : ~/Library/Application Support/PokerStars/HandHistory/<pseudo>/

Ce module :
  1. Trouve automatiquement le dossier HandHistory de PokerStars
  2. Surveille les nouveaux fichiers .txt en temps réel (watchdog)
  3. Parse chaque main : cartes, board, pot, résultat, adversaires
  4. Injecte dans tracker.py (HandRecord + session auto)

Usage :
    python hh_parser.py                          # surveillance continue
    python hh_parser.py --parse fichier.txt      # parser un fichier existant
    python hh_parser.py --import-all             # importer tout l'historique
    python hh_parser.py --folder C:\\...\\HH\\Pseudo  # dossier spécifique
"""

import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chemins PokerStars par défaut
# ---------------------------------------------------------------------------

HH_PATHS_WIN = [
    Path.home() / "AppData" / "Local" / "PokerStars" / "HandHistory",
    Path.home() / "AppData" / "Local" / "PokerStars.EU" / "HandHistory",
    Path("C:/") / "Users" / Path.home().name / "AppData" / "Local" / "PokerStars" / "HandHistory",
]

HH_PATHS_MAC = [
    Path.home() / "Library" / "Application Support" / "PokerStars" / "HandHistory",
]

HH_PATHS_LINUX = [
    Path.home() / ".wine" / "drive_c" / "users" / Path.home().name /
    "Local Settings" / "Application Data" / "PokerStars" / "HandHistory",
]

# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------

RANKS = {"2","3","4","5","6","7","8","9","T","J","Q","K","A"}
SUITS = {"s","h","d","c"}

@dataclass
class ParsedPlayer:
    name:         str
    seat:         int
    stack:        float
    cards:        list[str] = field(default_factory=list)
    action_total: float     = 0.0
    result:       float     = 0.0   # gain (+) ou perte (-) net
    position:     str       = ""
    went_to_sd:   bool      = False  # est allé au showdown

@dataclass
class ParsedAction:
    player:  str
    street:  str
    action:  str   # fold / call / raise / check / bet / all-in
    amount:  float = 0.0

@dataclass
class ParsedHand:
    """Représentation complète d'une main parsée."""
    hand_id:      str
    table_name:   str
    game_type:    str          # "No Limit Hold'em"
    stakes:       str          # ex. "$0.01/$0.02"
    datetime_str: str
    num_players:  int
    dealer_seat:  int
    players:      list[ParsedPlayer] = field(default_factory=list)
    board:        list[str]    = field(default_factory=list)
    pot_total:    float        = 0.0
    rake:         float        = 0.0
    hero_name:    str          = ""
    hero_cards:   list[str]    = field(default_factory=list)
    hero_result:  float        = 0.0
    hero_seat:    int          = 0
    hero_position:str          = ""
    actions:      list[ParsedAction] = field(default_factory=list)
    winner:       str          = ""
    is_tournament:bool         = False
    tournament_id:str          = ""
    raw_text:     str          = ""

    @property
    def stage_final(self) -> str:
        n = len(self.board)
        if n == 0:   return "preflop"
        if n == 3:   return "flop"
        if n == 4:   return "turn"
        return "river"

    @property
    def hero_hand_class(self) -> str:
        """Évaluation rapide de la main du héros (sans treys pour éviter dépendance)."""
        if not self.hero_cards or not self.board:
            return "Préflop"
        try:
            from engine import evaluate_hand
            _, cls = evaluate_hand(self.hero_cards, self.board)
            return cls
        except Exception:
            return "Inconnue"


# ---------------------------------------------------------------------------
# Parser principal
# ---------------------------------------------------------------------------

class HandHistoryParser:
    """
    Parse les fichiers Hand History PokerStars (format texte standard).

    Exemple de format PokerStars :
        PokerStars Hand #123456789: Hold'em No Limit ($0.01/$0.02 USD) - 2024/01/15 20:30:00 ET
        Table 'Altair II' 9-max Seat #5 is the button
        Seat 1: Player1 ($2.50 in chips)
        ...
        *** HOLE CARDS ***
        Dealt to Hero [Ks 7h]
        Player2: folds
        Hero: raises $0.06 to $0.08
        ...
        *** FLOP *** [Kd 7d 2c]
        ...
        *** SUMMARY ***
        Total pot $0.28 | Rake $0.02
        Board [Kd 7d 2c As 3h]
        Seat 3: Hero showed [Ks 7h] and won ($0.26)
    """

    # Regex principaux
    _RE_HAND_ID   = re.compile(r"Hand #(\d+)")
    _RE_TOURN_ID  = re.compile(r"Tournament #(\d+)")
    _RE_STAKES    = re.compile(r"\(\$?([\d.]+)/\$?([\d.]+)")
    _RE_DATETIME  = re.compile(r"(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})")
    _RE_TABLE     = re.compile(r"Table '([^']+)'")
    _RE_MAXSEATS  = re.compile(r"(\d+)-max")
    _RE_BUTTON    = re.compile(r"Seat #(\d+) is the button")
    _RE_SEAT      = re.compile(r"Seat (\d+): (.+?) \(\$?([\d,]+\.?\d*) in chips\)")
    _RE_DEALT     = re.compile(r"Dealt to (.+?) \[([2-9TJQKAshdc ]+)\]")
    _RE_CARD      = re.compile(r"([2-9TJQKA][shdc])")
    _RE_ACTION    = re.compile(
        r"^(.+?): (folds|calls|raises|checks|bets|is all-in|calls all-in|raises all-in)"
        r"(?:\s+\$?([\d,]+\.?\d*))?",
        re.IGNORECASE
    )
    _RE_BOARD     = re.compile(r"\*\*\* (?:FLOP|TURN|RIVER) \*\*\* [\[\(]([^\]\)]+)[\]\)]")
    _RE_POT       = re.compile(r"Total pot \$?([\d,]+\.?\d*)")
    _RE_RAKE      = re.compile(r"Rake \$?([\d,]+\.?\d*)")
    _RE_WINNER    = re.compile(r"(.+?) (?:collected|won) \$?([\d,]+\.?\d*)")
    _RE_SHOWED    = re.compile(r"Seat \d+: (.+?) showed \[([^\]]+)\] and (won|lost)")
    _RE_MUCKED    = re.compile(r"Seat \d+: (.+?) mucked \[([^\]]+)\]")
    _RE_RESULT    = re.compile(r"Seat \d+: (.+?) (?:showed.+and )?(won|collected) \(\$?([\d,]+\.?\d*)\)")

    def __init__(self, hero_name: str = ""):
        """
        hero_name : pseudo du joueur dont on suit les résultats.
                    Si vide, tente de le détecter automatiquement.
        """
        self.hero_name = hero_name

    def parse_file(self, path: Path) -> list[ParsedHand]:
        """Parse toutes les mains d'un fichier .txt PokerStars."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            log.error(f"Impossible de lire {path} : {e}")
            return []

        # Chaque main est séparée par une ligne vide double
        raw_hands = re.split(r"\n\n+(?=PokerStars)", content.strip())
        hands = []
        for raw in raw_hands:
            raw = raw.strip()
            if not raw or "PokerStars Hand" not in raw:
                continue
            try:
                hand = self._parse_hand(raw)
                if hand:
                    hands.append(hand)
            except Exception as e:
                log.debug(f"Erreur parsing main : {e}")
        log.info(f"{path.name} : {len(hands)} mains parsées")
        return hands

    def _parse_hand(self, raw: str) -> Optional[ParsedHand]:
        """Parse une seule main."""
        lines = raw.splitlines()
        if not lines:
            return None

        hand = ParsedHand(
            hand_id      = "",
            table_name   = "",
            game_type    = "No Limit Hold'em",
            stakes       = "",
            datetime_str = "",
            num_players  = 0,
            dealer_seat  = 0,
            raw_text     = raw,
        )

        # ── Ligne 1 : identifiants ────────────────────────────────────────
        header = lines[0]
        m = self._RE_HAND_ID.search(header)
        if m:
            hand.hand_id = m.group(1)

        m = self._RE_TOURN_ID.search(header)
        if m:
            hand.tournament_id = m.group(1)
            hand.is_tournament = True

        m = self._RE_STAKES.search(header)
        if m:
            hand.stakes = f"${m.group(1)}/${m.group(2)}"

        m = self._RE_DATETIME.search(header)
        if m:
            hand.datetime_str = m.group(1)

        # ── Ligne 2 : table ───────────────────────────────────────────────
        if len(lines) > 1:
            m = self._RE_TABLE.search(lines[1])
            if m:
                hand.table_name = m.group(1)
            m = self._RE_BUTTON.search(lines[1])
            if m:
                hand.dealer_seat = int(m.group(1))
            m = self._RE_MAXSEATS.search(lines[1])
            if m:
                hand.num_players = int(m.group(1))

        # ── Sièges ────────────────────────────────────────────────────────
        players: dict[str, ParsedPlayer] = {}
        for line in lines:
            m = self._RE_SEAT.match(line)
            if m:
                seat  = int(m.group(1))
                name  = m.group(2).strip()
                stack = float(m.group(3).replace(",", ""))
                p = ParsedPlayer(name=name, seat=seat, stack=stack)
                players[name] = p

        hand.num_players = len(players) if players else hand.num_players

        # ── Héros détecté ─────────────────────────────────────────────────
        hero_name = self.hero_name
        for line in lines:
            m = self._RE_DEALT.search(line)
            if m:
                if not hero_name:
                    hero_name = m.group(1).strip()
                    self.hero_name = hero_name
                cards_raw = m.group(2).strip()
                hand.hero_cards = self._RE_CARD.findall(cards_raw)
                hand.hero_name  = hero_name
                if hero_name in players:
                    players[hero_name].cards = hand.hero_cards
                    hand.hero_seat = players[hero_name].seat
                break

        # ── Position du héros ─────────────────────────────────────────────
        if hand.hero_seat and hand.dealer_seat and hand.num_players:
            hand.hero_position = self._compute_position(
                hand.hero_seat, hand.dealer_seat, hand.num_players, players
            )

        # ── Actions ───────────────────────────────────────────────────────
        current_street = "preflop"
        for line in lines:
            if "*** HOLE CARDS ***" in line:
                current_street = "preflop"
            elif "*** FLOP ***" in line:
                current_street = "flop"
                board_cards = self._RE_CARD.findall(line)
                hand.board.extend(board_cards[:3])
            elif "*** TURN ***" in line:
                current_street = "turn"
                board_cards = self._RE_CARD.findall(line)
                if len(board_cards) >= 4:
                    if len(hand.board) == 3:
                        hand.board.append(board_cards[3])
                else:
                    new_cards = [c for c in board_cards if c not in hand.board]
                    if new_cards:
                        hand.board.append(new_cards[0])
            elif "*** RIVER ***" in line:
                current_street = "river"
                board_cards = self._RE_CARD.findall(line)
                if len(board_cards) >= 5:
                    if len(hand.board) == 4:
                        hand.board.append(board_cards[4])
                else:
                    new_cards = [c for c in board_cards if c not in hand.board]
                    if new_cards:
                        hand.board.append(new_cards[0])
            elif "*** SUMMARY ***" in line:
                break
            else:
                m = self._RE_ACTION.match(line)
                if m:
                    pname  = m.group(1).strip()
                    act    = m.group(2).lower()
                    amount = float(m.group(3).replace(",","")) if m.group(3) else 0.0
                    hand.actions.append(ParsedAction(pname, current_street, act, amount))
                    if pname in players:
                        players[pname].action_total += amount

        # ── Summary ───────────────────────────────────────────────────────
        in_summary = False
        for line in lines:
            if "*** SUMMARY ***" in line:
                in_summary = True
                continue
            if not in_summary:
                continue

            # Pot total
            m = self._RE_POT.search(line)
            if m:
                hand.pot_total = float(m.group(1).replace(",",""))

            m = self._RE_RAKE.search(line)
            if m:
                hand.rake = float(m.group(1).replace(",",""))

            # Gagnant
            m = self._RE_WINNER.search(line)
            if m and not hand.winner:
                hand.winner = m.group(1).strip()

            # Résultat du héros
            m = self._RE_RESULT.search(line)
            if m:
                pname  = m.group(1).strip()
                gained = float(m.group(3).replace(",",""))
                if pname in players:
                    players[pname].result = gained
                if pname == hero_name:
                    hand.hero_result = gained

            # Cartes montrées
            m = self._RE_SHOWED.search(line)
            if m:
                pname = m.group(1).strip()
                cards = self._RE_CARD.findall(m.group(2))
                if pname in players:
                    players[pname].cards = cards
                    players[pname].went_to_sd = True

        # Calculer les gains/pertes nets (résultat - mise totale)
        for p in players.values():
            if p.result > 0:
                p.result = p.result - p.action_total  # gain net

        hand.players = list(players.values())

        # Résultat net du héros
        if hero_name in players:
            hand.hero_result = players[hero_name].result

        return hand if hand.hand_id else None

    def _compute_position(
        self,
        hero_seat:   int,
        dealer_seat: int,
        num_players: int,
        players:     dict,
    ) -> str:
        """Calcule la position du héros par rapport au bouton."""
        seats = sorted(players.keys(), key=lambda n: players[n].seat)
        n = len(seats)
        if n < 2:
            return ""

        # Trouver l'index du dealer et du héros dans la liste de sièges
        hero_name_by_seat = {p.seat: p.name for p in players.values()}
        dealer_name = hero_name_by_seat.get(dealer_seat, "")

        if dealer_name not in seats:
            return ""

        dealer_idx = seats.index(dealer_name)
        hero_name  = hero_name_by_seat.get(hero_seat, "")
        if hero_name not in seats:
            return ""

        hero_idx = seats.index(hero_name)
        # Distance depuis le dealer (sens horaire)
        dist = (hero_idx - dealer_idx) % n

        pos_maps = {
            2:  {0:"BTN", 1:"BB"},
            3:  {0:"BTN", 1:"SB", 2:"BB"},
            4:  {0:"BTN", 1:"SB", 2:"BB", 3:"UTG"},
            5:  {0:"BTN", 1:"SB", 2:"BB", 3:"UTG", 4:"CO"},
            6:  {0:"BTN", 1:"SB", 2:"BB", 3:"UTG", 4:"MP",  5:"CO"},
            7:  {0:"BTN", 1:"SB", 2:"BB", 3:"UTG", 4:"MP",  5:"HJ", 6:"CO"},
            8:  {0:"BTN", 1:"SB", 2:"BB", 3:"UTG", 4:"UTG+1",5:"MP",6:"HJ",7:"CO"},
            9:  {0:"BTN", 1:"SB", 2:"BB", 3:"UTG", 4:"UTG+1",5:"MP",6:"MP+1",7:"HJ",8:"CO"},
            10: {0:"BTN", 1:"SB", 2:"BB", 3:"UTG", 4:"UTG+1",5:"UTG+2",6:"MP",7:"MP+1",8:"HJ",9:"CO"},
        }
        mapping = pos_maps.get(n, pos_maps.get(9, {}))
        return mapping.get(dist, "")

    @staticmethod
    def _normalize_card(c: str) -> Optional[str]:
        if len(c) == 2 and c[0].upper() in RANKS and c[1].lower() in SUITS:
            return c[0].upper() + c[1].lower()
        return None


# ---------------------------------------------------------------------------
# Importateur dans tracker.py
# ---------------------------------------------------------------------------

class HandHistoryImporter:
    """
    Importe les mains parsées dans la base de données tracker.py.
    Crée automatiquement les sessions par date de jeu.
    """

    def __init__(self, tracker, hero_name: str = ""):
        self.tracker   = tracker
        self.parser    = HandHistoryParser(hero_name=hero_name)
        self._sessions: dict[str, int] = {}   # date → session_id
        self._imported: set[str] = set()       # hand_ids déjà importés
        self._load_imported_hands()

    def _load_imported_hands(self) -> None:
        """Charge les hand_ids déjà en base pour éviter les doublons."""
        try:
            with self.tracker._conn() as conn:
                rows = conn.execute(
                    "SELECT notes FROM hands WHERE notes LIKE 'HH#%'"
                ).fetchall()
            for row in rows:
                hid = row["notes"].split("#")[1].split(" ")[0]
                self._imported.add(hid)
            log.info(f"{len(self._imported)} mains déjà importées en base.")
        except Exception as e:
            log.debug(f"Chargement mains importées : {e}")

    def import_file(self, path: Path) -> int:
        """Importe toutes les mains d'un fichier. Retourne le nombre importées."""
        hands = self.parser.parse_file(path)
        count = 0
        for hand in hands:
            if hand.hand_id in self._imported:
                continue
            try:
                self._import_hand(hand)
                self._imported.add(hand.hand_id)
                count += 1
            except Exception as e:
                log.error(f"Erreur import main #{hand.hand_id} : {e}")
        if count:
            log.info(f"{count} nouvelles mains importées depuis {path.name}")
        return count

    def import_folder(self, folder: Path, recursive: bool = True) -> int:
        """Importe toutes les mains d'un dossier."""
        total = 0
        pattern = "**/*.txt" if recursive else "*.txt"
        files   = list(folder.glob(pattern))
        log.info(f"Importation de {len(files)} fichiers depuis {folder}")
        for path in sorted(files):
            total += self.import_file(path)
        log.info(f"Import terminé : {total} mains importées au total.")
        return total

    def _import_hand(self, hand: ParsedHand) -> None:
        """Importe une seule main dans tracker.py."""
        from tracker import HandRecord

        # Trouver ou créer la session pour ce jour
        session_id = self._get_or_create_session(hand)

        # Construire le HandRecord
        followed = self._check_followed_advice(hand)

        record = HandRecord(
            session_id         = session_id,
            stage_final        = hand.stage_final,
            player_cards       = hand.hero_cards,
            board_cards        = hand.board,
            num_opponents      = max(0, hand.num_players - 1),
            pot_final          = hand.pot_total,
            hand_class         = hand.hero_hand_class,
            win_probability    = 0.0,          # calculé offline
            recommended_action = "",            # pas de conseil en offline
            action_taken       = self._hero_final_action(hand),
            followed_advice    = followed,
            result             = hand.hero_result,
            ev_estimate        = 0.0,
            ev_realized        = hand.hero_result,
            notes              = f"HH#{hand.hand_id} {hand.table_name} {hand.stakes}",
            timestamp          = self._parse_timestamp(hand.datetime_str),
        )
        self.tracker.record_hand(record)

    def _get_or_create_session(self, hand: ParsedHand) -> int:
        """Retourne l'ID de session du jour, ou en crée une nouvelle."""
        date_key = hand.datetime_str[:10] if hand.datetime_str else "unknown"

        if date_key in self._sessions:
            return self._sessions[date_key]

        # Créer une session pour ce jour
        session_id = self.tracker.start_session(
            buy_in      = 0.0,
            game_type   = "Tournoi" if hand.is_tournament else "Cash Game",
            num_players = hand.num_players,
            notes       = f"Import HH {date_key} — {hand.table_name} {hand.stakes}",
        )
        self._sessions[date_key] = session_id
        log.info(f"Session #{session_id} créée pour {date_key}")
        return session_id

    def _hero_final_action(self, hand: ParsedHand) -> str:
        """Détermine la dernière action du héros dans la main."""
        hero_actions = [
            a for a in hand.actions
            if a.player == hand.hero_name
        ]
        if not hero_actions:
            return "UNKNOWN"
        last = hero_actions[-1]
        action_map = {
            "folds":   "FOLD",  "calls":    "CALL",
            "raises":  "RAISE", "checks":   "CHECK",
            "bets":    "BET",   "is all-in":"ALL-IN",
        }
        return action_map.get(last.action, last.action.upper())

    @staticmethod
    def _check_followed_advice(hand: ParsedHand) -> bool:
        """En import offline, on ne peut pas savoir si le conseil a été suivi."""
        return False   # neutre par défaut

    @staticmethod
    def _parse_timestamp(dt_str: str) -> float:
        """Convertit '2024/01/15 20:30:00' en timestamp Unix."""
        try:
            dt = datetime.strptime(dt_str, "%Y/%m/%d %H:%M:%S")
            return dt.timestamp()
        except Exception:
            return time.time()


# ---------------------------------------------------------------------------
# Détection automatique du dossier HandHistory
# ---------------------------------------------------------------------------

def find_hh_folder() -> Optional[Path]:
    """Trouve automatiquement le dossier HandHistory de PokerStars."""
    import platform
    system = platform.system()

    if system == "Windows":
        candidates = HH_PATHS_WIN
    elif system == "Darwin":
        candidates = HH_PATHS_MAC
    else:
        candidates = HH_PATHS_LINUX

    for path in candidates:
        if path.exists():
            log.info(f"Dossier HandHistory trouvé : {path}")
            return path

    log.warning("Dossier HandHistory non trouvé automatiquement.")
    log.warning("Utilisez --folder pour spécifier le chemin manuellement.")
    return None


# ---------------------------------------------------------------------------
# Surveillance en temps réel (watchdog)
# ---------------------------------------------------------------------------

class HHWatcher:
    """
    Surveille le dossier HandHistory en temps réel.
    Chaque nouveau fichier .txt ou modification → import automatique.

    Usage :
        watcher = HHWatcher(tracker, hh_folder, hero_name="MonPseudo")
        watcher.start()     # non-bloquant
        watcher.stop()      # arrêt propre
    """

    def __init__(
        self,
        tracker,
        hh_folder:  Path,
        hero_name:  str  = "",
        interval:   float = 2.0,   # secondes entre chaque vérification (fallback)
    ):
        self.tracker    = tracker
        self.hh_folder  = hh_folder
        self.interval   = interval
        self.importer   = HandHistoryImporter(tracker, hero_name=hero_name)
        self._running   = False
        self._thread    = None
        self._seen_files: dict[Path, float] = {}   # path → mtime

    def start(self) -> None:
        """Lance la surveillance dans un thread daemon."""
        import threading
        self._running = True
        self._thread  = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        log.info(f"Surveillance HH démarrée : {self.hh_folder}")

        # Essayer watchdog d'abord, fallback sur polling
        self._try_watchdog()

    def stop(self) -> None:
        """Arrête la surveillance."""
        self._running = False
        if hasattr(self, "_observer") and self._observer:
            try:
                self._observer.stop()
                self._observer.join()
            except Exception:
                pass
        log.info("Surveillance HH arrêtée.")

    def _try_watchdog(self) -> None:
        """Tente de lancer watchdog pour une surveillance événementielle."""
        try:
            from watchdog.observers import Observer
            from watchdog.events    import FileSystemEventHandler

            class _Handler(FileSystemEventHandler):
                def __init__(self_, importer):
                    self_.importer = importer

                def on_created(self_, event):
                    if not event.is_directory and event.src_path.endswith(".txt"):
                        path = Path(event.src_path)
                        time.sleep(0.5)   # attendre que PS finisse d'écrire
                        n = self_.importer.import_file(path)
                        if n:
                            log.info(f"Watchdog : {n} mains importées depuis {path.name}")

                def on_modified(self_, event):
                    self_.on_created(event)

            self._observer = Observer()
            self._observer.schedule(
                _Handler(self.importer),
                str(self.hh_folder),
                recursive=True,
            )
            self._observer.start()
            log.info("Watchdog événementiel activé (réactif, pas de polling).")

        except ImportError:
            log.info("watchdog non installé — utilisation du polling toutes les 2s.")
            log.info("Pour une surveillance plus réactive : pip install watchdog")

    def _watch_loop(self) -> None:
        """Fallback polling : vérifie les fichiers modifiés toutes les N secondes."""
        while self._running:
            try:
                for path in self.hh_folder.rglob("*.txt"):
                    mtime = path.stat().st_mtime
                    if path not in self._seen_files or self._seen_files[path] < mtime:
                        self._seen_files[path] = mtime
                        n = self.importer.import_file(path)
                        if n:
                            log.info(f"Polling : {n} mains importées depuis {path.name}")
            except Exception as e:
                log.error(f"Erreur boucle de surveillance : {e}")
            time.sleep(self.interval)


# ---------------------------------------------------------------------------
# Intégration dans main.py
# ---------------------------------------------------------------------------

def start_hh_watcher(tracker, hero_name: str = "", hh_folder: Optional[Path] = None) -> Optional[HHWatcher]:
    """
    Raccourci pour démarrer la surveillance depuis main.py.

    Usage dans main.py :
        from hh_parser import start_hh_watcher
        hh_watcher = start_hh_watcher(self.tracker, hero_name="MonPseudo")
        # La surveillance tourne en arrière-plan automatiquement
    """
    folder = hh_folder or find_hh_folder()
    if not folder:
        log.warning("Surveillance HH désactivée — dossier non trouvé.")
        return None

    watcher = HHWatcher(tracker, folder, hero_name=hero_name)
    watcher.start()
    return watcher


# ---------------------------------------------------------------------------
# Affichage d'une main parsée (debug)
# ---------------------------------------------------------------------------

def print_hand_summary(hand: ParsedHand) -> None:
    """Affiche un résumé lisible d'une main parsée."""
    print(f"\n{'─'*55}")
    print(f"  Main #{hand.hand_id} — {hand.datetime_str}")
    print(f"  Table : {hand.table_name} | Stakes : {hand.stakes}")
    print(f"  Héros : {hand.hero_name} ({hand.hero_position}) "
          f"| Cartes : {hand.hero_cards}")
    print(f"  Board : {hand.board} | Stage : {hand.stage_final}")
    print(f"  Pot   : {hand.pot_total}$ | Rake : {hand.rake}$")
    print(f"  Résultat héros : {hand.hero_result:+.2f}$")
    print(f"  Gagnant : {hand.winner}")
    if hand.actions:
        print(f"  Actions ({len(hand.actions)}) :")
        for a in hand.actions[-5:]:
            amt = f" {a.amount:.2f}$" if a.amount else ""
            print(f"    [{a.street}] {a.player}: {a.action}{amt}")
    print(f"{'─'*55}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Parser Hand History PokerStars → tracker.py"
    )
    parser.add_argument("--parse",      type=str, metavar="FICHIER",
                        help="Parser un fichier .txt spécifique")
    parser.add_argument("--import-all", action="store_true",
                        help="Importer tout l'historique dans tracker.py")
    parser.add_argument("--watch",      action="store_true",
                        help="Surveillance continue en temps réel")
    parser.add_argument("--folder",     type=str, default=None,
                        help="Dossier HandHistory (détecté auto si absent)")
    parser.add_argument("--hero",       type=str, default="",
                        help="Pseudo du héros (détecté auto si absent)")
    parser.add_argument("--debug",      action="store_true",
                        help="Afficher le détail de chaque main parsée")
    args = parser.parse_args()

    # ── Parse seul (sans import tracker) ─────────────────────────────────
    if args.parse:
        path   = Path(args.parse)
        parser_obj = HandHistoryParser(hero_name=args.hero)
        hands  = parser_obj.parse_file(path)
        print(f"\n{len(hands)} mains parsées dans {path.name}")
        for hand in hands:
            print_hand_summary(hand)
        sys.exit(0)

    # ── Import dans tracker ───────────────────────────────────────────────
    try:
        from tracker import PokerTracker
        tracker_obj = PokerTracker()
    except ImportError:
        print("ERREUR : tracker.py introuvable dans le dossier.")
        sys.exit(1)

    folder = Path(args.folder) if args.folder else find_hh_folder()
    if not folder:
        print("\nDossier HandHistory non trouvé.")
        print("Spécifiez-le avec --folder :")
        print("  python hh_parser.py --watch --folder \"C:\\...\\HandHistory\\MonPseudo\"")
        sys.exit(1)

    importer = HandHistoryImporter(tracker_obj, hero_name=args.hero)

    if args.import_all or not args.watch:
        print(f"\nImportation depuis : {folder}")
        total = importer.import_folder(folder)
        print(f"\n✓ {total} mains importées dans poker_stats.db")

    if args.watch:
        print(f"\nSurveillance en temps réel : {folder}")
        print("(Ctrl+C pour arrêter)\n")
        watcher = HHWatcher(tracker_obj, folder, hero_name=args.hero)
        watcher.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            watcher.stop()
            print("\nSurveillance arrêtée.")
