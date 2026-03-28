"""
main.py — Point d'entrée principal de l'assistant poker temps réel
Assemble : capture.py + engine.py + claude_client.py + overlay.py
           + tracker.py (enregistrement BD) + stats_viewer.py (dashboard)

Usage :
    python main.py
    python main.py --interval 2.0 --simulations 3000 --debug
    python main.py --demo          (mode démo sans PokerStars)
    python main.py --stats         (ouvrir le dashboard stats)
    python main.py --buy-in 50     (définir le buy-in de la session)
    python main.py --game Tournoi  (type de jeu)
"""

import argparse
import logging
import signal
import sys
import time
import threading
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration du logger global
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("poker_hud.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Imports des modules locaux
# ---------------------------------------------------------------------------
try:
    from capture       import PokerExtractor, GameState, run_capture_loop
    from engine        import PokerEngine, EquityResult
    from claude_client import CachedClaudeClient, PokerAdvice
    from overlay       import PokerHUD, HUDThread, DisplayData, ACTION_COLORS
    from tracker       import PokerTracker, HandRecord
    from stats_viewer  import StatsViewer
except ImportError as e:
    log.critical(f"Module manquant : {e}")
    log.critical("Vérifiez que tous les modules .py sont dans le même dossier.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
VERSION = "1.0.0"
BANNER  = f"""
╔══════════════════════════════════════╗
║      ♠ POKER HUD ASSISTANT v{VERSION}    ║
║      Propulsé par Claude Sonnet      ║
╚══════════════════════════════════════╝
"""


# ---------------------------------------------------------------------------
# Gestionnaire d'arrêt propre
# ---------------------------------------------------------------------------
_shutdown_event = threading.Event()

def _handle_signal(sig, frame):
    log.info(f"Signal {sig} reçu — arrêt en cours…")
    _shutdown_event.set()

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

class PokerAssistant:
    """
    Orchestre les 6 modules en un pipeline cohérent :

    PokerStars (écran)
        → PokerExtractor     (capture + OCR)
        → PokerEngine        (calcul equity)
        → CachedClaudeClient (conseil IA)
        → HUDThread          (affichage overlay)
        → PokerTracker       (enregistrement BD SQLite)
    """

    def __init__(
        self,
        capture_interval:  float = 1.5,
        simulations:       int   = 2000,
        hud_x:             int   = 40,
        hud_y:             int   = 200,
        debug_capture:     bool  = False,
        api_key:           str   = None,
        buy_in:            float = 0.0,
        game_type:         str   = "Tournoi",
        num_players:       int   = 8,
    ):
        self.capture_interval = capture_interval
        self.debug_capture    = debug_capture
        self.buy_in           = buy_in
        self.game_type        = game_type
        self.num_players      = num_players

        log.info("Initialisation des modules…")

        # Module 1 — Capture
        self.extractor = PokerExtractor(debug=debug_capture)
        log.info("  ✓ capture.py        — PokerExtractor prêt")

        # Module 2 — Engine
        self.engine = PokerEngine(simulations=simulations)
        log.info(f"  ✓ engine.py         — PokerEngine prêt ({simulations} simulations MC)")

        # Module 3 — Claude
        self.claude = CachedClaudeClient(api_key=api_key)
        log.info("  ✓ claude_client.py  — CachedClaudeClient prêt")

        # Module 4 — HUD
        self.hud_thread = HUDThread(x=hud_x, y=hud_y)
        log.info("  ✓ overlay.py        — HUDThread prêt")

        # Module 5 — Tracker
        self.tracker    = PokerTracker()
        self.session_id = self.tracker.start_session(
            buy_in      = buy_in,
            game_type   = game_type,
            num_players = num_players,
        )
        log.info(f"  ✓ tracker.py        — Session #{self.session_id} démarrée (buy-in={buy_in}$)")

        # État de la main en cours
        self._current_hand_id:    int   = 0
        self._current_hand_cards: list  = []
        self._hand_result:        float = 0.0

        # Statistiques de session en mémoire
        self._stats = {
            "hands":           0,
            "states_captured": 0,
            "api_calls":       0,
            "errors":          0,
            "start_time":      time.time(),
        }

    # ------------------------------------------------------------------
    # Démarrage
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Lance le pipeline complet (bloquant jusqu'à fermeture du HUD)."""
        print(BANNER)

        # Démarrer le HUD dans son thread
        self.hud_thread.start()
        if not self.hud_thread.wait_ready(timeout=5.0):
            log.error("Le HUD n'a pas démarré dans les 5s — abandon.")
            return

        log.info("HUD démarré. En attente de la table PokerStars…")
        log.info("Raccourcis HUD  : Esc=quitter  Ctrl+H=minimiser  Ctrl+O=opacité")
        log.info("Console         : tapez 'n' + Entrée = nouvelle main")
        log.info("                  tapez 'end' + Entrée = clôturer la session")
        log.info("                  tapez 'stats' + Entrée = ouvrir le dashboard")

        self._show_waiting()

        # Thread de capture
        worker = threading.Thread(target=self._capture_loop, daemon=True)
        worker.start()

        # Thread d'écoute console (commandes manuelles)
        console = threading.Thread(target=self._console_listener, daemon=True)
        console.start()

        # Attendre la fermeture du HUD
        self.hud_thread.join()
        _shutdown_event.set()

        # Clôturer la session automatiquement à la fermeture
        self._close_session()
        self._print_session_stats()

    # ------------------------------------------------------------------
    # Boucle de capture (thread worker)
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Boucle principale : capture → engine → Claude → HUD."""
        log.info("Boucle de capture démarrée.")

        for state in run_capture_loop(
            interval=self.capture_interval,
            debug=self.debug_capture,
        ):
            if _shutdown_event.is_set():
                break

            self._stats["states_captured"] += 1
            self._process_state(state)

        log.info("Boucle de capture terminée.")

    def _process_state(self, state: GameState) -> None:
        """Traite un seul GameState : engine → Claude → HUD."""

        # Valider les données minimales
        if not state.player_cards or len(state.player_cards) < 2:
            log.debug("Cartes joueur non détectées — état ignoré.")
            return

        log.info(
            f"État détecté — {state.stage.upper()} | "
            f"main={state.player_cards} | board={state.board_cards} | "
            f"pot={state.pot} | bet={state.current_bet}"
        )

        # Afficher "chargement" pendant le calcul
        self.hud_thread.hud.update_async(DisplayData(
            is_loading   = True,
            player_cards = state.player_cards,
            board_cards  = state.board_cards,
            stage        = state.stage,
            summary      = "Calcul en cours…",
        ))

        # ── Étape 1 : Engine ──────────────────────────────────────────
        try:
            result = self.engine.analyse_from_state(state)
            log.info(
                f"Engine → win={result.win_probability:.1%} | "
                f"main={result.hand_class} | action={result.recommended_action}"
            )
        except Exception as e:
            log.error(f"Erreur engine : {e}")
            self._stats["errors"] += 1
            self._show_error(state, f"Engine : {e}")
            return

        # ── Étape 2 : Claude API ──────────────────────────────────────
        try:
            advice = self.claude.get_advice(
                game_state    = state.to_dict(),
                equity_result = result.to_dict(),
            )
            self._stats["api_calls"] += 1

            if advice.error:
                log.warning(f"Claude a retourné une erreur : {advice.error}")
                self._show_engine_fallback(state, result, advice.error)
                return

        except Exception as e:
            log.error(f"Erreur Claude API : {e}")
            self._stats["errors"] += 1
            self._show_engine_fallback(state, result, str(e))
            return

        # ── Étape 3 : Mise à jour HUD ─────────────────────────────────
        display = DisplayData.from_advice(advice, state.to_dict())
        self.hud_thread.hud.update_async(display)

        log.info(
            f"HUD mis à jour → {advice.recommended_action} | "
            f"latence={advice.latency_ms:.0f}ms"
        )

        # ── Étape 4 : Enregistrement tracker ──────────────────────────
        self._record_to_tracker(state, result, advice)

    # ------------------------------------------------------------------
    # Gestion des nouvelles mains
    # ------------------------------------------------------------------

    def new_hand(self) -> None:
        """Réinitialise le contexte pour une nouvelle main."""
        self.claude.new_hand()
        self._current_hand_id    = 0
        self._current_hand_cards = []
        self._stats["hands"] += 1
        log.info(f"Nouvelle main #{self._stats['hands']} — contexte réinitialisé.")
        self._show_waiting()

    # ------------------------------------------------------------------
    # Enregistrement tracker
    # ------------------------------------------------------------------

    def _record_to_tracker(
        self,
        state:  GameState,
        result: EquityResult,
        advice: PokerAdvice,
    ) -> None:
        """Enregistre la décision courante dans la BD."""
        try:
            # Nouvelle main détectée si les cartes changent
            new_hand = state.player_cards != self._current_hand_cards
            if new_hand:
                self._current_hand_cards = state.player_cards
                hand_id = self.tracker.record_hand(HandRecord(
                    session_id         = self.session_id,
                    stage_final        = state.stage,
                    player_cards       = state.player_cards,
                    board_cards        = state.board_cards,
                    num_opponents      = max(0, state.num_players - 1),
                    pot_final          = state.pot,
                    hand_class         = result.hand_class,
                    win_probability    = result.win_probability,
                    recommended_action = advice.recommended_action,
                    action_taken       = advice.recommended_action,  # auto = conseil suivi
                    followed_advice    = True,
                    result             = 0.0,   # mis à jour à la fin de la main
                    ev_estimate        = result.ev_estimate,
                    ev_realized        = 0.0,
                ))
                self._current_hand_id = hand_id
                log.debug(f"Nouvelle main enregistrée → hand_id={hand_id}")
            else:
                # Décision sur une street existante
                if self._current_hand_id:
                    self.tracker.record_decision(
                        hand_id            = self._current_hand_id,
                        session_id         = self.session_id,
                        stage              = state.stage,
                        player_cards       = state.player_cards,
                        board_cards        = state.board_cards,
                        win_probability    = result.win_probability,
                        recommended_action = advice.recommended_action,
                        action_taken       = advice.recommended_action,
                        pot_size           = state.pot,
                        bet_size           = state.current_bet,
                        ev_estimate        = result.ev_estimate,
                    )
        except Exception as e:
            log.error(f"Erreur enregistrement tracker : {e}")

    # ------------------------------------------------------------------
    # Clôture de session
    # ------------------------------------------------------------------

    def _close_session(self) -> None:
        """Demande le résultat final et clôture la session dans la BD."""
        print("\n" + "=" * 50)
        print("  CLÔTURE DE SESSION")
        print("=" * 50)
        try:
            placement = int(input("  Votre placement final (ex: 1, 2, 3…) : ") or 0)
            prize     = float(input("  Prix remporté en $ (0 si aucun)       : ") or 0)
        except (ValueError, EOFError):
            placement, prize = 0, 0.0

        self.tracker.end_session(
            session_id  = self.session_id,
            placement   = placement,
            prize       = prize,
        )
        profit = prize - self.buy_in
        log.info(
            f"Session #{self.session_id} clôturée — "
            f"placement={placement} | prize={prize}$ | profit={profit:+.2f}$"
        )

    # ------------------------------------------------------------------
    # Listener console (commandes pendant la partie)
    # ------------------------------------------------------------------

    def _console_listener(self) -> None:
        """
        Écoute les commandes tapées dans la console pendant la partie :
          n      → nouvelle main
          end    → clôturer la session manuellement
          stats  → ouvrir le dashboard stats
        """
        while not _shutdown_event.is_set():
            try:
                cmd = input().strip().lower()
                if cmd == "n":
                    self.new_hand()
                    print("  ✓ Nouvelle main — contexte réinitialisé.")
                elif cmd == "end":
                    _shutdown_event.set()
                    self.hud_thread.hud.stop()
                elif cmd == "stats":
                    threading.Thread(
                        target=lambda: StatsViewer(self.tracker).run(),
                        daemon=True,
                    ).start()
                else:
                    print("  Commandes : 'n' = nouvelle main | 'end' = quitter | 'stats' = dashboard")
            except EOFError:
                break
            except Exception as e:
                log.debug(f"Console listener : {e}")

    # ------------------------------------------------------------------
    # États HUD spéciaux
    # ------------------------------------------------------------------

    def _show_waiting(self) -> None:
        self.hud_thread.hud.update_async(DisplayData(
            is_loading = True,
            summary    = "En attente d'une table PokerStars…",
        ))

    def _show_error(self, state: GameState, msg: str) -> None:
        self.hud_thread.hud.update_async(DisplayData(
            error        = msg,
            player_cards = state.player_cards,
            board_cards  = state.board_cards,
            stage        = state.stage,
        ))

    def _show_engine_fallback(
        self,
        state:   GameState,
        result:  EquityResult,
        error:   str,
    ) -> None:
        """Affiche les données engine locales si Claude est indisponible."""
        action = result.recommended_action or "—"
        self.hud_thread.hud.update_async(DisplayData(
            action           = action,
            action_color     = ACTION_COLORS.get(action, "#E8E8E8"),
            sizing           = result.recommended_sizing,
            win_probability  = result.win_probability,
            hand_class       = result.hand_class,
            stage            = state.stage,
            player_cards     = state.player_cards,
            board_cards      = state.board_cards,
            num_opponents    = max(0, state.num_players - 1),
            hands_beating    = result.num_combos_beating,
            ev_estimate      = result.ev_estimate,
            summary          = f"[Fallback local] {error[:80]}",
        ))

    # ------------------------------------------------------------------
    # Stats de session
    # ------------------------------------------------------------------

    def _print_session_stats(self) -> None:
        elapsed = time.time() - self._stats["start_time"]
        log.info("=" * 50)
        log.info("  RÉSUMÉ DE SESSION")
        log.info(f"  Durée          : {elapsed/60:.1f} min")
        log.info(f"  Mains jouées   : {self._stats['hands']}")
        log.info(f"  États capturés : {self._stats['states_captured']}")
        log.info(f"  Appels Claude  : {self._stats['api_calls']}")
        log.info(f"  Erreurs        : {self._stats['errors']}")
        log.info("=" * 50)


# ---------------------------------------------------------------------------
# Mode démo (sans PokerStars)
# ---------------------------------------------------------------------------

def run_demo(hud_x: int = 40, hud_y: int = 200) -> None:
    """
    Simule une main complète sans PokerStars pour tester le pipeline.
    Utile pour vérifier l'installation et l'API Claude.
    """
    log.info("Mode démo activé — aucune capture d'écran réelle.")

    hud_thread = HUDThread(x=hud_x, y=hud_y)
    hud_thread.start()
    hud_thread.wait_ready()

    from engine        import PokerEngine, EquityResult
    from claude_client import CachedClaudeClient

    engine = PokerEngine(simulations=1500)
    claude = CachedClaudeClient()

    # Scénarios d'une main complète
    scenarios = [
        # (label, hole_cards, board, pot, bet, num_opponents)
        ("Préflop", ["Ks", "7h"], [],                          0,    0,    7),
        ("Flop",    ["Ks", "7h"], ["Kd", "7d", "2c"],        120,   40,   3),
        ("Turn",    ["Ks", "7h"], ["Kd", "7d", "2c", "As"],  200,   80,   2),
        ("River",   ["Ks", "7h"], ["Kd", "7d", "2c", "As", "3h"], 360, 150, 1),
    ]

    for label, hole, board, pot, bet, opponents in scenarios:
        if _shutdown_event.is_set():
            break

        log.info(f"─── Démo : {label} ───")

        # Afficher loading
        hud_thread.hud.update_async(DisplayData(
            is_loading   = True,
            player_cards = hole,
            board_cards  = board,
            stage        = label.lower(),
            summary      = f"Calcul {label}…",
        ))
        time.sleep(0.5)

        # Engine
        result = engine.analyse(
            hole_cards    = hole,
            board         = board,
            num_opponents = opponents,
            pot           = pot,
            call_amount   = bet,
        )

        # Claude
        mock_state = {
            "player_cards": hole,
            "board_cards":  board,
            "pot":          pot,
            "player_stack": 1000.0,
            "current_bet":  bet,
            "num_players":  opponents + 1,
            "stage":        label.lower(),
            "timestamp":    time.time(),
        }

        try:
            advice = claude.get_advice(mock_state, result.to_dict())
            display = DisplayData.from_advice(advice, mock_state)
        except Exception as e:
            log.warning(f"Claude indisponible ({e}) — fallback engine")
            action = result.recommended_action or "—"
            display = DisplayData(
                action          = action,
                action_color    = ACTION_COLORS.get(action, "#E8E8E8"),
                sizing          = result.recommended_sizing,
                win_probability = result.win_probability,
                hand_class      = result.hand_class,
                stage           = label.lower(),
                player_cards    = hole,
                board_cards     = board,
                num_opponents   = opponents,
                ev_estimate     = result.ev_estimate,
                summary         = f"Fallback : {result.recommended_action} | {result.win_probability:.0%}",
            )

        hud_thread.hud.update_async(display)
        log.info(f"  → {display.action} | equity={display.win_probability:.0%}")
        time.sleep(3.5)

    if not _shutdown_event.is_set():
        log.info("Démo terminée. Ferme le HUD pour quitter.")
        hud_thread.join()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="♠ Poker HUD Assistant — Assistant poker temps réel propulsé par Claude"
    )
    p.add_argument("--interval",    type=float, default=1.5,
                   help="Intervalle de capture en secondes (défaut : 1.5)")
    p.add_argument("--simulations", type=int,   default=2000,
                   help="Simulations Monte Carlo (défaut : 2000)")
    p.add_argument("--hud-x",       type=int,   default=40,
                   help="Position X du HUD (défaut : 40)")
    p.add_argument("--hud-y",       type=int,   default=200,
                   help="Position Y du HUD (défaut : 200)")
    p.add_argument("--buy-in",      type=float, default=0.0,
                   help="Buy-in de la session en $ (défaut : 0)")
    p.add_argument("--game",        type=str,   default="Tournoi",
                   help="Type de jeu : Tournoi / Cash Game (défaut : Tournoi)")
    p.add_argument("--players",     type=int,   default=8,
                   help="Nombre de joueurs à la table (défaut : 8)")
    p.add_argument("--debug",       action="store_true",
                   help="Sauvegarder les images de capture pour debug")
    p.add_argument("--demo",        action="store_true",
                   help="Mode démo : simule une main sans PokerStars")
    p.add_argument("--stats",       action="store_true",
                   help="Ouvrir uniquement le dashboard de statistiques")
    p.add_argument("--api-key",     type=str,   default=None,
                   help="Clé API Anthropic (sinon utilise ANTHROPIC_API_KEY)")
    p.add_argument("--verbose",     action="store_true",
                   help="Active les logs DEBUG")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entrée principale
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Mode stats uniquement
    if args.stats:
        tracker = PokerTracker()
        StatsViewer(tracker).run()
        return

    # Mode démo
    if args.demo:
        run_demo(hud_x=args.hud_x, hud_y=args.hud_y)
        return

    # Mode normal
    assistant = PokerAssistant(
        capture_interval = args.interval,
        simulations      = args.simulations,
        hud_x            = args.hud_x,
        hud_y            = args.hud_y,
        debug_capture    = args.debug,
        api_key          = args.api_key,
        buy_in           = args.buy_in,
        game_type        = args.game,
        num_players      = args.players,
    )
    assistant.run()


if __name__ == "__main__":
    main()