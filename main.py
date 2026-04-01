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
    from ai_client      import create_ai_client, PokerAIClient
    from overlay       import PokerHUD, HUDThread, DisplayData, ACTION_COLORS
    from tracker        import PokerTracker, HandRecord
    from stats_viewer   import StatsViewer
    from card_detector  import HybridCardExtractor as _HybridExtractor
    from hh_parser      import start_hh_watcher, find_hh_folder
    from profil_builder import ProfilBuilder, ProfileDatabase, build_opponent_profiles_context
    from bluff_detector import PatternEngine, update_engine_from_game_state
    from alerts         import create_alert_manager, AlertManager
    from auto_new_hand  import AutoNewHandDetector
    from rapport_pdf    import generate_session_report_async
    from hand_replay    import HandReplayViewer
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
        hero_name:         str   = "",
        hh_folder:         str   = "",
        import_hh:         bool  = False,
        provider:          str   = "auto",
        openai_key:        str   = "",
        azure_key:         str   = "",
        azure_endpoint:    str   = "",
        azure_deploy:      str   = "",
    ):
        self.capture_interval = capture_interval
        self.debug_capture    = debug_capture
        self.buy_in           = buy_in
        self.game_type        = game_type
        self.num_players      = num_players

        log.info("Initialisation des modules…")

        # Module 1 — Capture
        self.extractor = PokerExtractor(debug=debug_capture)
        # Vérifier si la détection par templates est active
        try:
            from card_detector import HybridCardExtractor
            log.info("  ✓ capture.py        — PokerExtractor prêt (templates OpenCV actifs)")
        except ImportError:
            log.info("  ✓ capture.py        — PokerExtractor prêt (OCR Tesseract)")

        # Module 2 — Engine
        self.engine = PokerEngine(simulations=simulations)
        log.info(f"  ✓ engine.py         — PokerEngine prêt ({simulations} simulations MC)")

        # Module 3 — Claude
        # Créer le client IA (Claude / OpenAI / Copilot selon provider)
        try:
            self.claude = create_ai_client(
                provider       = provider,
                claude_key     = api_key,
                openai_key     = openai_key,
                azure_key      = azure_key,
                azure_endpoint = azure_endpoint,
                azure_deploy   = azure_deploy,
            )
            log.info(f"  ✓ ai_client.py      — {self.claude.active_provider} prêt")
        except Exception as e:
            log.warning(f"ai_client non disponible ({e}) — fallback CachedClaudeClient")
            self.claude = CachedClaudeClient(api_key=api_key)

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

        # Module 6 — Surveillance Hand History
        self._hh_watcher = None
        self._hero_name  = hero_name
        self._hh_folder  = hh_folder

        # Import HH existant au démarrage si demandé
        if import_hh:
            try:
                from hh_parser import HandHistoryImporter, find_hh_folder
                from pathlib import Path
                hh_path = Path(hh_folder) if hh_folder else find_hh_folder()
                if hh_path:
                    imp = HandHistoryImporter(self.tracker, hero_name=hero_name)
                    n   = imp.import_folder(hh_path)
                    log.info(f"  ✓ hh_parser.py — {n} mains importées depuis {hh_path}")
                else:
                    log.warning("Import HH : dossier non trouvé.")
            except Exception as e:
                log.warning(f"Import HH initial échoué : {e}")

        # Module 7 — ProfilBuilder adverses
        try:
            self.profil_db      = ProfileDatabase()
            self.profil_builder = ProfilBuilder(self.profil_db)
            log.info("  ✓ profil_builder.py — ProfilBuilder prêt")
        except Exception as e:
            self.profil_builder = None
            log.warning(f"ProfilBuilder non disponible : {e}")

        # Module 8 — PatternEngine (bluff + tendances)
        try:
            self.pattern_engine = PatternEngine(
                api_key        = api_key,
                profil_builder = self.profil_builder,
            )
            log.info("  ✓ bluff_detector.py — PatternEngine prêt")
        except Exception as e:
            self.pattern_engine = None
            log.warning(f"PatternEngine non disponible : {e}")

        # Module 9 — Alertes sonores
        try:
            self.alerter = create_alert_manager(enabled=True, volume=0.7)
            log.info("  ✓ alerts.py         — AlertManager prêt")
        except Exception as e:
            self.alerter = None
            log.warning(f"AlertManager non disponible : {e}")

        # Module 10 — Détection automatique nouvelle main
        try:
            self._auto_detector = AutoNewHandDetector(
                callback       = self.new_hand,
                min_interval_s = 6.0,
                hand_timeout_s = 90.0,
                enabled        = True,
            )
            log.info("  ✓ auto_new_hand.py  — AutoNewHandDetector actif")
        except Exception as e:
            self._auto_detector = None
            log.warning(f"AutoNewHandDetector non disponible : {e}")

        # État de la main en cours
        self._current_hand_id:    int   = 0
        self._current_hand_cards: list  = []
        self._hand_result:        float = 0.0
        # Historique des décisions de la main courante (max 10 entrées)
        self._hand_history:       list  = []

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

        # Démarrer la surveillance Hand History (tourne en arrière-plan)
        try:
            from hh_parser import start_hh_watcher, find_hh_folder
            from pathlib import Path
            hh_path = Path(self._hh_folder) if hasattr(self, "_hh_folder") and self._hh_folder else None
            self._hh_watcher = start_hh_watcher(
                self.tracker,
                hero_name = self._hero_name,
                hh_folder = hh_path,
            )
            if self._hh_watcher:
                log.info("  ✓ hh_parser.py — Surveillance Hand History active")
        except Exception as e:
            log.warning(f"Surveillance HH non démarrée : {e}")
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

        # Détection automatique nouvelle main
        if self._auto_detector:
            self._auto_detector.update(state)

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

        # Construire les profils adverses pour Claude
        opponent_profiles = None
        if self.profil_builder:
            try:
                all_names = getattr(state, "player_names", [])
                if all_names:
                    if self.pattern_engine:
                        # Profils enrichis avec patterns de bluff
                        opponent_profiles = self.pattern_engine.get_structured_profiles(all_names)
                    else:
                        opponent_profiles = build_opponent_profiles_context(
                            self.profil_builder, all_names, hero_name=self._hero_name,
                        )
            except Exception as e:
                log.debug(f"Profils adverses indisponibles : {e}")

        try:
            advice = self.claude.get_advice(
                game_state        = state.to_dict(),
                equity_result     = result.to_dict(),
                opponent_profiles = opponent_profiles,
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

        log.info(
            f"HUD mis à jour → {advice.recommended_action} | "
            f"latence={advice.latency_ms:.0f}ms"
        )

        # Enregistrer dans l'historique de la main
        self._hand_history.append({
            "stage":  state.stage,
            "action": advice.recommended_action or "—",
            "equity": result.win_probability,
            "cards":  state.player_cards,
            "board":  state.board_cards,
            "sizing": advice.recommended_sizing,
            "spr":    result.spr,
        })
        if len(self._hand_history) > 10:
            self._hand_history = self._hand_history[-10:]

        # Injecter l'historique dans le DisplayData
        display.hand_history = list(self._hand_history)
        self.hud_thread.hud.update_async(display)

        # Alerte sonore selon le conseil
        if self.alerter:
            try:
                self.alerter.on_advice(advice, result)
                self.alerter.on_spr(result.spr)
            except Exception as e:
                log.debug(f"Alerte sonore : {e}")

        # ── Étape 4 : Mise à jour profils + patterns adverses ───────────
        if self.profil_builder and state.opponent_actions:
            try:
                for action in state.opponent_actions:
                    seat_name = f"Seat_{action.seat}"
                    act_str = action.action.value if hasattr(action.action, "value") else str(action.action)
                    self.profil_builder.update_realtime(
                        player_name=seat_name, action=act_str,
                        amount=action.amount, stage=state.stage,
                    )
            except Exception as e:
                log.debug(f"Profils RT : {e}")

        if self.pattern_engine:
            try:
                update_engine_from_game_state(self.pattern_engine, state, self._hero_name)
            except Exception as e:
                log.debug(f"PatternEngine RT : {e}")

        # Alertes actions adverses
        if self.alerter and state.opponent_actions:
            try:
                for action in state.opponent_actions:
                    act_str = action.action.value if hasattr(action.action, "value") else str(action.action)
                    self.alerter.on_opponent_action(act_str, action.amount, state.pot)
            except Exception as e:
                log.debug(f"Alerte adverse : {e}")

        # ── Étape 5 : Enregistrement tracker ──────────────────────────
        self._record_to_tracker(state, result, advice)

    # ------------------------------------------------------------------
    # Gestion des nouvelles mains
    # ------------------------------------------------------------------

    def new_hand(self) -> None:
        """Réinitialise le contexte pour une nouvelle main."""
        self.claude.new_hand()
        self._current_hand_id    = 0
        self._current_hand_cards = []
        self._hand_history       = []
        self._stats["hands"] += 1
        log.info(f"Nouvelle main #{self._stats['hands']} — contexte réinitialisé.")
        if self.alerter:
            self.alerter.on_new_hand()
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
        if self._hh_watcher:
            self._hh_watcher.stop()
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

        # Sauvegarder les profils adverses
        if self.profil_builder:
            try:
                self.profil_builder.end_hand_realtime()
                log.info("Profils adverses sauvegardés.")
            except Exception as e:
                log.warning(f"Sauvegarde profils : {e}")

        # Générer le rapport PDF de session
        try:
            pdf_path = generate_session_report_async(self.tracker, self.session_id)
            if pdf_path:
                print(f"\n  Rapport PDF sauvegardé : {pdf_path}")
        except Exception as e:
            log.warning(f"Rapport PDF : {e}")
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
                    if self._auto_detector:
                        self._auto_detector.force("commande console")
                    else:
                        self.new_hand()
                    print("  ✓ Nouvelle main — contexte réinitialisé.")
                elif cmd == "end":
                    _shutdown_event.set()
                    self.hud_thread.hud.stop()
                elif cmd == "provider":
                    prov = getattr(self.claude, "active_provider", "Claude")
                    print(f"  Provider actif : {prov}")
                elif cmd in ("auto on", "auto off"):
                    if self._auto_detector:
                        self._auto_detector.enabled = (cmd == "auto on")
                        state = "activée" if self._auto_detector.enabled else "désactivée"
                        print(f"  Détection auto : {state}")
                elif cmd == "auto status":
                    if self._auto_detector:
                        print(f"  {self._auto_detector.status_str()}")
                elif cmd == "stats":
                    threading.Thread(
                        target=lambda: StatsViewer(self.tracker).run(),
                        daemon=True,
                    ).start()
                elif cmd == "pdf":
                    try:
                        pdf_path = generate_session_report_async(
                            self.tracker, self.session_id
                        )
                        if pdf_path:
                            print(f"  Rapport PDF : {pdf_path}")
                    except Exception as e:
                        print(f"  Erreur PDF : {e}")
                elif cmd == "replay":
                    threading.Thread(
                        target=lambda: HandReplayViewer(
                            self.tracker,
                            session_id=self.session_id
                        ).run(),
                        daemon=True,
                    ).start()
                elif cmd.startswith("p ") or cmd.startswith("profil "):
                    # Afficher le profil d'un adversaire : "p Villain42"
                    name = cmd.split(" ", 1)[1].strip()
                    if self.profil_builder:
                        p = self.profil_builder.get_profile(name)
                        from profil_builder import print_profile
                        print_profile(p)
                    else:
                        print("  ProfilBuilder non disponible.")
                elif cmd.startswith("patterns ") or cmd.startswith("pat "):
                    name = cmd.split(" ", 1)[1].strip()
                    if self.pattern_engine:
                        report = self.pattern_engine.get_tendency_report(name)
                        print(f"\n  Patterns pour {name} :")
                        for p in report.patterns:
                            status = "✓" if p.is_reliable else "~"
                            print(f"  {status} {p.description}")
                            print(f"    → {p.exploitation}")
                        if report.top_exploitation:
                            print(f"\n  [ACTION] {report.top_exploitation}")
                    else:
                        print("  PatternEngine non disponible.")
                elif cmd == "top":
                    if self.profil_builder:
                        profiles = self.profil_builder.top_players(10)
                        header = f"  {'Nom':<18} {'Mains':>5} {'VPIP':>5} {'PFR':>5} {'AF':>5} Tendance"
                        print("")
                        print(header)
                        for p in profiles:
                            row = f"  {p.name:<18} {p.hands_total:>5} {p.vpip:>4.0f}% {p.pfr:>4.0f}% {p.af:>5.1f} {p.tendency}"
                            print(row)
                elif cmd == "sound on":
                    if self.alerter:
                        self.alerter.set_enabled(True)
                        print("  Alertes sonores activées.")
                elif cmd == "sound off":
                    if self.alerter:
                        self.alerter.set_enabled(False)
                        print("  Alertes sonores désactivées.")
                elif cmd == "sound test":
                    if self.alerter:
                        threading.Thread(target=self.alerter.test_all, daemon=True).start()
                else:
                    print("  Commandes : 'n'=nouvelle main | 'end'=quitter | 'stats'=dashboard")
                    print("              'p NOM'=profil | 'top'=top 10 | 'pdf'=rapport PDF | 'replay'=rejouer mains")
                    print("              'sound on/off/test'=alertes | 'patterns NOM'=bluff patterns")
                    print("              'auto on/off'=détection auto | 'auto status'=état détection")
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
    p.add_argument("--calibrate",   action="store_true",
                   help="Lancer le calibrateur visuel de régions (hud_calibrator.py)")
    p.add_argument("--gen-templates", action="store_true",
                   help="Générer les templates de détection de cartes")
    p.add_argument("--hero",          type=str, default="",
                   help="Pseudo PokerStars du joueur (pour import Hand History)")
    p.add_argument("--provider",      type=str, default="auto",
                   choices=["auto","claude","openai","copilot","heuristic"],
                   help="Provider IA : auto (défaut), claude, openai, copilot, heuristic")
    p.add_argument("--openai-key",    type=str, default=None,
                   help="Clé API OpenAI (ou variable OPENAI_API_KEY)")
    p.add_argument("--azure-key",     type=str, default=None,
                   help="Clé Azure OpenAI pour Copilot (ou AZURE_OPENAI_API_KEY)")
    p.add_argument("--azure-url",     type=str, default=None,
                   help="Endpoint Azure OpenAI (ou AZURE_OPENAI_ENDPOINT)")
    p.add_argument("--azure-deploy",  type=str, default=None,
                   help="Déploiement Azure (ou AZURE_OPENAI_DEPLOYMENT, défaut: gpt-4o)")
    p.add_argument("--hh-folder",     type=str, default=None,
                   help="Dossier HandHistory (détecté auto si absent)")
    p.add_argument("--import-hh",     action="store_true",
                   help="Importer tout l'historique HH existant au démarrage")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entrée principale
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Mode calibration visuelle
    if args.calibrate:
        try:
            from hud_calibrator import run_calibrator
            run_calibrator()
        except ImportError:
            log.error("hud_calibrator.py introuvable dans le dossier.")
        return

    # Mode génération de templates
    if args.gen_templates:
        try:
            from card_detector import TemplateGenerator
            gen = TemplateGenerator()
            n   = gen.generate_all()
            print(f"\n✓ {n} templates générés dans le dossier templates/")
            print("  Relancez main.py normalement pour utiliser la détection améliorée.")
        except ImportError:
            log.error("card_detector.py introuvable dans le dossier.")
        return

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
        hero_name        = args.hero,
        hh_folder        = args.hh_folder or "",
        import_hh        = args.import_hh,
        provider         = args.provider,
        openai_key       = args.openai_key  or "",
        azure_key        = args.azure_key   or "",
        azure_endpoint   = args.azure_url   or "",
        azure_deploy     = args.azure_deploy or "",
    )
    assistant.run()


if __name__ == "__main__":
    main()
