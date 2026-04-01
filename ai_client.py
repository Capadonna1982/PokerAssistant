"""
ai_client.py — Client IA multi-provider pour l'assistant poker
Supporte : Anthropic Claude, OpenAI ChatGPT, Microsoft Copilot (Azure OpenAI)

Installation :
    pip install anthropic          # Claude
    pip install openai             # ChatGPT + Copilot

Variables d'environnement :
    ANTHROPIC_API_KEY              # Anthropic Claude
    OPENAI_API_KEY                 # OpenAI ChatGPT
    AZURE_OPENAI_API_KEY           # Microsoft Copilot (Azure)
    AZURE_OPENAI_ENDPOINT          # ex: https://monressource.openai.azure.com/
    AZURE_OPENAI_DEPLOYMENT        # nom du déploiement, ex: gpt-4o

Priorité par défaut : Claude → OpenAI → Copilot (Azure) → fallback heuristique
On peut forcer un provider spécifique via provider="claude"|"openai"|"copilot"

Usage :
    client = PokerAIClient(provider="auto")          # choisit automatiquement
    client = PokerAIClient(provider="openai")        # force ChatGPT
    client = PokerAIClient(provider="copilot")       # force Azure/Copilot

    advice = client.get_advice(game_state, equity_result)
    print(advice.recommended_action)
    print(f"Provider utilisé : {client.active_provider}")
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

# Réimporter le prompt et PokerAdvice depuis claude_client pour compatibilité
try:
    from claude_client import (
        PokerAdvice,
        SYSTEM_PROMPT,
        build_user_prompt,
        MAX_TOKENS,
        RETRY_LIMIT,
        RETRY_DELAY,
    )
    _HAS_CLAUDE_CLIENT = True
except ImportError:
    _HAS_CLAUDE_CLIENT = False
    log.warning("claude_client.py introuvable — fonctionnement dégradé.")

# ---------------------------------------------------------------------------
# Modèles disponibles par provider
# ---------------------------------------------------------------------------

PROVIDER_MODELS = {
    "claude":  "claude-sonnet-4-20250514",
    "openai":  "gpt-4o",
    "copilot": "gpt-4o",             # déploiement Azure — surchargeable via env
}

# Noms affichables
PROVIDER_NAMES = {
    "claude":  "Anthropic Claude",
    "openai":  "OpenAI ChatGPT",
    "copilot": "Microsoft Copilot",
    "heuristic": "Heuristique locale",
}


# ---------------------------------------------------------------------------
# Backends individuels
# ---------------------------------------------------------------------------

class _ClaudeBackend:
    """Backend Anthropic Claude."""

    name = "claude"

    def __init__(self, api_key: Optional[str] = None):
        import anthropic
        self._client = (
            anthropic.Anthropic(api_key=api_key)
            if api_key else
            anthropic.Anthropic()
        )

    def call(self, system: str, messages: list[dict], max_tokens: int) -> str:
        response = self._client.messages.create(
            model      = PROVIDER_MODELS["claude"],
            max_tokens = max_tokens,
            system     = system,
            messages   = messages,
        )
        for block in response.content:
            if hasattr(block, "text"):
                return block.text.strip()
        return ""

    @staticmethod
    def is_available() -> bool:
        try:
            import anthropic
            return bool(os.environ.get("ANTHROPIC_API_KEY"))
        except ImportError:
            return False


class _OpenAIBackend:
    """Backend OpenAI ChatGPT (gpt-4o, gpt-4-turbo, etc.)."""

    name = "openai"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._model  = model or PROVIDER_MODELS["openai"]

    def call(self, system: str, messages: list[dict], max_tokens: int) -> str:
        # OpenAI utilise le message system dans la liste de messages
        full_messages = [{"role": "system", "content": system}] + messages
        response = self._client.chat.completions.create(
            model      = self._model,
            max_tokens = max_tokens,
            messages   = full_messages,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def is_available() -> bool:
        try:
            import openai
            return bool(os.environ.get("OPENAI_API_KEY"))
        except ImportError:
            return False


class _CopilotBackend:
    """
    Backend Microsoft Copilot via Azure OpenAI Service.

    Prérequis :
        pip install openai
        Variables d'environnement :
            AZURE_OPENAI_API_KEY      → clé Azure
            AZURE_OPENAI_ENDPOINT     → https://xxx.openai.azure.com/
            AZURE_OPENAI_DEPLOYMENT   → nom du déploiement (ex: gpt-4o)
    """

    name = "copilot"

    def __init__(
        self,
        api_key:    Optional[str] = None,
        endpoint:   Optional[str] = None,
        deployment: Optional[str] = None,
    ):
        from openai import AzureOpenAI

        self._deployment = (
            deployment
            or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        )
        self._client = AzureOpenAI(
            api_key         = api_key     or os.environ.get("AZURE_OPENAI_API_KEY"),
            azure_endpoint  = endpoint    or os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            api_version     = "2024-02-01",
        )

    def call(self, system: str, messages: list[dict], max_tokens: int) -> str:
        full_messages = [{"role": "system", "content": system}] + messages
        response = self._client.chat.completions.create(
            model      = self._deployment,
            max_tokens = max_tokens,
            messages   = full_messages,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def is_available() -> bool:
        try:
            import openai
            return bool(
                os.environ.get("AZURE_OPENAI_API_KEY") and
                os.environ.get("AZURE_OPENAI_ENDPOINT")
            )
        except ImportError:
            return False


class _HeuristicBackend:
    """
    Fallback purement local — aucune API requise.
    Utilise engine.py pour une recommandation basique sans LLM.
    """

    name = "heuristic"

    def call(self, system: str, messages: list[dict], max_tokens: int) -> str:
        # Extraire le payload JSON du dernier message utilisateur
        try:
            user_content = messages[-1]["content"]
            # Trouver le bloc JSON dans le prompt
            start = user_content.rfind("{")
            end   = user_content.rfind("}") + 1
            if start >= 0 and end > start:
                payload = json.loads(user_content[start:end])
                action  = payload.get("recommended_action", "CHECK")
                equity  = payload.get("win_probability", 0.5)
                spr     = payload.get("spr", 5.0)
                comment = f"Analyse heuristique locale (pas de connexion IA). Équité : {equity:.0%}."
                return json.dumps({
                    "recommended_action":      action,
                    "recommended_bet_size":    "",
                    "estimated_win_probability": equity,
                    "hand_class":              payload.get("hand_class", ""),
                    "hands_that_beat_us":      0,
                    "summary":                 comment,
                    "action_explanations":     comment,
                    "position":                payload.get("position", ""),
                    "position_advantage":      99,
                    "position_comment":        "",
                    "spr":                     spr,
                    "spr_label":               "",
                    "spr_comment":             "",
                    "pot_odds":                0.0,
                    "mdf":                     0.0,
                }, ensure_ascii=False)
        except Exception:
            pass
        return json.dumps({
            "recommended_action": "CHECK",
            "summary": "Pas de connexion IA disponible.",
        })

    @staticmethod
    def is_available() -> bool:
        return True   # toujours disponible


# ---------------------------------------------------------------------------
# Client unifié multi-provider
# ---------------------------------------------------------------------------

class PokerAIClient:
    """
    Client IA poker multi-provider avec fallback automatique.

    Paramètres :
        provider   : "auto" | "claude" | "openai" | "copilot" | "heuristic"
                     "auto" → essaie dans l'ordre : Claude → OpenAI → Copilot → heuristique
        api_key    : clé API du provider principal (sinon variable d'environnement)
        fallback   : True = tente un autre provider si le principal échoue

    Exemple :
        # Choisir automatiquement selon les clés disponibles
        client = PokerAIClient()

        # Forcer ChatGPT
        client = PokerAIClient(provider="openai", api_key="sk-...")

        # Forcer Copilot Azure
        client = PokerAIClient(provider="copilot")
        # → lit AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT depuis l'environnement
    """

    PROVIDER_ORDER = ["claude", "openai", "copilot", "heuristic"]

    def __init__(
        self,
        provider:   str            = "auto",
        api_key:    Optional[str]  = None,
        fallback:   bool           = True,
        openai_model: Optional[str] = None,
    ):
        self.requested_provider = provider
        self.fallback           = fallback
        self._history:          list[dict]            = []
        self._last_advice:      Optional[PokerAdvice] = None
        self._call_count:       int                   = 0
        self._error_count:      int                   = 0

        # Construire la liste ordonnée de backends
        self._backends = self._build_backends(provider, api_key, openai_model)

        if self._backends:
            log.info(
                f"PokerAIClient initialisé — "
                f"provider principal : {self._backends[0].name} "
                f"({'fallback activé' if fallback and len(self._backends) > 1 else 'no fallback'})"
            )
        else:
            log.error("Aucun backend IA disponible !")

    def _build_backends(
        self,
        provider:    str,
        api_key:     Optional[str],
        openai_model:Optional[str],
    ) -> list:
        backends = []

        if provider == "auto":
            # Essayer dans l'ordre de priorité selon les clés disponibles
            if _ClaudeBackend.is_available():
                try:
                    backends.append(_ClaudeBackend(api_key))
                except Exception as e:
                    log.warning(f"Claude init échoué : {e}")

            if _OpenAIBackend.is_available():
                try:
                    backends.append(_OpenAIBackend(model=openai_model))
                except Exception as e:
                    log.warning(f"OpenAI init échoué : {e}")

            if _CopilotBackend.is_available():
                try:
                    backends.append(_CopilotBackend())
                except Exception as e:
                    log.warning(f"Copilot init échoué : {e}")

            backends.append(_HeuristicBackend())

        elif provider == "claude":
            try:
                backends.append(_ClaudeBackend(api_key))
            except Exception as e:
                log.error(f"Claude init échoué : {e}")
            if self.fallback:
                backends.append(_HeuristicBackend())

        elif provider == "openai":
            try:
                backends.append(_OpenAIBackend(api_key, openai_model))
            except Exception as e:
                log.error(f"OpenAI init échoué : {e}")
            if self.fallback:
                backends.append(_HeuristicBackend())

        elif provider == "copilot":
            try:
                backends.append(_CopilotBackend(api_key))
            except Exception as e:
                log.error(f"Copilot init échoué : {e}")
            if self.fallback:
                backends.append(_HeuristicBackend())

        elif provider == "heuristic":
            backends.append(_HeuristicBackend())

        else:
            log.error(f"Provider inconnu : {provider}. Utilisation heuristique.")
            backends.append(_HeuristicBackend())

        return backends

    # ------------------------------------------------------------------
    # API publique (identique à ClaudePokerClient)
    # ------------------------------------------------------------------

    @property
    def active_provider(self) -> str:
        """Nom du provider actif (premier backend disponible)."""
        if self._backends:
            return PROVIDER_NAMES.get(self._backends[0].name, self._backends[0].name)
        return "aucun"

    @property
    def active_provider_key(self) -> str:
        if self._backends:
            return self._backends[0].name
        return "heuristic"

    def get_advice(
        self,
        game_state:        dict,
        equity_result:     dict,
        opponent_profiles: Optional[dict] = None,
        use_history:       bool = True,
    ) -> "PokerAdvice":
        """
        Envoie la situation à l'IA et retourne un PokerAdvice.
        Interface identique à ClaudePokerClient.get_advice().
        """
        if not _HAS_CLAUDE_CLIENT:
            return PokerAdvice(error="claude_client.py introuvable")

        user_msg = build_user_prompt(game_state, equity_result, opponent_profiles)

        if use_history:
            self._history.append({"role": "user", "content": user_msg})
            messages = self._history.copy()
        else:
            messages = [{"role": "user", "content": user_msg}]

        advice = self._call_with_fallback(messages)

        if use_history and advice.raw_json:
            self._history.append({
                "role":    "assistant",
                "content": json.dumps(advice.raw_json, ensure_ascii=False),
            })

        self._last_advice = advice
        self._call_count += 1
        return advice

    def new_hand(self) -> None:
        """Réinitialise le contexte pour une nouvelle main."""
        self._history     = []
        self._last_advice = None
        log.info("Contexte IA réinitialisé — nouvelle main.")

    @property
    def last_advice(self) -> Optional["PokerAdvice"]:
        return self._last_advice

    # ------------------------------------------------------------------
    # Appel avec fallback
    # ------------------------------------------------------------------

    def _call_with_fallback(self, messages: list[dict]) -> "PokerAdvice":
        """
        Essaie chaque backend dans l'ordre.
        Passe au suivant si le précédent échoue.
        """
        last_error = ""

        for backend in self._backends:
            for attempt in range(1, RETRY_LIMIT + 1):
                t0 = time.monotonic()
                try:
                    raw_text   = backend.call(SYSTEM_PROMPT, messages, MAX_TOKENS)
                    latency_ms = (time.monotonic() - t0) * 1000

                    log.info(
                        f"[{PROVIDER_NAMES.get(backend.name, backend.name)}] "
                        f"répondu en {latency_ms:.0f}ms"
                    )

                    from claude_client import ClaudePokerClient
                    advice = ClaudePokerClient._parse_response(raw_text, latency_ms)

                    # Annoter avec le provider utilisé
                    advice.provider = backend.name
                    return advice

                except Exception as e:
                    last_error = f"[{backend.name}] {e}"
                    log.warning(f"Tentative {attempt} échouée ({backend.name}) : {e}")
                    if attempt < RETRY_LIMIT:
                        time.sleep(RETRY_DELAY)

            # Ce backend a échoué — passer au suivant si fallback activé
            if not self.fallback:
                break
            log.info(f"Fallback → prochain provider…")

        self._error_count += 1
        return PokerAdvice(error=f"Tous les providers ont échoué : {last_error}")


# ---------------------------------------------------------------------------
# Configuration via main.py
# ---------------------------------------------------------------------------

def create_ai_client(
    provider:      str           = "auto",
    claude_key:    Optional[str] = None,
    openai_key:    Optional[str] = None,
    azure_key:     Optional[str] = None,
    azure_endpoint:Optional[str] = None,
    azure_deploy:  Optional[str] = None,
    fallback:      bool          = True,
) -> PokerAIClient:
    """
    Crée et configure le client IA depuis main.py.

    Injecte les clés dans les variables d'environnement si fournies,
    puis crée le client avec le provider demandé.

    Usage dans PokerAssistant.__init__() :
        from ai_client import create_ai_client
        self.claude = create_ai_client(
            provider   = args.provider,   # "auto"/"claude"/"openai"/"copilot"
            claude_key = args.api_key,
            openai_key = args.openai_key,
        )
    """
    # Injecter les clés dans l'environnement si fournies
    if claude_key:
        os.environ["ANTHROPIC_API_KEY"]   = claude_key
    if openai_key:
        os.environ["OPENAI_API_KEY"]      = openai_key
    if azure_key:
        os.environ["AZURE_OPENAI_API_KEY"]= azure_key
    if azure_endpoint:
        os.environ["AZURE_OPENAI_ENDPOINT"]    = azure_endpoint
    if azure_deploy:
        os.environ["AZURE_OPENAI_DEPLOYMENT"]  = azure_deploy

    client = PokerAIClient(provider=provider, fallback=fallback)
    log.info(f"Client IA créé — provider actif : {client.active_provider}")
    return client


# ---------------------------------------------------------------------------
# CLI de test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Test client IA poker multi-provider")
    parser.add_argument("--provider", default="auto",
                        choices=["auto", "claude", "openai", "copilot", "heuristic"],
                        help="Provider IA à utiliser")
    parser.add_argument("--claude-key",  default=None)
    parser.add_argument("--openai-key",  default=None)
    parser.add_argument("--azure-key",   default=None)
    parser.add_argument("--azure-url",   default=None)
    parser.add_argument("--azure-deploy",default=None)
    args = parser.parse_args()

    print(f"\nTest PokerAIClient — provider : {args.provider}")
    print("-" * 50)

    client = create_ai_client(
        provider       = args.provider,
        claude_key     = args.claude_key,
        openai_key     = args.openai_key,
        azure_key      = args.azure_key,
        azure_endpoint = args.azure_url,
        azure_deploy   = args.azure_deploy,
    )

    print(f"Provider actif : {client.active_provider}")
    print(f"Backends dispo : {[b.name for b in client._backends]}")

    # Situation de test
    mock_state = {
        "player_cards": ["Ks", "7h"],
        "board_cards":  ["Kd", "7d", "2c"],
        "pot":           120.0,
        "player_stack":  480.0,
        "current_bet":   40.0,
        "num_players":   4,
        "stage":         "flop",
        "position":      "BTN",
    }
    mock_equity = {
        "win_probability":   0.82,
        "hand_class":        "Deux Paires",
        "recommended_action":"RAISE",
        "spr":               4.0,
        "pot_odds":          0.25,
        "mdf":               0.60,
    }

    print("\nEnvoi d'une situation de test…")
    advice = client.get_advice(mock_state, mock_equity)

    print(f"\nRéponse :")
    print(f"  Action      : {advice.recommended_action}")
    print(f"  Sizing      : {advice.recommended_sizing}")
    print(f"  Équité      : {advice.win_probability:.0%}")
    print(f"  Résumé      : {advice.summary}")
    if advice.error:
        print(f"  Erreur      : {advice.error}")
    print(f"  Provider    : {getattr(advice, 'provider', client.active_provider_key)}")
