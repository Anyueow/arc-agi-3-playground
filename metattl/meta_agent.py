"""The meta-agent f_phi: rewrites the actor's guide rho after each episode (Eq. 3)."""

from agents.core.ollama import Ollama

from .episode import Episode
from .prompts import META_AGENT_HARNESS, META_AGENT_INPUT

MAX_GUIDE_CHARS = 1500
DETAILED_EPISODES = 2  # older episodes are shown as one-line summaries to fit the context


def adapt(llm: Ollama, phi: str, rho: str, history: list[Episode]) -> str:
    """rho_{k+1} ~ f_phi(. | rho_k, H_k). Keeps the old guide if the reply is unusable."""
    shown = [e.summary(detail=i >= len(history) - DETAILED_EPISODES) for i, e in enumerate(history)]
    user = META_AGENT_INPUT.format(rho=rho or "(empty: first attempt had no guide)", history="\n".join(shown), next_k=len(history) + 1)
    try:
        reply, _ = llm.chat_json(META_AGENT_HARNESS.format(phi=phi), user)
    except Exception:
        return rho
    guide = str(reply.get("guide", "")).strip()
    return guide[:MAX_GUIDE_CHARS] if guide else rho
