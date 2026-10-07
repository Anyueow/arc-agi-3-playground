"""RunTTL (Algorithm 1): K episodes on one game, adapting the guide between them."""

from dataclasses import dataclass, field

from agents.core.ollama import Ollama

from .episode import Episode, run_episode
from .meta_agent import adapt


@dataclass
class Session:
    game: str
    phi: str | None  # None = static: no adaptation between episodes
    episodes: list[Episode] = field(default_factory=list)

    @property
    def wauc(self) -> float:
        """Weighted area under the learning curve, w_k = k, J_max = 1."""
        if not self.episodes:
            return 0.0
        num = sum(e.k * e.ret for e in self.episodes)
        return num / sum(e.k for e in self.episodes)

    def text(self) -> str:
        """Full session for the proposer: each episode, the guide it used, what happened."""
        blocks = []
        for e in self.episodes:
            blocks.append(f"--- guide used in episode {e.k} ---\n{e.guide or '(none)'}\n{e.summary()}")
        return "\n".join(blocks)


def run_ttl(arc, game: str, phi: str | None, actor_model: str, meta_llm: Ollama, episodes: int, max_actions: int, seed: int = 0, log=print) -> Session:
    env = arc.make(game)
    if env is None:
        raise RuntimeError(f"game {game!r} not available locally; run fetch_games.py")
    frame = env.reset()
    session = Session(game, phi)
    rho = ""
    for k in range(1, episodes + 1):
        if session.episodes and session.episodes[-1].outcome == "won_game":
            # Nothing left to play; later episodes count as perfect.
            last = session.episodes[-1]
            session.episodes.append(Episode(k, last.level, "won_game", 0, 0, None, 1.0, rho))
            continue
        episode, frame = run_episode(env, frame, k, rho, actor_model, max_actions, seed + k)
        session.episodes.append(episode)
        log(f"    {game} ep{k}: L{episode.level} {episode.outcome} in {episode.actions} actions, J={episode.ret:.2f}")
        if k < episodes and phi is not None:
            rho = adapt(meta_llm, phi, rho, session.episodes)
    return session
