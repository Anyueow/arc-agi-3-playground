"""RunActor: one episode = one attempt at the current level."""

from dataclasses import dataclass, field

from arcengine import GameAction, GameState

from agents import LLMAgent

EXPLORATION_CREDIT = 0.2  # max return for an uncleared episode
EXPLORATION_SCALE = 40  # new states for full exploration credit


@dataclass
class Episode:
    k: int
    level: int
    outcome: str  # cleared | died | out_of_moves | won_game
    actions: int
    states_seen: int
    human_actions: int | None
    ret: float  # J(tau) in [0, 1]
    guide: str
    log: list[str] = field(default_factory=list)  # skill-level history
    thoughts: list[str] = field(default_factory=list)
    controls: list[str] = field(default_factory=list)

    def summary(self, detail: bool = True) -> str:
        head = (
            f"Episode {self.k}: level {self.level}, {self.outcome} after {self.actions} actions "
            f"(human baseline {self.human_actions}), {self.states_seen} states seen, return {self.ret:.2f}"
        )
        if not detail:
            return head
        parts = [head, "  controls learned: " + ("; ".join(self.controls) or "none")]
        parts += ["  " + line for line in self.log[-14:]]
        if self.thoughts:
            parts.append("  last thoughts: " + " | ".join(t[:160] for t in self.thoughts[-3:]))
        return "\n".join(parts)


def episode_return(cleared: bool, actions: int, human: int | None, states_seen: int) -> float:
    if cleared:
        efficiency = min(1.0, (human / max(actions, 1)) ** 2) if human else 0.0
        return 0.5 + 0.5 * efficiency
    return EXPLORATION_CREDIT * min(1.0, states_seen / EXPLORATION_SCALE)


def run_episode(env, frame, k: int, guide: str, model: str, max_actions: int, seed: int):
    """Play one attempt with a fresh actor whose only carry-over is `guide`.

    Returns (Episode, frame after the episode). If the level wasn't cleared,
    the level is restarted so the next episode begins from its initial state.
    """
    agent = LLMAgent(seed=seed, model=model, guide=guide, echo=False)
    start_level = frame.levels_completed
    baselines = env.environment_info.baseline_actions or []
    human = baselines[start_level] if start_level < len(baselines) else None

    outcome, actions = "out_of_moves", 0
    for actions in range(1, max_actions + 1):
        action = agent.choose_action(frame)
        frame = env.step(action.game_action, data=action.data or None, reasoning={"why": action.reason})
        if frame.state is GameState.WIN:
            outcome = "won_game"
            break
        if frame.levels_completed > start_level:
            outcome = "cleared"
            break
        if frame.state is GameState.GAME_OVER:
            outcome = "died"
            break
    if agent.skill is not None:
        agent._finish(f"episode ended: {outcome}")

    cleared = outcome in ("cleared", "won_game")
    states = len(agent.ctx.graph)
    episode = Episode(
        k=k,
        level=start_level + 1,
        outcome=outcome,
        actions=actions,
        states_seen=states,
        human_actions=human,
        ret=episode_return(cleared, actions, human, states),
        guide=guide,
        log=list(agent.history),
        thoughts=[t["thought"] for t in agent.thoughts if t.get("thought")],
        controls=agent.ctx.controls.describe(agent.ctx.obs.available) if agent.ctx.obs else [],
    )
    if not cleared and frame.state is not GameState.WIN:
        frame = env.step(GameAction.RESET)  # restart the level for the next attempt
    return episode, frame
