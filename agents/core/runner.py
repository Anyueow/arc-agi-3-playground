"""The game loop. Deliberately dumb: it moves frames and actions between the
environment and the agent, enforces the action budget, and reports what happened.
"""

from dataclasses import dataclass

from arcengine import GameAction, GameState

from .base import Agent


@dataclass
class RunResult:
    actions: int
    resets: int
    levels_completed: int
    win_levels: int
    final_state: GameState


def run_game(env, agent: Agent, max_actions: int, verbose: bool = False, tracer=None) -> RunResult:
    agent.reset()
    agent.tracing = tracer is not None
    frame = env.reset()
    resets = 0

    for step in range(1, max_actions + 1):
        grid_before = frame.frame[-1] if frame.frame else None
        action = agent.choose_action(frame)
        prev_levels = frame.levels_completed
        frame = env.step(action.game_action, data=action.data or None, reasoning={"why": action.reason})
        if frame is None:
            raise RuntimeError(f"env.step returned nothing for {action}")

        resets += action.game_action is GameAction.RESET
        if tracer is not None:
            tracer.write(
                {
                    "step": step,
                    "action": str(action),
                    "reason": action.reason,
                    **agent.trace,
                    "result": {"state": frame.state.name, "levels_completed": frame.levels_completed},
                },
                grid_before,
            )
        if verbose:
            print(f"{step:4d}  {str(action):18s} {action.reason:32s} {agent.status()}")
        if frame.levels_completed > prev_levels:
            print(f"  >> level {frame.levels_completed}/{frame.win_levels} cleared at action {step}")
        if frame.state is GameState.WIN:
            print(f"  >> game won at action {step}")
            break

    return RunResult(step, resets, frame.levels_completed, frame.win_levels, frame.state)
