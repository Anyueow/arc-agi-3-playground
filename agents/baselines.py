"""Baseline agents with no LLM, for comparison.

  random    pure random moves and random click pixels: the control
  explorer  always runs the explore skill: systematic, but goal-blind
"""

import random

from arcengine import FrameDataRaw, GameAction, GameState

from .core.base import Agent
from .core.context import Context
from .core.types import Action
from .skills.explore import Explore


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int = 0, **_) -> None:
        self.rng = random.Random(seed)

    def choose_action(self, frame: FrameDataRaw) -> Action:
        if frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return Action(GameAction.RESET, reason="start / game over")
        action = GameAction.from_id(self.rng.choice(frame.available_actions))
        if action.is_complex():
            return Action(action, self.rng.randrange(64), self.rng.randrange(64), reason="random click")
        return Action(action, reason="random")


class ExplorerAgent(Agent):
    name = "explorer"

    def __init__(self, seed: int = 0, **_) -> None:
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        self.ctx = Context(random.Random(self.seed))
        self.skill: Explore | None = None

    def choose_action(self, frame: FrameDataRaw) -> Action:
        ctx = self.ctx
        obs = ctx.perception.observe(frame)
        see = ctx.describe(obs) if self.tracing else {}
        learned = ctx.update(obs)

        if obs.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            action = Action(GameAction.RESET, reason=obs.state.name.lower())
        else:
            if learned["event"] == "level_up":
                self.skill = None
            if self.skill is None:
                self.skill = Explore(ctx, steps=10**9)
            action = self.skill.next_action(obs)
            if action is None:  # map fully explored: wander and let explore retry later
                self.skill = None
                action = ctx.rng.choice(ctx.candidates(obs))
                action = Action(action.game_action, action.x, action.y, reason="fully explored; wandering")

        if self.tracing:
            self.trace = {"see": see, "learn": learned, "decide": {"action": str(action), "reason": action.reason}, "know": ctx.knowledge()}
        ctx.prev_action = action
        return action

    def status(self) -> str:
        return f"{self.ctx.tracker.status()} states={len(self.ctx.graph)}"

    def report(self) -> str:
        return self.ctx.tracker.report()
