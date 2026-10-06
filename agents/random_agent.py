"""Baseline: pick a random available action. Useful as a floor to beat."""

import random

from arcengine import FrameDataRaw, GameAction, GameState

from .base import Agent
from .types import Action


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def choose_action(self, frame: FrameDataRaw) -> Action:
        if frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return Action(GameAction.RESET, reason="start / game over")
        action = GameAction.from_id(self.rng.choice(frame.available_actions))
        if action.is_complex():
            return Action(action, self.rng.randrange(64), self.rng.randrange(64), reason="random click")
        return Action(action, reason="random")
