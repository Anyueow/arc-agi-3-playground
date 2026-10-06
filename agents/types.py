"""Shared data types passed between perception, memory, agents and the runner."""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from arcengine import GameAction, GameState


@dataclass(frozen=True)
class Action:
    """One move: a GameAction, plus x/y when it's an ACTION6 click.

    `reason` is a human-readable note on why the agent chose it. It's excluded
    from equality/hashing so the same move with a different reason is still the
    same move in memory.
    """

    game_action: GameAction
    x: int | None = None
    y: int | None = None
    reason: str = field(default="", compare=False)

    @property
    def data(self) -> dict:
        return {} if self.x is None else {"x": self.x, "y": self.y}

    def __str__(self) -> str:
        name = self.game_action.name
        return name if self.x is None else f"{name}({self.x},{self.y})"


@dataclass(frozen=True)
class Obj:
    """A connected blob of same-colored pixels."""

    color: int
    size: int
    center: tuple[int, int]  # (x, y), i.e. (column, row)


@dataclass
class Observation:
    """What the agent "sees" after perception has processed a raw frame."""

    grid: np.ndarray  # 64x64 color indices (last frame of the step)
    state: GameState
    levels_completed: int
    available: list[GameAction]
    key: str  # hash that identifies this game state (HUD pixels masked out)

    @cached_property
    def objects(self) -> list[Obj]:
        from .perception import find_objects

        return find_objects(self.grid)
