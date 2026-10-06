"""Tracker: the agent's journal of the whole game so far.

memory.py is a map of *where moves lead* in the current level. The tracker is a
log of *what has happened*: every move, every death, every level cleared. From
that log it derives facts the strategy can act on:

  - which level we're on and how many actions it has cost so far
  - why we die: running out of energy/time, or stepping into hazards
  - how long a life lasts, and how many moves are left in this one
  - whether exploration has stalled (no new states seen for a while)
"""

from dataclasses import dataclass, field

import numpy as np
from arcengine import GameAction, GameState

from .types import Action, Observation

TIMER_TOLERANCE = 3  # lives within this many moves of each other => a fixed energy budget
STALL_WINDOW = 150  # this many actions without a new state => stalled


@dataclass
class Step:
    number: int
    level: int
    action: str
    reason: str
    event: str  # moved | no_change | game_over | reset | level_up
    key: str


@dataclass
class LevelStats:
    level: int
    actions: int = 0
    deaths: int = 0
    resets: int = 0  # voluntary ones, not the RESET after a game over
    life_lengths: list[int] = field(default_factory=list)
    safe_move_deaths: int = 0  # died on a move that had been safe before
    cleared_at: int | None = None  # actions spent on this level when it was cleared

    @property
    def death_cause(self) -> str:
        """"energy" if deaths come from running out of moves, "hazard" if from bad moves.

        Two pieces of evidence for energy: dying on a move that was safe before
        (so the move itself isn't what killed us), or every life lasting the
        same number of moves.
        """
        lives = self.life_lengths
        if not lives:
            return "none"
        if self.safe_move_deaths or (len(lives) >= 2 and max(lives) - min(lives) <= TIMER_TOLERANCE):
            return "energy"
        return "unknown" if len(lives) < 2 else "hazard"

    @property
    def budget(self) -> int | None:
        """Moves we can count on per life (the shortest life so far), if deaths are energy-based."""
        return min(self.life_lengths) if self.death_cause == "energy" else None


class Tracker:
    def __init__(self) -> None:
        self.history: list[Step] = []
        self.levels: list[LevelStats] = [LevelStats(1)]
        self.life_moves = 0
        self.start_grid: np.ndarray | None = None  # first frame of the current life
        self._seen: set[str] = set()
        self._last_new_state = 0

    @property
    def level(self) -> LevelStats:
        return self.levels[-1]

    def begin(self, obs: Observation) -> None:
        self.start_grid = obs.grid
        self._seen.add(obs.key)

    def record(self, prev: Observation, action: Action, obs: Observation, known_safe: bool = False) -> str:
        """Log one transition and return what kind of event it was.

        `known_safe`: the agent had made this exact move from this state before
        without dying. Only matters if this move turns out to be a game over.
        """
        level = self.level
        level.actions += 1

        if action.game_action is GameAction.RESET:
            event = "reset"
            if prev.state is not GameState.GAME_OVER:
                level.resets += 1
            self._new_life(obs)
        elif obs.levels_completed > prev.levels_completed:
            event = "level_up"
            level.cleared_at = level.actions
            self.levels.append(LevelStats(obs.levels_completed + 1))
            self._seen.clear()
            self._new_life(obs)
        elif obs.state is GameState.GAME_OVER:
            event = "game_over"
            level.deaths += 1
            level.safe_move_deaths += known_safe
            level.life_lengths.append(self.life_moves + 1)
        else:
            event = "moved" if obs.key != prev.key else "no_change"
            self.life_moves += 1

        self.history.append(Step(len(self.history) + 1, level.level, str(action), action.reason, event, obs.key))
        if obs.key not in self._seen:
            self._seen.add(obs.key)
            self._last_new_state = len(self.history)
        return event

    def _new_life(self, obs: Observation) -> None:
        self.life_moves = 0
        self.start_grid = obs.grid

    # -- facts for the strategy --------------------------------------------

    def moves_left(self) -> int | None:
        budget = self.level.budget
        return None if budget is None else budget - self.life_moves

    def stalled(self) -> bool:
        return len(self.history) - self._last_new_state > STALL_WINDOW

    def status(self) -> str:
        level = self.level
        budget = level.budget or "?"
        return f"L{level.level} life {self.life_moves}/{budget} deaths={level.deaths}({level.death_cause})"

    def report(self) -> str:
        lines = []
        for level in self.levels:
            result = f"cleared in {level.cleared_at} actions" if level.cleared_at else "not cleared"
            lives = f"lives lasted {sorted(set(level.life_lengths))}" if level.life_lengths else "no deaths"
            lines.append(
                f"level {level.level}: {result} | {level.actions} actions, {level.deaths} deaths "
                f"({level.death_cause}), {level.resets} voluntary resets, {lives}"
            )
        return "\n".join(lines)
