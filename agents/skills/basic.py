"""Basic skills: repeat one action, random moves, restart the level."""

from arcengine import GameAction

from ..core.types import Action, Observation
from ..core.skill import Skill


class Repeat(Skill):
    name = "repeat"
    description = "Press one action (e.g. ACTION1) several times in a row."
    args = {"action": "ACTION1..ACTION5 or ACTION7", "times": "default 1"}

    def setup(self, action: str = "ACTION1", times: int = 1) -> None:
        try:
            self.action = GameAction.from_name(str(action))
        except (KeyError, ValueError):
            self.action = None
        self.times = max(1, min(int(times), 50))
        self.start_key = None

    def step(self, obs: Observation) -> Action | None:
        if self.action is None or self.action not in obs.available or self.action.is_complex():
            return self.done(f"can't repeat that action; available: {[a.name for a in obs.available]}")
        if self.steps == 0:
            self.start_key = obs.key
        if self.steps >= self.times:
            moved = "state changed" if obs.key != self.start_key else "nothing changed"
            return self.done(f"pressed {self.action.name} x{self.times}: {moved}")
        return Action(self.action, reason=f"repeat {self.action.name}")


class RandomMoves(Skill):
    name = "random"
    description = "Make random moves. A last resort when nothing else works."
    args = {"steps": "default 20"}

    def setup(self, steps: int = 20) -> None:
        self.budget = max(1, min(int(steps), 200))

    def step(self, obs: Observation) -> Action | None:
        if self.steps >= self.budget:
            return self.done(f"made {self.steps} random moves")
        return self.ctx.rng.choice(self.ctx.candidates(obs))


class Restart(Skill):
    name = "restart"
    description = "Restart the current level (costs 1 action, refills energy, undoes progress in the level)."
    args = {}

    def step(self, obs: Observation) -> Action | None:
        if self.steps >= 1:
            return self.done("level restarted")
        return Action(GameAction.RESET, reason="restart requested by brain")
