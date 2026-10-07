"""Click: press ACTION6 on an object (by id) or a point."""

from arcengine import GameAction

from ..core.types import Action, Observation
from ..core.skill import Skill


class Click(Skill):
    name = "click"
    description = "Click an object (by id) or a point (x, y). Only for games with ACTION6. Reports what changed."
    args = {"object_id": "id from the object list (or give x and y)", "x": "x", "y": "y", "times": "default 1"}

    def setup(self, object_id=None, x=None, y=None, times: int = 1) -> None:
        self.point: tuple[int, int] | None = None
        listed = self.ctx.listed_objects
        if object_id is not None and 0 <= int(object_id) < len(listed):
            self.point = listed[int(object_id)].center
        elif x is not None and y is not None:
            self.point = (min(max(int(x), 0), 63), min(max(int(y), 0), 63))
        self.times = max(1, min(int(times), 10))
        self.before = None

    def step(self, obs: Observation) -> Action | None:
        if GameAction.ACTION6 not in obs.available:
            return self.done("clicking (ACTION6) isn't available in this game")
        if self.point is None:
            return self.done("no valid click target given")
        if self.steps == 0:
            self.before = obs.grid
        if self.steps >= self.times:
            changed = int((self.before != obs.grid).sum())
            return self.done(f"clicked {self.point} x{self.times}: {changed} pixels changed")
        return Action(GameAction.ACTION6, *self.point, reason=f"click {self.point}")
