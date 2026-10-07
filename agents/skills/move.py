"""MoveToward: walk the player toward a target, greedily.

Uses the ActionModel (controls.py) to know which action moves the player in
which direction, and the map to avoid moves known to be blocked. Greedy, so it
can get stuck behind walls; it reports that so the brain can try something else.
"""

from ..core.types import Action, Observation
from ..core.skill import Skill


class MoveToward(Skill):
    name = "move_toward"
    description = (
        "Walk the player toward an object (by id) or a point (x, y). "
        "Needs probe_controls first. Stops on arrival or when stuck."
    )
    args = {"object_id": "id from the object list (or give x and y)", "x": "target x", "y": "target y", "max_steps": "default 40"}

    def setup(self, object_id=None, x=None, y=None, max_steps: int = 40) -> None:
        self.target: tuple[int, int] | None = None
        listed = self.ctx.listed_objects
        if object_id is not None and 0 <= int(object_id) < len(listed):
            self.target = listed[int(object_id)].center
        elif x is not None and y is not None:
            self.target = (int(x), int(y))
        self.max_steps = max(1, min(int(max_steps), 200))
        self.stuck = 0

    def step(self, obs: Observation) -> Action | None:
        if self.target is None:
            return self.done("no valid target given")
        controls = self.ctx.controls
        player = controls.player(obs)
        if player is None:
            return self.done("don't know which object is the player yet; run probe_controls")
        dist = _dist(player.center, self.target)
        if dist <= 3:
            return self.done(f"arrived at {self.target} in {self.steps} moves")
        if self.steps >= self.max_steps:
            return self.done(f"gave up after {self.steps} moves, still {dist} away from {self.target}")

        known = self.ctx.graph.edges.get(obs.key, {})
        options = []
        for game_action in obs.available:
            d = controls.displacement(game_action)
            if d is None or d == (0, 0):
                continue
            action = Action(game_action)
            if known.get(action) == obs.key:
                continue  # known to be blocked from here
            new = (player.center[0] + d[0], player.center[1] + d[1])
            options.append((_dist(new, self.target), self.ctx.rng.random(), action))
        if not options:
            return self.done(f"stuck at {player.center}: every useful direction is blocked")
        best_dist, _, action = min(options)
        self.stuck = self.stuck + 1 if best_dist >= dist else 0
        if self.stuck > 6:
            return self.done(f"stuck near {player.center}, a wall is probably in the way of {self.target}")
        return Action(action.game_action, reason=f"move_toward {self.target}")


def _dist(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
