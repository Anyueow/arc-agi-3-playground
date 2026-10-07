"""Explore: systematic exploration with the state graph.

Each move: try an untried action here; if none, walk (via known moves) to the
nearest state that still has an untried action. Never re-tests a move whose
result is already on the map. Good for "I don't know what to do yet".
"""

from arcengine import GameAction

from ..core.types import Action, Observation
from ..core.skill import Skill


class Explore(Skill):
    name = "explore"
    description = "Systematically try moves never tried before (walks to unexplored areas). Use when unsure what to do."
    args = {"steps": "how many moves to spend (default 30)"}

    def setup(self, steps: int = 30) -> None:
        self.budget = max(1, min(int(steps), 500))
        self.plan: list[tuple[str, Action]] = []
        self.start_states = len(self.ctx.graph)

    def step(self, obs: Observation) -> Action | None:
        if self.steps >= self.budget:
            new = len(self.ctx.graph) - self.start_states
            return self.done(f"explored {self.steps} moves, discovered {new} new states")
        graph, rng = self.ctx.graph, self.ctx.rng
        graph.add_state(obs.key, self.ctx.candidates(obs))

        if self.plan and self.plan[0][0] == obs.key:
            _, action = self.plan.pop(0)
            return _copy(action, "explore: following route")
        self.plan = []

        untried = graph.untried(obs.key)
        if untried:
            return _copy(rng.choice(untried), "explore: untried here")

        route = graph.path_to_frontier(obs.key)
        if route:
            if self._should_restart(obs, len(route)):
                return Action(GameAction.RESET, reason=f"explore: frontier {len(route)} away, out of moves")
            self.plan = route[1:]
            return _copy(route[0][1], f"explore: route to frontier ({len(route)} steps)")

        return self.done(f"nothing left to explore after {self.steps} moves (map fully explored)")

    def _should_restart(self, obs: Observation, route_len: int) -> bool:
        tracker = self.ctx.tracker
        moves_left, budget = tracker.moves_left(), tracker.level.budget
        if moves_left is None or budget is None or route_len < moves_left:
            return False
        start = self.ctx.perception.key(tracker.start_grid)
        if start == obs.key:
            return False
        from_start = self.ctx.graph.path_to_frontier(start)
        return from_start is not None and len(from_start) < budget - 1


def _copy(action: Action, reason: str) -> Action:
    return Action(action.game_action, action.x, action.y, reason=reason)
