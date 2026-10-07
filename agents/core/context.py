"""Context: the agent's working memory, shared by the brain and every skill.

`update()` runs the passive skills (perceive, memory, journal, controls) on
each new frame.
"""

import random
from dataclasses import dataclass, field

from arcengine import GameAction

from ..skills.controls import ActionModel
from ..skills.journal import Tracker
from ..skills.memory import DEAD, StateGraph
from ..skills.perceive import Perception
from .types import Action, Obj, Observation


@dataclass
class Context:
    """The agent's working memory, shared by every skill."""

    rng: random.Random
    perception: Perception = field(default_factory=Perception)
    graph: StateGraph = field(default_factory=StateGraph)
    tracker: Tracker = field(default_factory=Tracker)
    controls: ActionModel = field(default_factory=ActionModel)
    obs: Observation | None = None
    prev_obs: Observation | None = None
    prev_action: Action | None = None
    listed_objects: list[Obj] = field(default_factory=list)  # ids the brain was shown

    def update(self, obs: Observation) -> dict:
        """Run the passive skills on a new frame. Returns what was learned (for the trace)."""
        self.prev_obs, self.obs = self.obs, obs
        prev, action = self.prev_obs, self.prev_action
        if prev is None or action is None:
            self.tracker.begin(obs)
            return {"event": "start"}

        known = self.graph.edges.get(prev.key, {}).get(action)
        event = self.tracker.record(prev, action, obs, known_safe=known not in (None, DEAD))
        learned = {"event": event, "move": f"{prev.key} --{action}--> {obs.key}"}
        if event == "level_up":
            self.graph = StateGraph()  # new level = new world; old map is useless
            return learned | {"note": "new level: map wiped"}
        if event == "game_over":
            if self.tracker.level.death_cause == "energy":
                self.graph.forget_edges_to(DEAD)  # ran out of energy; those moves were fine
            else:
                self.graph.record(prev.key, action, DEAD)
            return learned | {"known_safe_move": known not in (None, DEAD), "death_cause": self.tracker.level.death_cause}
        if event == "reset":
            return learned
        if self.perception.learn_from(prev.grid, obs.grid):
            self.graph = StateGraph()  # HUD mask changed: stored keys are stale
            obs.key = self.perception.key(obs.grid)
            return learned | {"note": f"HUD rows now {sorted(self.perception.hud_rows)}: map wiped"}

        self.graph.record(prev.key, action, obs.key)
        if (effect := self.controls.learn(action, prev, obs, self.perception)) is not None:
            learned["effect"] = effect
        if known not in (None, DEAD, obs.key):
            learned["contradiction"] = f"same move used to lead to {known}"
        return learned

    def candidates(self, obs: Observation) -> list[Action]:
        """Every distinct move worth trying in this state (clicks: object centers)."""
        actions = []
        for game_action in obs.available:
            if game_action is GameAction.RESET:
                continue
            if game_action.is_complex():
                targets = sorted(obs.objects, key=lambda o: o.size)[:16]
                actions += [Action(game_action, *o.center) for o in targets]
            else:
                actions.append(Action(game_action))
        return actions

    # -- snapshots for trace logs -----------------------------------------

    def describe(self, obs: Observation) -> dict:
        """What perception produced this turn."""
        see = {
            "state": obs.state.name,
            "level": obs.levels_completed + 1,
            "key": obs.key,
            "seen_before": obs.key in self.graph.candidates,
            "available": [a.name for a in obs.available],
            "object_count": len(obs.objects),
        }
        if self.prev_obs is not None:
            changed = self.prev_obs.grid != obs.grid
            see["pixels_changed"] = int(changed.sum())
            if changed.any():
                ys, xs = changed.nonzero()
                see["change_box"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        return see

    def knowledge(self) -> dict:
        """What the agent currently believes."""
        player = self.controls.player(self.obs) if self.obs is not None else None
        return {
            "states": len(self.graph),
            "hud_rows": sorted(self.perception.hud_rows),
            "life_moves": self.tracker.life_moves,
            "moves_left": self.tracker.moves_left(),
            "death_cause": self.tracker.level.death_cause,
            "stalled": self.tracker.stalled(),
            "player": None if player is None else {"color": player.color, "at": player.center},
        }
