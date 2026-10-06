"""ExplorerAgent: systematic exploration with a learned state graph.

Each turn runs the same three-stage pipeline every serious agent has:

    1. PERCEIVE  raw frame -> Observation (with a HUD-masked state key)
    2. LEARN     log the previous move in the Tracker (journal) and record
                 where it led in the StateGraph (map)
    3. DECIDE    try something new here, or walk (via known moves) to the
                 nearest state that still has something new to try

The tracker lets DECIDE adapt to what the game has revealed so far:
  - deaths look like hazards -> never repeat a move that killed us
  - deaths look like running out of energy -> don't start a route we can't
    finish this life; restart instead if the frontier is closer from the start

It still has no idea what the goal is. Smarter DECIDE logic (goal hypotheses,
an LLM, a learned model) is where the real competition is.
"""

import random

from arcengine import FrameDataRaw, GameAction, GameState

from .base import Agent
from .memory import DEAD, StateGraph
from .perception import Perception
from .tracker import Tracker
from .types import Action, Observation

MAX_CLICK_TARGETS = 16


class ExplorerAgent(Agent):
    name = "explorer"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        self.rng = random.Random(self.seed)
        self.perception = Perception()
        self.tracker = Tracker()
        self.graph = StateGraph()
        self.plan: list[tuple[str, Action]] = []
        self.prev_obs: Observation | None = None
        self.prev_action: Action | None = None

    def choose_action(self, frame: FrameDataRaw) -> Action:
        obs = self.perception.observe(frame)  # 1. PERCEIVE
        self._learn(obs)  # 2. LEARN
        action = self._decide(obs)  # 3. DECIDE
        self.prev_obs, self.prev_action = obs, action
        return action

    def status(self) -> str:
        return f"{self.tracker.status()} states={len(self.graph)} plan={len(self.plan)}"

    def report(self) -> str:
        return self.tracker.report()

    # -- LEARN -------------------------------------------------------------

    def _learn(self, obs: Observation) -> None:
        prev, action = self.prev_obs, self.prev_action
        if prev is None or action is None:
            self.tracker.begin(obs)
            return

        known_safe = self.graph.edges.get(prev.key, {}).get(action) not in (None, DEAD)
        event = self.tracker.record(prev, action, obs, known_safe)
        if event == "level_up":
            self._forget()  # new level = new world; old map is useless
            return
        if event == "game_over":
            self._learn_from_death(prev, action)
            return
        if event == "reset":
            return  # the level layout (and our map) stays valid
        if self.perception.learn_from(prev.grid, obs.grid):
            # The HUD mask changed, so every key we've stored is stale.
            self._forget()
            obs.key = self.perception.key(obs.grid)
            return
        self.graph.record(prev.key, action, obs.key)

    def _learn_from_death(self, prev: Observation, action: Action) -> None:
        if self.tracker.level.death_cause == "energy":
            # We ran out of energy, not into danger: those moves are safe after all.
            self.graph.forget_edges_to(DEAD)
        else:
            self.graph.record(prev.key, action, DEAD)

    def _forget(self) -> None:
        self.graph = StateGraph()
        self.plan = []

    # -- DECIDE ------------------------------------------------------------

    def _decide(self, obs: Observation) -> Action:
        if obs.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return Action(GameAction.RESET, reason=obs.state.name.lower())

        self.graph.add_state(obs.key, self._candidates(obs))

        # Continue a route we planned earlier, if we're where we expected to be.
        if self.plan and self.plan[0][0] == obs.key:
            _, action = self.plan.pop(0)
            return Action(action.game_action, action.x, action.y, reason="following plan")
        self.plan = []

        untried = self.graph.untried(obs.key)
        if untried:
            action = self.rng.choice(untried)
            return Action(action.game_action, action.x, action.y, reason="untried here")

        route = self.graph.path_to_frontier(obs.key)
        if route:
            if self._should_restart(obs, len(route)):
                return Action(GameAction.RESET, reason=f"frontier {len(route)} away, out of moves")
            self.plan = route[1:]
            action = route[0][1]
            return Action(
                action.game_action, action.x, action.y, reason=f"route to frontier ({len(route)} steps)"
            )

        reason = "stalled; wandering" if self.tracker.stalled() else "fully explored; wandering"
        action = self.rng.choice(self.graph.candidates[obs.key])
        return Action(action.game_action, action.x, action.y, reason=reason)

    def _should_restart(self, obs: Observation, route_len: int) -> bool:
        """Under an energy limit: is the frontier out of reach this life but reachable from the start?"""
        moves_left = self.tracker.moves_left()
        budget = self.tracker.level.budget
        if moves_left is None or budget is None or route_len < moves_left:
            return False
        start = self.perception.key(self.tracker.start_grid)
        if start == obs.key:
            return False
        from_start = self.graph.path_to_frontier(start)
        return from_start is not None and len(from_start) < budget - 1

    def _candidates(self, obs: Observation) -> list[Action]:
        """Every distinct move worth trying in this state."""
        actions = []
        for game_action in obs.available:
            if game_action is GameAction.RESET:
                continue
            if game_action.is_complex():
                # Clicking: try the center of each object, smallest first.
                targets = sorted(obs.objects, key=lambda o: o.size)[:MAX_CLICK_TARGETS]
                actions += [Action(game_action, *o.center) for o in targets]
            else:
                actions.append(Action(game_action))
        return actions
