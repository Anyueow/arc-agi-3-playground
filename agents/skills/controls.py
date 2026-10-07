"""Controls: learn what each action does.

Passive part (ActionModel): after every move, compare objects before/after.
An object that kept its color and size but changed position "moved"; that
gives a displacement (dx, dy) for the action, and its color is a vote for
"this is the thing I control" (the player).

Active part (ProbeControls skill): press each available simple action a few
times so the ActionModel has data. Usually the first thing to do in a new game.
"""

from collections import Counter, defaultdict

from arcengine import GameAction

from ..core.skill import Skill
from ..core.types import Action, Obj, Observation

MAX_OBJECTS_TO_MATCH = 60


class ActionModel:
    def __init__(self) -> None:
        self.moves: dict[GameAction, Counter] = defaultdict(Counter)  # action -> Counter[(dx, dy)]
        self.nothing: Counter = Counter()  # action -> times it changed nothing
        self.player_votes: Counter = Counter()  # color -> times an object of that color moved
        self.last_seen: dict[int, tuple[int, int]] = {}  # color -> where an object of it last moved to

    def learn(self, action: Action, prev: Observation, obs: Observation, perception) -> str | None:
        if action.game_action.is_complex():
            return None
        if prev.key == obs.key:
            self.nothing[action.game_action] += 1
            return "no change"
        moved = _moved_objects(prev.objects, obs.objects)
        if not moved:
            return "changed, nothing moved"
        for color, dx, dy, center in moved:
            self.player_votes[color] += 1
            self.last_seen[color] = center
        _, dx, dy, _ = moved[0]
        self.moves[action.game_action][(dx, dy)] += 1
        return f"color {moved[0][0]} moved by ({dx},{dy})"

    def displacement(self, action: GameAction) -> tuple[int, int] | None:
        if not self.moves[action]:
            return None
        return self.moves[action].most_common(1)[0][0]

    @property
    def player_color(self) -> int | None:
        return self.player_votes.most_common(1)[0][0] if self.player_votes else None

    def player(self, obs: Observation) -> Obj | None:
        color = self.player_color
        if color is None:
            return None
        matches = [o for o in obs.objects if o.color == color]
        if not matches:
            return None
        # Same-colored decorations (e.g. a HUD icon) exist; pick the one nearest where the player last moved.
        x, y = self.last_seen[color]
        return min(matches, key=lambda o: abs(o.center[0] - x) + abs(o.center[1] - y))

    def describe(self, available: list[GameAction]) -> list[str]:
        lines = []
        for action in available:
            if action is GameAction.RESET:
                continue
            if action.is_complex():
                lines.append(f"{action.name}: click at (x, y)")
                continue
            d = self.displacement(action)
            tried = sum(self.moves[action].values()) + self.nothing[action]
            if tried == 0:
                lines.append(f"{action.name}: unknown (never tried)")
            elif d is None:
                lines.append(f"{action.name}: nothing moved in {tried} tries")
            else:
                blocked = self.nothing[action]
                lines.append(f"{action.name}: moves player by dx={d[0]}, dy={d[1]} ({blocked} of {tried} times blocked)")
        return lines


def _moved_objects(before: list[Obj], after: list[Obj]) -> list[tuple[int, int, int, tuple[int, int]]]:
    """Objects with the same color+size in both frames whose center shifted."""
    if len(before) > MAX_OBJECTS_TO_MATCH or len(after) > MAX_OBJECTS_TO_MATCH:
        return []
    after_by_sig = defaultdict(list)
    for o in after:
        after_by_sig[(o.color, o.size)].append(o)
    moved = []
    for o in before:
        candidates = after_by_sig.get((o.color, o.size), [])
        if len(candidates) != 1 or candidates[0].center == o.center:
            continue
        n = candidates[0]
        moved.append((o.color, n.center[0] - o.center[0], n.center[1] - o.center[1], n.center))
    return moved


class ProbeControls(Skill):
    name = "probe_controls"
    description = "Press each keyboard action a few times to learn what it does. Do this first in a new game."
    args = {"times": "presses per action (default 2)"}

    def setup(self, times: int = 2) -> None:
        self.queue: list[GameAction] = []
        self.times = max(1, min(int(times), 5))

    def step(self, obs: Observation) -> Action | None:
        if not self.queue and self.steps == 0:
            simple = [a for a in obs.available if a is not GameAction.RESET and not a.is_complex()]
            self.queue = [a for a in simple for _ in range(self.times)]
        if not self.queue:
            return self.done("; ".join(self.ctx.controls.describe(obs.available)) or "no keyboard actions")
        return Action(self.queue.pop(0), reason="probing controls")
