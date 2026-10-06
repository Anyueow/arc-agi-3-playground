"""Perception: turn raw 64x64 frames into Observations the agent can reason about.

The key job here is producing a *state key*: a hash that is equal whenever the
game is "in the same situation". Raw frames are a bad key because most games
have a HUD (move counter, energy bar) that changes on every action, which would
make every frame look brand new. So we learn which rows/columns behave like a
HUD and mask them out before hashing.
"""

import hashlib
from collections import Counter, deque

import numpy as np
from arcengine import FrameDataRaw, GameAction

from .types import Obj, Observation

MAX_HUD_LINES = 8  # a HUD is a thin strip; anything bigger is probably gameplay
HUD_THRESHOLD = 0.9  # changes on >=90% of actions (not 100%: animations can skip a tick)


class Perception:
    def __init__(self, warmup: int = 10) -> None:
        self.warmup = warmup
        self.hud_rows: frozenset[int] = frozenset()
        self.hud_cols: frozenset[int] = frozenset()
        self._row_changes = np.zeros(64, dtype=int)
        self._col_changes = np.zeros(64, dtype=int)
        self._transitions = 0

    def observe(self, raw: FrameDataRaw) -> Observation:
        grid = (
            np.asarray(raw.frame[-1], dtype=np.int16)
            if raw.frame
            else np.zeros((64, 64), dtype=np.int16)
        )
        return Observation(
            grid=grid,
            state=raw.state,
            levels_completed=raw.levels_completed,
            available=[GameAction.from_id(a) for a in raw.available_actions],
            key=self.key(grid),
        )

    def key(self, grid: np.ndarray) -> str:
        masked = grid.copy()
        if self.hud_rows:
            masked[sorted(self.hud_rows), :] = -1
        if self.hud_cols:
            masked[:, sorted(self.hud_cols)] = -1
        return hashlib.blake2b(masked.tobytes(), digest_size=8).hexdigest()

    def learn_from(self, before: np.ndarray, after: np.ndarray) -> bool:
        """Update the HUD guess from one transition. Returns True if the mask changed.

        Heuristic: a HUD row/column is one that changes on (nearly) every action,
        even actions that do nothing in the game world (e.g. walking into a
        wall). Gameplay rows only change when something actually moves there.
        The mask only ever grows, so one odd frame can't wipe the agent's map.
        """
        changed = before != after
        self._row_changes += changed.any(axis=1)
        self._col_changes += changed.any(axis=0)
        self._transitions += 1
        if self._transitions < self.warmup:
            return False

        new_rows = self.hud_rows | _frequent(self._row_changes, self._transitions)
        new_cols = self.hud_cols | _frequent(self._col_changes, self._transitions)
        if (new_rows, new_cols) == (self.hud_rows, self.hud_cols):
            return False
        self.hud_rows, self.hud_cols = new_rows, new_cols
        return True


def _frequent(counts: np.ndarray, total: int) -> frozenset[int]:
    lines = np.flatnonzero(counts >= HUD_THRESHOLD * total)
    return frozenset(lines.tolist()) if len(lines) <= MAX_HUD_LINES else frozenset()


def find_objects(grid: np.ndarray) -> list[Obj]:
    """Connected components of same color, ignoring the background (most common color)."""
    h, w = grid.shape
    background = Counter(grid.ravel().tolist()).most_common(1)[0][0]
    seen = np.zeros_like(grid, dtype=bool)
    objects: list[Obj] = []
    for y in range(h):
        for x in range(w):
            if seen[y, x] or grid[y, x] == background:
                continue
            color = int(grid[y, x])
            queue, pixels = deque([(y, x)]), []
            seen[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                pixels.append((cy, cx))
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and grid[ny, nx] == color:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            ys, xs = zip(*pixels)
            objects.append(Obj(color, len(pixels), (round(sum(xs) / len(xs)), round(sum(ys) / len(ys)))))
    return objects
