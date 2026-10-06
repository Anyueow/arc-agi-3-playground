"""Memory: a graph of what the agent has learned about the current level.

Nodes are state keys (from Perception). Edges are "in state S, action A led to
state S2". Because ARC-AGI-3 games are deterministic and turn-based, this graph
is a learned world model: once an edge is known, the agent never has to try it
again to know what it does, and it can plan routes through known states.
"""

from collections import deque

from .types import Action

DEAD = "<game over>"  # sink node: edges to it mark moves that killed us


class StateGraph:
    def __init__(self) -> None:
        self.edges: dict[str, dict[Action, str]] = {}
        self.candidates: dict[str, list[Action]] = {}

    def __len__(self) -> int:
        return len(self.candidates)

    def add_state(self, key: str, candidates: list[Action]) -> None:
        self.candidates.setdefault(key, candidates)
        self.edges.setdefault(key, {})

    def record(self, key: str, action: Action, next_key: str) -> None:
        self.edges.setdefault(key, {})[action] = next_key

    def forget_edges_to(self, target: str) -> None:
        for edges in self.edges.values():
            for action in [a for a, t in edges.items() if t == target]:
                del edges[action]

    def untried(self, key: str) -> list[Action]:
        tried = self.edges.get(key, {})
        return [a for a in self.candidates.get(key, []) if a not in tried]

    def path_to_frontier(self, start: str) -> list[tuple[str, Action]] | None:
        """BFS over known edges to the nearest state that still has untried actions.

        Returns [(state_key, action), ...]: the moves to make and the state we
        expect to be in before each one, so the caller can detect surprises.
        """
        parents: dict[str, tuple[str, Action] | None] = {start: None}
        queue = deque([start])
        while queue:
            key = queue.popleft()
            if key != start and self.untried(key):
                path = []
                while parents[key] is not None:
                    prev, action = parents[key]
                    path.append((prev, action))
                    key = prev
                return path[::-1]
            for action, nxt in self.edges.get(key, {}).items():
                if nxt not in parents:
                    parents[nxt] = (key, action)
                    queue.append(nxt)
        return None
