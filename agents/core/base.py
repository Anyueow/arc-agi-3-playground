"""The Agent interface. Every agent, however clever, is just this:

    frame in  ->  choose_action()  ->  action out

The runner owns the game loop; the agent owns everything about deciding.
"""

from abc import ABC, abstractmethod

from arcengine import FrameDataRaw

from .types import Action


class Agent(ABC):
    name = "agent"
    tracing = False  # set by the runner when a trace log is being written
    trace: dict = {}  # filled by choose_action when tracing: what the agent saw, learned, decided

    def reset(self) -> None:
        """Called once before a game starts. Clear any per-game state here."""

    @abstractmethod
    def choose_action(self, frame: FrameDataRaw) -> Action:
        """Look at the latest frame (and whatever you remember) and pick a move."""

    def status(self) -> str:
        """One short line about internal state, shown in verbose logs."""
        return ""

    def report(self) -> str:
        """End-of-game summary of what the agent learned."""
        return ""
