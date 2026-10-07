"""Base class for active skills."""

from typing import TYPE_CHECKING

from .types import Action, Observation

if TYPE_CHECKING:
    from .context import Context


class Skill:
    """An active skill. Subclasses set name/description/args and implement step()."""

    name = "skill"
    description = ""
    args: dict[str, str] = {}  # arg name -> description, shown to the brain

    def __init__(self, ctx: "Context", **kwargs) -> None:
        self.ctx = ctx
        self.steps = 0
        self.outcome = ""
        self.setup(**kwargs)

    def setup(self, **kwargs) -> None:
        pass

    def next_action(self, obs: Observation) -> Action | None:
        """The next move, or None when the skill is finished (set self.outcome first)."""
        action = self.step(obs)
        if action is not None:
            self.steps += 1
        return action

    def step(self, obs: Observation) -> Action | None:
        raise NotImplementedError

    def done(self, outcome: str) -> None:
        self.outcome = outcome
        return None
