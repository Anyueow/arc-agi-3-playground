"""Skills: everything the brain can use.

Two kinds:

  PASSIVE skills run on every frame, no matter what. They keep the agent's
  working memory (the Context) up to date:
      perceive.py   pixels -> state key + objects, learns the HUD mask
      memory.py     map of the level: state --action--> next state
      journal.py    log of moves/deaths/levels; energy and death-cause facts
      controls.py   what each action does (moves the player by dx,dy? nothing?)

  ACTIVE skills are what the brain chooses between. Each one is started with
  arguments, then emits actions one at a time until it decides it's done, and
  finally reports an outcome string the brain reads before choosing again:
      explore.py    systematic exploration of untried moves
      controls.py   probe_controls: try each action to learn what it does
      move.py       move_toward: walk the player toward a point/object
      click.py      click an object
      basic.py      repeat an action, random moves, restart the level
"""

from ..core.skill import Skill
from .basic import RandomMoves, Repeat, Restart
from .click import Click
from .controls import ProbeControls
from .explore import Explore
from .move import MoveToward

SKILLS: dict[str, type[Skill]] = {
    s.name: s for s in (ProbeControls, Explore, MoveToward, Click, Repeat, RandomMoves, Restart)
}

__all__ = ["SKILLS", "Skill"]
