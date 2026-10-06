from .base import Agent
from .explorer import ExplorerAgent
from .random_agent import RandomAgent
from .runner import RunResult, run_game

AGENTS: dict[str, type[Agent]] = {a.name: a for a in (RandomAgent, ExplorerAgent)}

__all__ = ["AGENTS", "Agent", "ExplorerAgent", "RandomAgent", "RunResult", "run_game"]
