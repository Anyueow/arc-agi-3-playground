"""THE BRAIN: an LLM (via Ollama) that plays by choosing skills.

Every frame, the passive skills (perceive, memory, journal, controls) update
the shared Context. When no skill is running, or something important happens
(death, level cleared), the brain is asked to think: it gets a text summary of
the situation and replies with JSON naming the next skill and its arguments.
That skill then emits moves until it finishes and reports an outcome, and the
brain thinks again.

    frame -> passive skills -> [running skill has a move?] -> action
                                    | no
                                    v
                        BRAIN (LLM): situation -> {"skill", "args"}

`guide` is the editable part of the brain's system prompt. In Meta-TTL
(metattl/) the meta-agent rewrites it between episodes; that's the actor
prompt rho in the paper.
"""

import json
import random

from arcengine import FrameDataRaw, GameAction, GameState

from .baselines import ExplorerAgent, RandomAgent
from .core.base import Agent
from .core.context import Context
from .core.ollama import DEFAULT_MODEL, Ollama
from .core.skill import Skill
from .core.types import Action, Obj, Observation
from .skills import SKILLS
from .skills.explore import Explore

MAX_LISTED_OBJECTS = 25
LARGE_OBJECT = 400  # walls/floors: counted but not listed individually
HISTORY_LINES = 10

SYSTEM_PROMPT = """You are the brain of an agent playing an unknown turn-based video game (ARC-AGI-3).
Nobody tells you the rules or the goal. Discover them by experimenting, then clear each
level using as FEW actions as possible: you are scored on efficiency against humans.

Facts about these games:
- The screen is a 64x64 grid of colors 0-15. You see a list of objects (same-color blobs), not pixels.
- ACTION1-ACTION5 and ACTION7 are buttons (often ACTION1-4 = up/down/left/right). ACTION6 is a click at (x, y).
- Clearing a level is the only reward. Some games have an energy/move limit; running out means game over and the level restarts.
- Common goals: move the player onto something, make shapes/colors match a target, collect items, click things in the right order.
- Objects with a unique color or that look like an icon or target are often important.

You act through SKILLS. Each runs for one or more moves and then reports its outcome to you.
{skills}

Reply with ONLY a JSON object:
{{"thought": "what you notice and believe (1-3 sentences)",
 "notes": "your hypotheses about the rules and goal, carried to your next turn (max 400 chars)",
 "skill": "<skill name>",
 "args": {{...}}}}
{guide}"""


class LLMAgent(Agent):
    name = "llm"

    def __init__(self, seed: int = 0, model: str = DEFAULT_MODEL, guide: str = "", max_thinks: int = 300, echo: bool = True, **_) -> None:
        self.seed = seed
        self.llm = Ollama(model)
        self.guide = guide
        self.max_thinks = max_thinks
        self.echo = echo
        self.reset()

    def reset(self) -> None:
        self.ctx = Context(random.Random(self.seed))
        self.skill: Skill | None = None
        self.skill_label = ""
        self.history: list[str] = []
        self.thoughts: list[dict] = []  # every brain call, kept for logs and Meta-TTL
        self.notes = ""
        self.thinks = 0

    # -- the main loop -------------------------------------------------------

    def choose_action(self, frame: FrameDataRaw) -> Action:
        ctx = self.ctx
        obs = ctx.perception.observe(frame)
        see = ctx.describe(obs) if self.tracing else {}
        learned = ctx.update(obs)
        think = None

        if learned["event"] == "level_up":
            self._finish(f"interrupted: LEVEL {obs.levels_completed} CLEARED")
            self.history.append(f"*** level {obs.levels_completed} cleared; now on level {obs.levels_completed + 1}")
        if obs.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if obs.state is GameState.GAME_OVER:
                self._finish(f"interrupted: GAME OVER ({ctx.tracker.level.death_cause})")
                self.history.append(f"*** game over after {ctx.tracker.life_moves + 1} moves this life; level restarts")
            action = Action(GameAction.RESET, reason=obs.state.name.lower())
        else:
            action = None
            for _ in range(3):
                if self.skill is None:
                    self.skill, think = self._think(obs)
                action = self.skill.next_action(obs)
                if action is not None:
                    break
                self._finish(self.skill.outcome)
            if action is None:
                action = ctx.rng.choice(ctx.candidates(obs))
                action = Action(action.game_action, action.x, action.y, reason="no skill produced a move; random")

        if self.tracing:
            self.trace = {
                "see": see,
                "learn": learned,
                "decide": {"action": str(action), "reason": action.reason, "skill": self.skill_label},
                "know": ctx.knowledge(),
            }
            if think:
                self.trace["think"] = think
        ctx.prev_action = action
        return action

    def _finish(self, outcome: str) -> None:
        if self.skill is not None:
            self.history.append(f"{self.skill_label} -> {outcome} [{self.skill.steps} moves]")
            self.skill = None

    # -- thinking ----------------------------------------------------------

    def _think(self, obs: Observation) -> tuple[Skill, dict]:
        if self.thinks >= self.max_thinks:
            self.skill_label = "explore(steps=50) [think budget used up]"
            return Explore(self.ctx, steps=50), {"note": "think budget used up"}
        self.thinks += 1

        situation = self.situation(obs)
        try:
            reply, raw = self.llm.chat_json(self.system_prompt(), situation)
        except Exception as e:  # network/model errors shouldn't kill the game
            reply, raw = {}, f"LLM error: {e}"

        name = str(reply.get("skill", "")).strip()
        args = reply.get("args") if isinstance(reply.get("args"), dict) else {}
        if reply.get("notes"):
            self.notes = str(reply["notes"])[:600]
        skill_cls = SKILLS.get(name)
        error = None
        if skill_cls is None:
            error = f"unknown skill {name!r}"
        else:
            args = {k: v for k, v in args.items() if k in skill_cls.args}
            try:
                skill = skill_cls(self.ctx, **args)
            except (TypeError, ValueError) as e:
                error = f"bad args for {name}: {e}"
        if error:
            skill_cls, args = Explore, {"steps": 20}
            skill = Explore(self.ctx, steps=20)
            self.history.append(f"(brain reply unusable: {error}; fell back to explore)")

        self.skill_label = f"{skill_cls.name}({', '.join(f'{k}={v}' for k, v in args.items())})"
        think = {
            "thought": str(reply.get("thought", ""))[:500],
            "notes": self.notes,
            "skill": self.skill_label,
            "error": error,
            "situation": situation,
            "raw": raw[:1500],
        }
        self.thoughts.append(think)
        if self.echo:
            print(f"  [brain #{self.thinks}] {self.skill_label}  <- {think['thought'][:140]}")
        return skill, think

    def system_prompt(self) -> str:
        skills = "\n".join(
            f"- {s.name}({', '.join(f'{k}: {v}' for k, v in s.args.items())}): {s.description}" for s in SKILLS.values()
        )
        guide = f"\nGUIDE FROM YOUR COACH (lessons from earlier attempts; follow it):\n{self.guide}\n" if self.guide else ""
        return SYSTEM_PROMPT.format(skills=skills, guide=guide)

    def situation(self, obs: Observation) -> str:
        ctx = self.ctx
        tracker, level = ctx.tracker, ctx.tracker.level
        lines = [f"LEVEL {level.level} of {obs.win_levels}. Actions spent on this level: {level.actions}."]

        lines.append("CONTROLS (learned so far):")
        lines += [f"  {line}" for line in ctx.controls.describe(obs.available)]

        player = ctx.controls.player(obs)
        lines.append(f"PLAYER: color {player.color} at {player.center}" if player else "PLAYER: unknown (use probe_controls)")

        moves_left = tracker.moves_left()
        lines.append(
            f"LIVES: {level.deaths} deaths on this level (cause: {level.death_cause}); "
            f"{tracker.life_moves} moves this life" + (f", about {moves_left} left" if moves_left is not None else "")
        )
        frontier = sum(1 for k in ctx.graph.candidates if ctx.graph.untried(k))
        lines.append(
            f"MAP: {len(ctx.graph)} distinct states seen, {frontier} with untried moves"
            + (", exploration has STALLED" if tracker.stalled() else "")
        )

        lines.append("OBJECTS (id: color, size in px, center (x,y)):")
        lines += [f"  {line}" for line in self._list_objects(obs, player)]

        if self.notes:
            lines.append(f"YOUR NOTES: {self.notes}")
        if self.history:
            lines.append("RECENT (oldest first):")
            lines += [f"  {h}" for h in self.history[-HISTORY_LINES:]]
        lines.append("Choose the next skill.")
        return "\n".join(lines)

    def _list_objects(self, obs: Observation, player: Obj | None) -> list[str]:
        objects = obs.objects
        large = [o for o in objects if o.size > LARGE_OBJECT]
        small = sorted((o for o in objects if o.size <= LARGE_OBJECT), key=lambda o: o.size)[:MAX_LISTED_OBJECTS]
        self.ctx.listed_objects = small
        color_counts: dict[int, int] = {}
        for o in objects:
            color_counts[o.color] = color_counts.get(o.color, 0) + 1
        lines = []
        for i, o in enumerate(small):
            tags = []
            if player is not None and o.color == player.color:
                tags.append("player color")
            if color_counts[o.color] == 1:
                tags.append("unique color")
            lines.append(f"{i}: color {o.color}, {o.size}px at {o.center}" + (f" ({', '.join(tags)})" if tags else ""))
        if large:
            lines.append(f"(+{len(large)} large regions, e.g. walls/floor, colors {sorted({o.color for o in large})})")
        return lines or ["(none)"]

    # -- reporting ---------------------------------------------------------

    def status(self) -> str:
        return f"{self.ctx.tracker.status()} skill={self.skill_label}"

    def report(self) -> str:
        return (
            self.ctx.tracker.report()
            + f"\nbrain: {self.llm.calls} LLM calls, {self.llm.seconds:.0f}s thinking ({self.llm.model})"
        )


AGENTS: dict[str, type[Agent]] = {a.name: a for a in (LLMAgent, ExplorerAgent, RandomAgent)}

__all__ = ["AGENTS", "Agent", "ExplorerAgent", "LLMAgent", "RandomAgent", "SKILLS"]
