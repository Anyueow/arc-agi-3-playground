"""Prompts for Meta-TTL.

The meta-prompt phi is the thing being learned. The wrappers around it (what
the meta-agent is shown, and the JSON output format) are fixed harness, so
evolution can change *how* the meta-agent adapts but can't break parsing.
"""

# phi_0: deliberately generic, like the paper's one-line seed
# ("analyze the game trajectory and provide feedback").
SEED_META_PROMPT = "Analyze the game trajectory and provide feedback that will help the player do better next time."

META_AGENT_HARNESS = """You are a META-AGENT coaching a game-playing agent across repeated attempts.
Your instructions (follow them):
---
{phi}
---
The player is an LLM that plays an unknown ARC-AGI-3 grid game by choosing skills
(probe_controls, explore, move_toward, click, repeat, random, restart). Each attempt
("episode") ends when it dies, clears the level, or runs out of moves. Your output
becomes the GUIDE section of the player's system prompt for the NEXT episode;
it is the only thing that carries over between episodes.

Reply with ONLY a JSON object: {{"guide": "<the complete new guide, max 1500 characters>"}}"""

META_AGENT_INPUT = """CURRENT GUIDE (used in the last episode):
{rho}

HISTORY OF EPISODES SO FAR (oldest first):
{history}

Write the guide for episode {next_k}."""

PROPOSER_SYSTEM = """You improve META-PROMPTS. A meta-prompt instructs a meta-agent that coaches a
game-playing LLM between attempts at unknown ARC-AGI-3 grid games (rewriting the
player's guide after every episode). A good meta-prompt makes the player improve
quickly from episode to episode: it is scored by W-AUC, a weighted average of
per-episode returns where later episodes count more.

Episode return J: 0.5 + 0.5*efficiency if the level was cleared (efficiency =
(human actions / player actions)^2, capped at 1), otherwise a small credit
(up to 0.2) for exploring new states.

You will see the current meta-prompt and a full session it produced. Diagnose
how the coaching helped or hurt, then write an improved meta-prompt. Prefer
general adaptation strategies (credit assignment, what facts to record, how to
balance exploiting what worked vs. trying one new experiment, concrete action
plans) over facts about one specific game, because it must work on unseen games.

Reply with ONLY a JSON object:
{"analysis": "<what worked and what didn't, 2-5 sentences>",
 "meta_prompt": "<the complete improved meta-prompt, max 2500 characters>"}"""

PROPOSER_INPUT = """CURRENT META-PROMPT:
---
{phi}
---

SESSION ON GAME {game} (W-AUC {wauc:.3f}):
{session}"""
