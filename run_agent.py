"""Run an agent on a game.

    uv run python run_agent.py                       # explorer on ls20, offline
    uv run python run_agent.py --agent random -n 300
    uv run python run_agent.py -v                    # print every action + why
    uv run python run_agent.py --online              # play via the ARC API (needs ARC_API_KEY)
"""

import argparse
import logging

import arc_agi
from arc_agi.base import OperationMode

from agents import AGENTS, run_game


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--agent", default="explorer", choices=AGENTS)
    parser.add_argument("-n", "--max-actions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--online", action="store_true", help="use the ARC API instead of local game files")
    parser.add_argument("--render", action="store_true", help="draw frames in the terminal")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    arc = arc_agi.Arcade(operation_mode=OperationMode.ONLINE if args.online else OperationMode.OFFLINE)
    env = arc.make(args.game, render_mode="terminal" if args.render else None)
    if env is None:
        raise SystemExit(f"Game {args.game!r} not found locally. Run once with --online to download it.")

    agent = AGENTS[args.agent](seed=args.seed)
    result = run_game(env, agent, args.max_actions, verbose=args.verbose)

    baseline = (env.environment_info.baseline_actions or [None])[0]
    print(
        f"\n{agent.name} on {args.game}: {result.levels_completed}/{result.win_levels} levels, "
        f"{result.actions} actions ({result.resets} resets), final state {result.final_state.name}"
    )
    print(f"human baseline for level 1: {baseline} actions")
    if report := agent.report():
        print(f"\n{report}")
    scorecard = arc.get_scorecard()
    if scorecard is not None:
        print(f"scorecard score: {scorecard.score}")


if __name__ == "__main__":
    main()
