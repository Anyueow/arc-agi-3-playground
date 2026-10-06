"""Run an agent on one game, several, or all of them.

    uv run python run_agent.py                         # explorer on ls20, offline
    uv run python run_agent.py --game all -n 500       # every downloaded game, scoreboard at the end
    uv run python run_agent.py --game ls20,vc33,ft09   # a few games
    uv run python run_agent.py --agent random -n 300
    uv run python run_agent.py -v                      # print every action + why
    uv run python run_agent.py --online                # play via the ARC API (needs ARC_API_KEY)

Download games first with `uv run python fetch_games.py`.
"""

import argparse
import logging

import arc_agi
from arc_agi.base import OperationMode

from agents import AGENTS, run_game


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ls20", help="game id, comma-separated ids, or 'all'")
    parser.add_argument("--agent", default="explorer", choices=AGENTS)
    parser.add_argument("-n", "--max-actions", type=int, default=1000, help="action budget per game")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--online", action="store_true", help="use the ARC API instead of local game files")
    parser.add_argument("--render", action="store_true", help="draw frames in the terminal")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("arc_agi").setLevel(logging.WARNING)
    arc = arc_agi.Arcade(operation_mode=OperationMode.ONLINE if args.online else OperationMode.OFFLINE)
    games = (
        sorted(e.game_id.split("-")[0] for e in arc.get_environments())
        if args.game == "all"
        else args.game.split(",")
    )

    results = {}
    for game in games:
        env = arc.make(game, render_mode="terminal" if args.render else None)
        if env is None:
            print(f"{game}: not found locally; run `uv run python fetch_games.py` (or use --online)")
            continue
        print(f"\n=== {game} ({', '.join(env.environment_info.tags or ['no tags'])}) ===")
        agent = AGENTS[args.agent](seed=args.seed)
        results[game] = (env.environment_info, run_game(env, agent, args.max_actions, verbose=args.verbose))
        print(agent.report())

    print_scoreboard(arc, args.agent, results)


def print_scoreboard(arc, agent_name: str, results: dict) -> None:
    scorecard = arc.get_scorecard()
    scores = {}
    if scorecard is not None:
        for env_scores in scorecard.environments:
            if env_scores.runs:
                scores[env_scores.id.split("-")[0]] = env_scores.runs[-1].score

    print(f"\n{agent_name}: {'game':6s} {'type':15s} {'levels':>7s} {'actions':>8s} {'lvl-1 human':>12s} {'score %':>8s}")
    for game, (info, result) in results.items():
        tags = ",".join(info.tags or ["-"])
        human = (info.baseline_actions or ["?"])[0]
        levels = f"{result.levels_completed}/{result.win_levels}"
        score = scores.get(game, 0.0)
        print(f"{'':{len(agent_name)}s}  {game:6s} {tags:15s} {levels:>7s} {result.actions:>8d} {human:>12} {score:>8.2f}")

    cleared = sum(1 for _, r in results.values() if r.levels_completed > 0)
    total = scorecard.score if scorecard is not None else 0.0
    print(f"\ngames with at least one level cleared: {cleared}/{len(results)}   overall score: {total:.2f}% (average over games)")


if __name__ == "__main__":
    main()
