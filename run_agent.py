"""Run an agent on one game, several, or all of them.

    uv run python run_agent.py                         # explorer on ls20, offline
    uv run python run_agent.py --game all -n 500       # every downloaded game, scoreboard at the end
    uv run python run_agent.py --game ls20,vc33,ft09   # a few games
    uv run python run_agent.py --agent random -n 300
    uv run python run_agent.py -v                      # print every action + why
    uv run python run_agent.py --trace -n 300          # full see/learn/decide logs in logs/
    uv run python run_agent.py --online                # play via the ARC API (needs ARC_API_KEY)

Controls and comparisons:

    # build the random control (stashed in the repo)
    uv run python run_agent.py --agent random --game all -n 10000 --seeds 3 --save baselines/random_control.json
    # compare any run against it
    uv run python run_agent.py --game all -n 500 --compare baselines/random_control.json

Download games first with `uv run python fetch_games.py`.
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import arc_agi
from arc_agi.base import OperationMode

from agents import AGENTS, run_game
from agents.trace import Tracer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ls20", help="game id, comma-separated ids, or 'all'")
    parser.add_argument("--agent", default="explorer", choices=AGENTS)
    parser.add_argument("-n", "--max-actions", type=int, default=1000, help="action budget per game")
    parser.add_argument("--seed", type=int, default=0, help="first seed")
    parser.add_argument("--seeds", type=int, default=1, help="runs per game, with seeds seed..seed+N-1")
    parser.add_argument("--save", type=Path, help="write results as JSON (e.g. baselines/random_control.json)")
    parser.add_argument("--compare", type=Path, help="results JSON to show alongside, e.g. the random control")
    parser.add_argument("--online", action="store_true", help="use the ARC API instead of local game files")
    parser.add_argument("--render", action="store_true", help="draw frames in the terminal")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--trace", action="store_true", help="write step-by-step logs to logs/ (see/learn/decide)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("arc_agi").setLevel(logging.WARNING)
    arc = arc_agi.Arcade(operation_mode=OperationMode.ONLINE if args.online else OperationMode.OFFLINE)
    games = (
        sorted(e.game_id.split("-")[0] for e in arc.get_environments())
        if args.game == "all"
        else args.game.split(",")
    )
    seeds = list(range(args.seed, args.seed + args.seeds))

    results = {
        "agent": args.agent,
        "max_actions": args.max_actions,
        "seeds": seeds,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "games": {},
    }
    for game in games:
        runs = []
        for seed in seeds:
            env = arc.make(game, render_mode="terminal" if args.render else None)
            if env is None:
                print(f"{game}: not found locally; run `uv run python fetch_games.py` (or use --online)")
                break
            info = env.environment_info
            print(f"\n=== {game} ({', '.join(info.tags or ['no tags'])}) seed {seed} ===")
            agent = AGENTS[args.agent](seed=seed)
            tracer = Tracer(game, args.agent) if args.trace else None
            result = run_game(env, agent, args.max_actions, verbose=args.verbose, tracer=tracer)
            if tracer:
                tracer.close()
                print(f"trace: {tracer.text_path}  (+ .jsonl)")
            if report := agent.report():
                print(report)
            runs.append({"seed": seed, **_run_record(arc, game, result)})
        if runs:
            results["games"][game] = {
                "tags": info.tags or [],
                "win_levels": runs[0]["win_levels"],
                "human_baseline_actions": info.baseline_actions or [],
                "mean_score": mean(r["score"] for r in runs),
                "mean_levels": mean(r["levels_completed"] for r in runs),
                "runs": runs,
            }

    scored = results["games"].values()
    results["summary"] = {
        "overall_score": mean(g["mean_score"] for g in scored) if scored else 0.0,
        "games_with_progress": sum(1 for g in scored if g["mean_levels"] > 0),
        "games": len(results["games"]),
    }

    control = json.loads(args.compare.read_text()) if args.compare else None
    print_scoreboard(results, control)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nsaved results to {args.save}")


def _run_record(arc, game: str, result) -> dict:
    """One run's numbers, with the official score taken from the scorecard."""
    record = {
        "levels_completed": result.levels_completed,
        "win_levels": result.win_levels,
        "actions": result.actions,
        "resets": result.resets,
        "final_state": result.final_state.name,
        "score": 0.0,
        "level_actions": [],
    }
    scorecard = arc.get_scorecard()
    for env_scores in scorecard.environments if scorecard else []:
        if env_scores.id.split("-")[0] == game and env_scores.runs:
            latest = env_scores.runs[-1]
            record["score"] = latest.score
            record["level_actions"] = latest.level_actions or []
    return record


def print_scoreboard(results: dict, control: dict | None) -> None:
    n_seeds = len(results["seeds"])
    header = f"{'game':6s} {'type':15s} {'levels':>7s} {'lvl-1 human':>12s} {'score %':>8s}"
    if control:
        header += f" {'control %':>10s} {'diff':>7s}"
    print(f"\n{results['agent']} ({results['max_actions']} actions/game, {n_seeds} seed(s); levels = mean)")
    print(header)

    for game, g in results["games"].items():
        tags = ",".join(g["tags"] or ["-"])
        human = (g["human_baseline_actions"] or ["?"])[0]
        levels = f"{g['mean_levels']:.1f}/{g['win_levels']}" if n_seeds > 1 else f"{g['mean_levels']}/{g['win_levels']}"
        line = f"{game:6s} {tags:15s} {levels:>7s} {human:>12} {g['mean_score']:>8.2f}"
        if control:
            ref = control["games"].get(game)
            line += f" {ref['mean_score']:>10.2f} {g['mean_score'] - ref['mean_score']:>+7.2f}" if ref else f" {'-':>10s}"
        print(line)

    s = results["summary"]
    print(
        f"\ngames with progress: {s['games_with_progress']}/{s['games']}   "
        f"overall score: {s['overall_score']:.3f}% (mean over games)"
    )
    if control:
        c = control["summary"]
        print(
            f"control ({control['agent']}, {control['max_actions']} actions x {len(control['seeds'])} seeds): "
            f"{c['games_with_progress']}/{c['games']} games with progress, overall {c['overall_score']:.3f}%"
        )


if __name__ == "__main__":
    main()
