"""Meta-TTL on ARC-AGI-3: meta-train an adaptation policy, then evaluate it.

    # meta-train phi (outer loop); writes runs/metattl/<time>/best_meta_prompt.txt
    uv run python meta_ttl.py train --iterations 10

    # evaluate on held-out games: static (no adaptation) vs naive (seed phi) vs learned phi*
    uv run python meta_ttl.py eval --phi runs/metattl/<time>/best_meta_prompt.txt

    # tiny smoke test
    uv run python meta_ttl.py train --iterations 1 --episodes 2 --max-actions 40 --train ls20 --val vc33

Games are split by a fixed seed into train / val / test (see `split`).
"""

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from statistics import mean

import arc_agi
from arc_agi.base import OperationMode

from agents.core.ollama import DEFAULT_MODEL, Ollama
from metattl.evolve import Config, MetaTrainer
from metattl.prompts import SEED_META_PROMPT
from metattl.session import run_ttl


def split(games: list[str], n_train: int, n_val: int, seed: int = 0) -> tuple[list[str], list[str], list[str]]:
    shuffled = sorted(games)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "eval", "split"])
    parser.add_argument("--train", help="comma-separated training games (default: from split)")
    parser.add_argument("--val", help="comma-separated validation games (default: from split)")
    parser.add_argument("--test", help="comma-separated test games for eval (default: from split)")
    parser.add_argument("--n-train", type=int, default=6)
    parser.add_argument("--n-val", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10, help="outer-loop budget T")
    parser.add_argument("--episodes", type=int, default=4, help="episodes per TTL session K")
    parser.add_argument("--max-actions", type=int, default=150, help="action budget per episode")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="actor model")
    parser.add_argument("--meta-model", default=None, help="meta-agent model (default: --model)")
    parser.add_argument("--proposer-model", default=None, help="proposer model (default: --meta-model)")
    parser.add_argument("--phi", type=Path, help="learned meta-prompt file, for eval")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("arc_agi").setLevel(logging.WARNING)
    arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    all_games = sorted(e.game_id.split("-")[0] for e in arc.get_environments())
    train, val, test = split(all_games, args.n_train, args.n_val, args.seed)
    train = args.train.split(",") if args.train else train
    val = args.val.split(",") if args.val else val
    test = args.test.split(",") if args.test else test
    meta_model = args.meta_model or args.model
    proposer_model = args.proposer_model or meta_model

    if args.command == "split":
        print(f"train: {train}\nval:   {val}\ntest:  {test}")
        return
    for model in {args.model, meta_model, proposer_model}:
        Ollama(model).check()

    if args.command == "train":
        out = Path("runs/metattl") / datetime.now().strftime("%Y%m%d-%H%M%S")
        cfg = Config(train, val, args.iterations, args.episodes, args.max_actions, args.model, meta_model, proposer_model, args.seed)
        print(f"meta-training: train={train} val={val}; outputs in {out}")
        MetaTrainer(arc, cfg, out).train()
        return

    # eval: the paper's comparison, on held-out games
    learned = args.phi.read_text().strip() if args.phi else None
    methods = {"static": None, "naive": SEED_META_PROMPT}
    if learned:
        methods["meta-ttl"] = learned
    meta_llm = Ollama(meta_model)
    results: dict[str, dict[str, float]] = {m: {} for m in methods}
    for game in test:
        for method, phi in methods.items():
            print(f"[eval] {game} / {method}")
            session = run_ttl(arc, game, phi, args.model, meta_llm, args.episodes, args.max_actions, args.seed)
            results[method][game] = session.wauc
    print(f"\n{'game':6s} " + " ".join(f"{m:>9s}" for m in methods))
    for game in test:
        print(f"{game:6s} " + " ".join(f"{results[m][game]:>9.3f}" for m in methods))
    print(f"{'mean':6s} " + " ".join(f"{mean(results[m].values()):>9.3f}" for m in methods))
    out = Path("runs/metattl") / f"eval-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"test": test, "episodes": args.episodes, "max_actions": args.max_actions, "wauc": results}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
