"""Benchmark Ollama models (and the no-LLM baselines) on the same games and budget.

    uv run python benchmark.py                                  # models in benchmarks/models.txt
    uv run python benchmark.py --models llama3.1:8b,mistral:latest --games ls20,vc33,r11l -n 200
    uv run python benchmark.py --baselines                      # also run explorer (random control already exists)

Each run is saved to benchmarks/results/<label>.json (and traces to logs/ with --trace),
which `uv run streamlit run dashboard.py` picks up automatically.
"""

import argparse
import subprocess
import sys
from pathlib import Path

MODELS_FILE = Path("benchmarks/models.txt")
RESULTS_DIR = Path("benchmarks/results")


def read_models() -> list[str]:
    lines = MODELS_FILE.read_text().splitlines() if MODELS_FILE.exists() else []
    return [line.split("#")[0].strip() for line in lines if line.split("#")[0].strip()]


def run(agent: str, label: str, games: str, actions: int, seeds: int, trace: bool, model: str | None = None) -> None:
    out = RESULTS_DIR / f"{label.replace(':', '_').replace('/', '_')}.json"
    cmd = [sys.executable, "run_agent.py", "--agent", agent, "--game", games, "-n", str(actions),
           "--seeds", str(seeds), "--label", label, "--save", str(out),
           "--compare", "baselines/random_control.json"]
    if model:
        cmd += ["--model", model]
    if trace:
        cmd.append("--trace")
    print(f"\n##### {label} #####\n{' '.join(cmd)}")
    subprocess.run(cmd, check=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="comma-separated Ollama models (default: benchmarks/models.txt)")
    parser.add_argument("--games", default="ls20,vc33,r11l,sp80,tu93", help="games, or 'all' (LLM runs are slow)")
    parser.add_argument("-n", "--max-actions", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--baselines", action="store_true", help="also benchmark the explorer agent")
    parser.add_argument("--no-trace", action="store_true", help="skip step-by-step trace logs")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    models = args.models.split(",") if args.models else read_models()
    trace = not args.no_trace
    if args.baselines:
        run("explorer", "explorer", args.games, args.max_actions, args.seeds, trace)
    for model in models:
        run("llm", f"llm:{model}", args.games, args.max_actions, args.seeds, trace, model)


if __name__ == "__main__":
    main()
