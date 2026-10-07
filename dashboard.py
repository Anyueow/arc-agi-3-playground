"""Interactive dashboard for agent runs, traces and Meta-TTL training.

    uv run streamlit run dashboard.py

Reads (nothing is written):
  baselines/*.json, benchmarks/results/*.json   results from run_agent.py --save / benchmark.py
  logs/*.jsonl, examples/traces/*.jsonl          step-by-step traces from --trace
  runs/metattl/, examples/metattl/               Meta-TTL training runs and evals

logs/ and runs/ are git-ignored; examples/ holds a few committed samples so a
cloud deployment has something to show in every tab.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PALETTE = np.array([
    [255, 255, 255], [204, 204, 204], [153, 153, 153], [102, 102, 102], [51, 51, 51], [0, 0, 0],
    [229, 58, 163], [255, 123, 204], [249, 60, 49], [30, 147, 255], [136, 216, 241], [255, 220, 0],
    [255, 133, 27], [146, 18, 49], [79, 204, 48], [163, 86, 214],
], dtype=np.uint8)  # ARC-AGI-3 colors 0-15 (from arc_agi.rendering)

st.set_page_config(page_title="ARC agent dashboard", layout="wide")


# -- loading ---------------------------------------------------------------

@st.cache_data
def load_results(paths: tuple[str, ...]) -> dict[str, dict]:
    out = {}
    for p in paths:
        data = json.loads(Path(p).read_text())
        label = data.get("label") or data.get("agent", Path(p).stem)
        if label in out:
            label = f"{label} ({Path(p).stem})"
        out[label] = data | {"_path": p}
    return out


@st.cache_data
def load_trace(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def grid_image(grid, scale: int = 8) -> np.ndarray:
    arr = PALETTE[np.asarray(grid, dtype=int) % 16]
    return arr.repeat(scale, axis=0).repeat(scale, axis=1)


# -- tab 1: leaderboard ----------------------------------------------------

def leaderboard() -> None:
    files = sorted(str(p) for p in [*Path("baselines").glob("*.json"), *Path("benchmarks/results").glob("*.json")])
    if not files:
        st.info("No results yet. Run `uv run python benchmark.py` or `run_agent.py --save ...`.")
        return
    results = load_results(tuple(files))
    chosen = st.multiselect("Runs to compare", list(results), default=list(results))
    if not chosen:
        return
    runs = {k: results[k] for k in chosen}

    common = set.intersection(*(set(r["games"]) for r in runs.values()))
    only_common = st.checkbox(f"Score only on games every run played ({len(common)} games)", value=True)

    rows = []
    for label, r in runs.items():
        games = {g: v for g, v in r["games"].items() if not only_common or g in common}
        rows.append({
            "run": label,
            "agent": r.get("agent"),
            "model": r.get("model") or "-",
            "actions/game": r.get("max_actions"),
            "seeds": len(r.get("seeds", [])),
            "games": len(games),
            "games with progress": sum(1 for g in games.values() if g["mean_levels"] > 0),
            "levels cleared (mean, total)": round(sum(g["mean_levels"] for g in games.values()), 2),
            "score % (mean over games)": round(float(np.mean([g["mean_score"] for g in games.values()])) if games else 0.0, 4),
        })
    table = pd.DataFrame(rows).sort_values("score % (mean over games)", ascending=False)
    st.dataframe(table, hide_index=True, width="stretch")
    if len({r.get("max_actions") for r in runs.values()}) > 1:
        st.caption("Runs used different action budgets per game; more actions makes clearing levels easier, "
                   "but the efficiency-based score punishes slow clears.")

    games = sorted(common if only_common else set().union(*(r["games"] for r in runs.values())))
    metric = st.radio("Per-game metric", ["mean_score", "mean_levels"], horizontal=True,
                      format_func=lambda m: {"mean_score": "score %", "mean_levels": "levels cleared"}[m])
    per_game = pd.DataFrame(
        {label: [r["games"].get(g, {}).get(metric, np.nan) for g in games] for label, r in runs.items()}, index=games
    )
    st.bar_chart(per_game, stack=False, height=380)
    with st.expander("Per-game table"):
        st.dataframe(per_game, width="stretch")


# -- tab 2: trace viewer ---------------------------------------------------

def trace_viewer() -> None:
    logs = sorted([*Path("logs").glob("*.jsonl"), *Path("examples/traces").glob("*.jsonl")],
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        st.info("No traces yet. Run with `--trace`, e.g. `uv run python run_agent.py --agent llm --trace -n 150`.")
        return
    path = st.selectbox("Trace", [str(p) for p in logs],
                        format_func=lambda p: Path(p).name + ("  (example)" if p.startswith("examples") else ""))
    steps = load_trace(path)
    if not steps:
        st.warning("Empty trace.")
        return

    events = Counter(s.get("learn", {}).get("event", "?") for s in steps)
    skills = Counter((s.get("decide", {}).get("skill") or s.get("reason", "")).split("(")[0] for s in steps)
    thinks = [s for s in steps if s.get("think")]
    c = st.columns(5)
    c[0].metric("actions", len(steps))
    c[1].metric("levels cleared", steps[-1]["result"]["levels_completed"])
    c[2].metric("deaths", events.get("game_over", 0))
    c[3].metric("states discovered", steps[-1].get("know", {}).get("states", "-"))
    c[4].metric("brain calls", len(thinks))

    left, right = st.columns([1, 1])
    with left:
        st.caption("Moves per skill / reason")
        st.bar_chart(pd.Series(skills).sort_values(ascending=False).head(12))
    with right:
        st.caption("Map size and moves this life over time")
        st.line_chart(pd.DataFrame({
            "states on map": [s.get("know", {}).get("states", np.nan) for s in steps],
            "moves this life": [s.get("know", {}).get("life_moves", np.nan) for s in steps],
        }))

    st.subheader("Step through")
    jump = st.selectbox("Jump to", ["(slider)"] + [f"{s['step']}: {s['learn'].get('event')}" for s in steps
                                                     if s.get("learn", {}).get("event") in ("level_up", "game_over")]
                        + [f"{s['step']}: brain - {s['think'].get('skill', '')}" for s in thinks])
    default = int(jump.split(":")[0]) if jump != "(slider)" else 1
    step_no = st.slider("Step", 1, len(steps), default)
    s = steps[step_no - 1]

    g, info = st.columns([1, 1.3])
    with g:
        if s.get("grid"):
            st.image(grid_image(s["grid"]), caption=f"what the agent saw before step {s['step']}")
        else:
            st.caption("(this trace has no grids; re-run with --trace to record them)")
    with info:
        st.markdown(f"**Step {s['step']}: `{s['action']}`**, {s['reason']}")
        if s.get("think"):
            t = s["think"]
            st.markdown(f"**THINK** → `{t.get('skill')}`  \n{t.get('thought', '')}")
            if t.get("notes"):
                st.markdown(f"*notes:* {t['notes']}")
            if t.get("error"):
                st.error(t["error"])
        for key in ("see", "learn", "decide", "know", "result"):
            if s.get(key):
                st.markdown(f"**{key.upper()}**")
                st.json(s[key], expanded=key in ("learn", "decide"))
        if s.get("think", {}).get("situation"):
            with st.expander("Exact situation text the brain received"):
                st.code(s["think"]["situation"], language=None)

    if thinks:
        with st.expander(f"All {len(thinks)} brain decisions"):
            st.dataframe(pd.DataFrame([{"step": t["step"], "skill": t["think"].get("skill"),
                                        "thought": t["think"].get("thought")} for t in thinks]),
                         hide_index=True, width="stretch")


# -- tab 3: Meta-TTL -------------------------------------------------------

def metattl_view() -> None:
    bases = [Path("runs/metattl"), Path("examples/metattl")]
    runs = sorted((p for b in bases for p in b.glob("*") if p.is_dir()), reverse=True)
    evals = sorted((p for b in bases for p in b.glob("eval-*.json")), reverse=True)
    if not runs and not evals:
        st.info("No Meta-TTL runs yet. `uv run python meta_ttl.py train --iterations 3`.")
        return
    if runs:
        run = st.selectbox("Training run", [str(p) for p in runs],
                       format_func=lambda p: Path(p).name + ("  (example)" if p.startswith("examples") else ""))
        run = Path(run)
        if (run / "config.json").exists():
            st.json(json.loads((run / "config.json").read_text()), expanded=False)
        log = run / "log.jsonl"
        if log.exists():
            entries = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
            st.caption("Outer-loop iterations")
            st.dataframe(pd.DataFrame(entries), hide_index=True, width="stretch")
        cols = st.columns(2)
        for col, name in zip(cols, ("pool.json", "selection.json")):
            if (run / name).exists():
                col.caption(name)
                col.json(json.loads((run / name).read_text()))
        cands = sorted((run / "candidates").glob("*.txt"))
        if cands:
            pick = st.selectbox("Meta-prompt candidate", [c.stem for c in cands])
            st.code((run / "candidates" / f"{pick}.txt").read_text(), language=None)
    for e in evals:
        data = json.loads(e.read_text())
        st.caption(f"{e.name}: W-AUC by method ({data['episodes']} episodes x {data['max_actions']} actions)")
        st.dataframe(pd.DataFrame(data["wauc"]), width="stretch")


st.title("ARC-AGI-3 agent dashboard")
tab1, tab2, tab3 = st.tabs(["Leaderboard", "Trace viewer", "Meta-TTL"])
with tab1:
    leaderboard()
with tab2:
    trace_viewer()
with tab3:
    metattl_view()
