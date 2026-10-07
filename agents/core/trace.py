"""Trace logs: a step-by-step record of what the agent saw, learned and did.

Writes two files per game into logs/:
  <game>-<agent>-<time>.jsonl   one JSON object per action, for analysis
  <game>-<agent>-<time>.txt     the same, readable, with grid snapshots at key moments
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

HEX = "0123456789abcdef"
SNAPSHOT_EVENTS = {"start", "level_up", "game_over"}


class Tracer:
    def __init__(self, game: str, agent: str, log_dir: Path = Path("logs")) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{game}-{agent}-{datetime.now():%Y%m%d-%H%M%S}"
        self.jsonl_path = log_dir / f"{stem}.jsonl"
        self.text_path = log_dir / f"{stem}.txt"
        self._jsonl = self.jsonl_path.open("w")
        self._text = self.text_path.open("w")

    def write(self, record: dict, grid_before: np.ndarray | None) -> None:
        grid = grid_before.tolist() if grid_before is not None else None
        self._jsonl.write(json.dumps(record | {"grid": grid}, default=str) + "\n")
        self._text.write(_format(record))
        event = record.get("learn", {}).get("event")
        if grid_before is not None and event in SNAPSHOT_EVENTS:
            self._text.write(f"      grid the agent saw ({event}):\n{_grid(grid_before)}\n")

    def close(self) -> None:
        self._jsonl.close()
        self._text.close()


def _format(r: dict) -> str:
    see, learn, decide, know = (r.get(k, {}) for k in ("see", "learn", "decide", "know"))
    result = r["result"]
    lines = [f"#{r['step']:05d}  L{see.get('level', '?')}  {r['action']}  ({r['reason']})"]
    if see:
        change = f"{see.get('pixels_changed', 0)}px changed"
        if "change_box" in see:
            change += " in box (x0,y0,x1,y1)=" + str(tuple(see["change_box"]))
        lines.append(
            f"      SEE    state {see['key']} ({'seen before' if see['seen_before'] else 'NEW'}), "
            f"{change}, {see['object_count']} objects"
        )
    if learn:
        extra = "; ".join(f"{k}={v}" for k, v in learn.items() if k not in ("event", "move"))
        lines.append(f"      LEARN  {learn['event']}: {learn.get('move', '')}" + (f"  [{extra}]" if extra else ""))
    if decide:
        lines.append(
            f"      DECIDE {decide['reason']} -> {decide['action']}" + (f"  [skill {decide['skill']}]" if decide.get("skill") else "")
        )
    if think := r.get("think"):
        lines.append(f"      THINK  {think.get('skill', '')}  <- {think.get('thought', '')}")
        if think.get("notes"):
            lines.append(f"             notes: {think['notes']}")
        if think.get("error"):
            lines.append(f"             error: {think['error']}")
    if know:
        lines.append(
            f"      KNOW   {know['states']} states, life {know['life_moves']} moves, "
            f"left {know['moves_left'] if know['moves_left'] is not None else '?'}, deaths: {know['death_cause']}"
            + (", STALLED" if know["stalled"] else "")
        )
    lines.append(f"      RESULT {result['state']}, levels {result['levels_completed']}")
    return "\n".join(lines) + "\n"


def _grid(grid: np.ndarray) -> str:
    return "\n".join("        " + "".join(HEX[int(v) % 16] for v in row) for row in grid)
