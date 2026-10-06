"""Download every public ARC-AGI-3 game into environment_files/ so agents can run offline.

    uv run python fetch_games.py
"""

import logging

import arc_agi


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    arc = arc_agi.Arcade()  # NORMAL mode: lists games via the API, caches them locally on make()
    games = arc.get_environments()
    print(f"{len(games)} games available")
    for info in games:
        env = arc.make(info.game_id)
        status = "ok" if env is not None else "FAILED"
        print(f"  {info.game_id:16s} {','.join(info.tags or ['-']):15s} {len(info.baseline_actions or [])} levels  {status}")


if __name__ == "__main__":
    main()
