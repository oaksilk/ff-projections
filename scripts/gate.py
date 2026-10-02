"""Decide whether a scheduled workflow run should proceed, and whether it pulls props.

Each ET slot in config.SCHEDULE_ET has two UTC crons (daylight and standard time);
only the one firing within GATE_MINUTES after the slot proceeds, and only during the
active season window (see SEASON_LEAD_DAYS in config). Manual runs always proceed.
Writes run=/props=/snapshot= lines for $GITHUB_OUTPUT. Standard library only.
"""
import csv
import datetime as dt
import io
import os
import sys
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ffmodel.config import (GATE_MINUTES, SCHEDULE_CSV, SCHEDULE_ET, SEASON_LEAD_DAYS,  # noqa: E402
                            SEASON_TAIL_DAYS, SNAPSHOT_SLOTS)


def season_windows(text: str):
    """[(start, end)] dates per season from the nflverse games CSV, regular season only."""
    days = {}
    for r in csv.DictReader(io.StringIO(text)):
        if r["game_type"] == "REG" and r["gameday"]:
            days.setdefault(r["season"], []).append(dt.date.fromisoformat(r["gameday"]))
    return [(min(d) - dt.timedelta(days=SEASON_LEAD_DAYS), max(d) + dt.timedelta(days=SEASON_TAIL_DAYS))
            for d in days.values()]


def in_season(today: dt.date) -> bool:
    """True inside a season window. Fails open (True) if the schedule can't be fetched."""
    try:
        with urllib.request.urlopen(SCHEDULE_CSV, timeout=30) as r:
            text = r.read().decode()
    except Exception as e:
        print(f"schedule fetch failed ({e}); assuming in season", file=sys.stderr)
        return True
    return any(a <= today <= b for a, b in season_windows(text))


def decide(now_et: dt.datetime, event: str, manual_props: bool, manual_snapshot: str = ""):
    """(run, props, snapshot label or "")."""
    if event != "schedule":
        return True, manual_props, manual_snapshot
    for wd, h, m, props in SCHEDULE_ET:
        slot = now_et.replace(hour=h, minute=m, second=0, microsecond=0)
        if now_et.weekday() == wd and dt.timedelta(0) <= now_et - slot < dt.timedelta(minutes=GATE_MINUTES):
            return True, props, SNAPSHOT_SLOTS.get((wd, h, m), "")
    return False, False, ""


if __name__ == "__main__":
    now = dt.datetime.now(ZoneInfo("America/New_York"))
    snap = os.environ.get("MANUAL_SNAPSHOT", "")
    event = os.environ.get("EVENT", "schedule")
    if event == "schedule" and not in_season(now.date()):
        print("run=false\nprops=false\nsnapshot=")
        print("off-season: skipping", file=sys.stderr)
        sys.exit(0)
    run, props, snap = decide(now, event, os.environ.get("MANUAL_PROPS") == "true",
                              "" if snap == "none" else snap)
    print(f"run={str(run).lower()}\nprops={str(props).lower()}\nsnapshot={snap}")
