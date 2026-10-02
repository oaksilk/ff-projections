"""Decide whether a scheduled workflow run should proceed, and whether it pulls props.

Each ET slot in config.SCHEDULE_ET has two UTC crons (daylight and standard time);
only the one firing within GATE_MINUTES after the slot proceeds. Manual runs always
proceed. Writes run=/props=/snapshot= lines for $GITHUB_OUTPUT. Standard library only.
"""
import datetime as dt
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ffmodel.config import GATE_MINUTES, SCHEDULE_ET, SNAPSHOT_SLOTS  # noqa: E402


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
    run, props, snap = decide(now, os.environ.get("EVENT", "schedule"), os.environ.get("MANUAL_PROPS") == "true",
                              "" if snap == "none" else snap)
    print(f"run={str(run).lower()}\nprops={str(props).lower()}\nsnapshot={snap}")
