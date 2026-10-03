"""Decide whether a scheduled workflow firing should build, and what it should do.

GitHub cron fires late (sometimes hours) and occasionally not at all, so the
workflow fires this cheap check twice an hour (config.CHECK_MINUTES). A build
runs when:

1. **A scheduled slot has passed and hasn't been served yet** (config.SCHEDULE_ET,
   catch-up up to MAX_LATE_HOURS). Several missed slots are served by one build:
   props if any of them pulls props, the latest snapshot label among them.
2. Otherwise, **injury news**: between INJURY_CHECK_HOURS_ET, if ESPN's injury
   statuses for relevant positions changed since the last build (and the last
   build was at least INJURY_REBUILD_GAP_MIN ago). No props.

Only during the active season window. Manual runs always build. The build job
records what it served in data/run_state.json (`--record`) after it succeeds,
so a failed build is retried by the next firing.

Writes run=/props=/snapshot=/slot=/injury_hash=/reason= lines for $GITHUB_OUTPUT.
Standard library only.
"""
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import os
import sys
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ffmodel.config import (ESPN_INJURIES, INJURY_CHECK_HOURS_ET, INJURY_CHECK_POSITIONS,  # noqa: E402
                            INJURY_REBUILD_GAP_MIN, MAX_LATE_HOURS, RUN_STATE, SCHEDULE_CSV,
                            SCHEDULE_ET, SEASON_LEAD_DAYS, SEASON_TAIL_DAYS, SNAPSHOT_SLOTS)

ET = ZoneInfo("America/New_York")


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


def read_state(path=RUN_STATE):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def slots_between(start: dt.datetime, end: dt.datetime):
    """Scheduled slot times (ET, aware) in (start, end], with their props flag and snapshot label."""
    out = []
    day = start.date()
    while day <= end.date():
        for wd, h, m, props in SCHEDULE_ET:
            if day.weekday() == wd:
                t = dt.datetime(day.year, day.month, day.day, h, m, tzinfo=ET)
                if start < t <= end:
                    out.append((t, props, SNAPSHOT_SLOTS.get((wd, h, m), "")))
        day += dt.timedelta(days=1)
    return sorted(out)


def pending_slots(now_et: dt.datetime, state: dict):
    """Slots that have passed, weren't served, and aren't more than MAX_LATE_HOURS old."""
    floor = now_et - dt.timedelta(hours=MAX_LATE_HOURS)
    served = state.get("last_slot")
    if served:
        floor = max(floor, dt.datetime.fromisoformat(served).astimezone(ET))
    return slots_between(floor, now_et)


def injury_hash():
    """Fingerprint of ESPN injury statuses for positions the model cares about, or None on failure."""
    try:
        req = urllib.request.Request(ESPN_INJURIES, headers={"Accept-Encoding": "gzip"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
        data = json.loads(raw)
    except Exception as e:
        print(f"injury check failed ({e})", file=sys.stderr)
        return None
    items = []
    for team in data.get("injuries", []):
        for i in team.get("injuries", []):
            a = i.get("athlete", {})
            pos = (a.get("position") or {}).get("abbreviation") if isinstance(a.get("position"), dict) else None
            if pos in INJURY_CHECK_POSITIONS:
                items.append(f"{a.get('id')}:{i.get('status')}")
    return hashlib.sha256("|".join(sorted(items)).encode()).hexdigest()[:16]


def decide(now_et, event, manual_props=False, manual_snapshot="", state=None, check_injuries=None):
    """dict(run, props, snapshot, slot, injury_hash, reason)."""
    out = {"run": False, "props": False, "snapshot": "", "slot": "", "injury_hash": "", "reason": ""}
    if event != "schedule":
        return {**out, "run": True, "props": manual_props, "snapshot": manual_snapshot, "reason": "manual"}
    state = state or {}
    pend = pending_slots(now_et, state)
    if pend:
        last = pend[-1][0]
        snaps = [s for _, _, s in pend if s]
        late = int((now_et - last).total_seconds() // 60)
        return {**out, "run": True, "props": any(p for _, p, _ in pend), "snapshot": snaps[-1] if snaps else "",
                "slot": last.isoformat(), "reason": f"slot {last:%a %H:%M} ET ({late} min after)"}
    lo, hi = INJURY_CHECK_HOURS_ET
    if not lo <= now_et.hour < hi:
        return {**out, "reason": "no slot due; outside injury-check hours"}
    last_build = state.get("last_build")
    if last_build and now_et - dt.datetime.fromisoformat(last_build).astimezone(ET) < \
            dt.timedelta(minutes=INJURY_REBUILD_GAP_MIN):
        return {**out, "reason": "no slot due; built recently"}
    h = (check_injuries or injury_hash)()
    if h is None:
        return {**out, "reason": "no slot due; injury check unavailable"}
    if h == state.get("injury_hash"):
        return {**out, "injury_hash": h, "reason": "no slot due; no injury changes"}
    return {**out, "run": True, "injury_hash": h, "reason": "injury statuses changed"}


def record(path=RUN_STATE, now=None):
    """After a successful build: remember the served slot, build time and injury hash."""
    state = read_state(path)
    now = now or dt.datetime.now(dt.timezone.utc)
    slot = os.environ.get("SLOT", "")
    if slot and (not state.get("last_slot") or dt.datetime.fromisoformat(slot) > dt.datetime.fromisoformat(state["last_slot"])):
        state["last_slot"] = slot
    state["last_build"] = now.isoformat(timespec="seconds")
    h = os.environ.get("INJURY_HASH") or injury_hash()
    if h:
        state["injury_hash"] = h
    Path(path).write_text(json.dumps(state, indent=1) + "\n")
    print(f"recorded {state}", file=sys.stderr)


if __name__ == "__main__":
    if "--record" in sys.argv:
        record()
        sys.exit(0)
    now = dt.datetime.now(ET)
    snap = os.environ.get("MANUAL_SNAPSHOT", "")
    event = os.environ.get("EVENT", "schedule")
    if event == "schedule" and not in_season(now.date()):
        print("run=false\nprops=false\nsnapshot=\nslot=\ninjury_hash=\nreason=off-season")
        sys.exit(0)
    res = decide(now, event, os.environ.get("MANUAL_PROPS") == "true",
                 "" if snap == "none" else snap, read_state())
    for k, v in res.items():
        print(f"{k}={str(v).lower() if isinstance(v, bool) else v}")
    print(f"gate: {res['reason']}", file=sys.stderr)
