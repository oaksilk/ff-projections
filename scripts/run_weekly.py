"""Build this week's rankings and write site/data/rankings.json.

Usage: python scripts/run_weekly.py [--props] [--snapshot LABEL] [--info friday|sunday]
  --props            fetch Vegas player props (spends ~4 Odds API credits per game)
  --snapshot LABEL   also save a frozen snapshot (fri/sun) for weekly evaluation
  --info SET         pregame information set to train on (default: by ET time, config.info_set)

Every run also writes an immutable forecast archive (archive.py) and refreshes
the prospective scorecard for finished weeks (scorecard.py).
"""
import argparse
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ffmodel import archive, live as lv, publish, scorecard, snapshot
from ffmodel.config import PROPS_DIR, ROOT, SCHEDULE_ET, info_set
from ffmodel.data import load_all
from ffmodel.features import build_features
from ffmodel.model import Distribution, Projector, add_distribution


def next_run(props_only=False, now=None):
    """Next scheduled run (optionally the next props pull), as an ISO time in UTC."""
    et = ZoneInfo("America/New_York")
    now = (now or dt.datetime.now(dt.timezone.utc)).astimezone(et)
    times = []
    for days in range(8):
        day = (now + dt.timedelta(days=days)).date()
        for wd, h, m, props in SCHEDULE_ET:
            if day.weekday() == wd and (props or not props_only):
                times.append(dt.datetime(day.year, day.month, day.day, h, m, tzinfo=et))
    return min(t for t in times if t > now).astimezone(dt.timezone.utc).isoformat(timespec="minutes")


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--props", action="store_true")
    ap.add_argument("--snapshot", choices=["fri", "sun"])
    ap.add_argument("--info", choices=["friday", "sunday"])
    args = ap.parse_args()
    load_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("run")

    d = load_all(max_age_hours=float(os.environ.get("CACHE_HOURS", 12)))
    season, week, games = lv.upcoming_week(d["schedules"])
    if season is None:
        log.info("no upcoming games; nothing to do")
        return
    log.info("projecting %s week %s (%d games)", season, week, len(games))

    players, db_out, _ = lv.availability(d, season, week, games)
    pl = d["players"].set_index("gsis_id")
    players["name"] = players.player_id.map(pl.display_name)
    players["headshot"] = players.player_id.map(pl.headshot)
    wx = lv.weather(games)
    n_outdoor = len(lv.outdoor_games(games))
    info = args.info or info_set(dt.datetime.now(ZoneInfo("America/New_York")))
    log.info("%d skill players expected active; %d DBs ruled out; %d of %d weather forecasts; %s information",
             len(players), len(db_out), len(wx), n_outdoor, info)

    absences = {}
    df = build_features(d, live_players=players[["player_id", "team", "season", "week", "position",
                                                 "inj_report", "inj_practice"]],
                        live_def_out=db_out, weather_override=wx, info=info, absences_out=absences)
    hist = df[~df.live]
    live = df[df.live & (df.season == season) & (df.week == week)]

    model = Projector().fit(df)
    pred = model.predict(live)
    dist = Distribution().fit(pd.read_parquet(ROOT / "model_data" / "oos.parquet"))
    model_pred = add_distribution(pred.copy(), live, dist)  # model-only, archived separately
    props_at, raw = lv.fetch_props(season, week, games, allow_fetch=args.props)
    pred = publish.blend_market(pred, players, raw)
    pred = add_distribution(pred, live, dist)
    log.info("%d players matched to Vegas props", int(pred.market.sum()))

    reasons = publish.explain(model, live, hist)
    weather = {r.game_id: {"temp": round(r.temp), "wind": round(r.wind), "precip": round(r.precip, 2)}
               for r in wx.itertuples() if pd.notna(r.temp) and pd.notna(r.wind)}
    pfr_names = d["players"].dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id").display_name
    ctx = publish.context(live, hist, d["players"].set_index("gsis_id").display_name, absences, pfr_names)
    ps = d["player_stats"]
    played = ps[(ps.season == season) & (ps.season_type == "REG")].game_id.unique()
    sched = d["schedules"]
    stats_through = sched[sched.game_id.isin(played)].gameday.max()
    freshness = {
        "next": next_run(),
        "injuries": players.attrs.get("injuries_at"),
        "weather": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes") if len(wx) else None,
        "outdoor_games": len(weather),
        "outdoor_expected": n_outdoor,
        "lines": lv.lines_updated_at(),
        "props": props_at.isoformat() if props_at is not None and len(raw) else None,
        "props_games": len(raw),
        "props_next": next_run(props_only=True),
        "stats_week": int(ps[(ps.season == season) & (ps.season_type == "REG")].week.max()),
        "stats_through": stats_through,
        # Latest completed game on the schedule (stats can lag it by a day or so).
        "games_through": sched[(sched.season == season) & (sched.game_type == "REG")
                               & sched.result.notna()].gameday.max(),
    }
    h2h = json.loads((ROOT / "model_data" / "h2h_calibration.json").read_text())
    payload = publish.to_json(pred, reasons, players, games, season, week,
                              {"weather": weather, "props_games": len(raw), "freshness": freshness,
                               "info_set": info, "h2h_slope": h2h["slope"]}, ctx)
    publish.write(payload)
    log.info("wrote %d players", len(payload["players"]))

    run_meta = {
        "git_sha": snapshot._git_sha(), "event": os.environ.get("GITHUB_EVENT_NAME", "local"),
        "slot": os.environ.get("SLOT") or None, "info_set": info,
        "market_weight": publish.MARKET_WEIGHT, "h2h_slope": h2h["slope"],
        "availability": {"expected_active": len(players), "ruled_out": players.attrs.get("ruled_out", []),
                         "db_out_pfr": sorted(db_out)},
        "market": {"props_games": len(raw), "players_with_props": int(pred.market.sum()),
                   "players": len(pred), "props": archive.props_reference(PROPS_DIR / f"{season}_w{week:02d}.json")},
        "weather": {"forecasts": len(weather), "outdoor_games": n_outdoor},
    }
    archive.save(pred, model_pred, live, payload, run_meta, d["ff_ids"])
    if args.snapshot:
        snapshot.save(payload, args.snapshot, d["ff_ids"])
    try:
        scorecard.update(d)
    except Exception as e:  # grading must never block publishing rankings
        log.warning("scorecard update failed: %s", e)


if __name__ == "__main__":
    main()
