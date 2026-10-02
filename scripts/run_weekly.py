"""Build this week's rankings and write site/data/rankings.json.

Usage: python scripts/run_weekly.py [--props]
  --props   fetch Vegas player props (spends ~4 Odds API credits per game)
"""
import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ffmodel import live as lv, publish
from ffmodel.config import ROOT, SCHEDULE_ET
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
    log.info("%d skill players expected active; %d DBs ruled out; %d weather forecasts",
             len(players), len(db_out), len(wx))

    df = build_features(d, live_players=players[["player_id", "team", "season", "week", "position",
                                                 "inj_report", "inj_practice"]],
                        live_def_out=db_out, weather_override=wx)
    hist = df[~df.live]
    live = df[df.live & (df.season == season) & (df.week == week)]

    model = Projector().fit(df)
    pred = model.predict(live)
    pred["model_half"] = pred.half_proj
    props_at, raw = lv.fetch_props(season, week, games, allow_fetch=args.props)
    pred = publish.blend_market(pred, players, raw)
    dist = Distribution().fit(pd.read_parquet(ROOT / "model_data" / "oos.parquet"))
    pred = add_distribution(pred, live, dist)
    log.info("%d players matched to Vegas props", int(pred.market.sum()))

    reasons = publish.explain(model, live, hist)
    weather = {r.game_id: {"temp": round(r.temp), "wind": round(r.wind), "precip": round(r.precip, 2)}
               for r in wx.itertuples()}
    ctx = publish.context(live, hist, d["players"].set_index("gsis_id").display_name)
    ps = d["player_stats"]
    played = ps[(ps.season == season) & (ps.season_type == "REG")].game_id.unique()
    sched = d["schedules"]
    stats_through = sched[sched.game_id.isin(played)].gameday.max()
    freshness = {
        "next": next_run(),
        "injuries": players.attrs.get("injuries_at"),
        "weather": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes") if len(wx) else None,
        "outdoor_games": len(wx),
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
    payload = publish.to_json(pred, reasons, players, games, season, week,
                              {"weather": weather, "props_games": len(raw), "freshness": freshness}, ctx)
    publish.write(payload, season, week)
    log.info("wrote %d players", len(payload["players"]))


if __name__ == "__main__":
    main()
