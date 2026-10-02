"""Live context for the upcoming week: who plays, weather, and Vegas props."""
import datetime as dt
import json
import logging
import os
import re

import numpy as np
import pandas as pd
import requests
from scipy import optimize, stats

from .config import CACHE_DIR, OUT_STATUSES, PROPS_DIR, OUTDOOR_STADIUMS, POSITIONS
from .features import DB_POSITIONS

log = logging.getLogger(__name__)
ODDS_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
PROP_MARKETS = ["player_reception_yds", "player_receptions", "player_rush_yds", "player_anytime_td"]


def upcoming_week(sched, today=None):
    """(season, week, games) for the next week with unplayed games."""
    s = sched[(sched.game_type == "REG") & sched.result.isna()]
    if s.empty:
        return None, None, s
    season, week = s.sort_values(["season", "week"]).iloc[0][["season", "week"]]
    return int(season), int(week), s[(s.season == season) & (s.week == week)]


# ---------------------------------------------------------------- availability

SLEEPER_TEAMS = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LA"}


def sleeper_players():
    path = CACHE_DIR / "sleeper_players.json"
    fresh = path.exists() and (dt.datetime.now().timestamp() - path.stat().st_mtime) < 6 * 3600
    if not fresh:  # Sleeper asks callers to fetch this at most ~daily
        r = requests.get("https://api.sleeper.app/v1/players/nfl", timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    sp = pd.DataFrame(json.loads(path.read_text()).values())
    sp["team"] = sp.team.replace(SLEEPER_TEAMS)
    return sp[["player_id", "full_name", "team", "position", "status", "injury_status",
               "depth_chart_order", "gsis_id"]].rename(columns={"player_id": "sleeper_id"})


def _id_maps(d, sp):
    ids = d["ff_ids"][["sleeper_id", "gsis_id", "pfr_id"]].copy()
    ids["sleeper_id"] = ids.sleeper_id.astype(str).str.replace(r"\.0$", "", regex=True)
    sp = sp.merge(ids.rename(columns={"gsis_id": "gsis_ff"}), on="sleeper_id", how="left")
    sp["gsis_id"] = sp.gsis_id.where(sp.gsis_id.notna() & (sp.gsis_id.astype(str).str.strip() != ""), sp.gsis_ff)
    sp["gsis_id"] = sp.gsis_id.astype(str).str.strip()
    pfr = d["players"][["gsis_id", "pfr_id"]].dropna()
    sp = sp.merge(pfr.rename(columns={"pfr_id": "pfr_dir"}), on="gsis_id", how="left")
    sp["pfr_id"] = sp.pfr_id.fillna(sp.pfr_dir)
    return sp.drop(columns=["gsis_ff", "pfr_dir"])


def _sleeper_out(row):
    return (row.injury_status in OUT_STATUSES) or (row.status not in ("Active", None) and pd.notna(row.status))


def availability(d, season, week, games):
    """Skill players expected to play, and pfr ids of defensive backs ruled out."""
    sp = _id_maps(d, sleeper_players())
    sp["out"] = sp.apply(_sleeper_out, axis=1)
    inj = d["injuries"]
    inj = inj[(inj.season == season) & (inj.week == week)]
    nfl_out = set(inj[inj.report_status.isin(["Out", "Doubtful"])].gsis_id)
    nfl_q = set(inj[inj.report_status == "Questionable"].gsis_id)
    prac = inj.set_index("gsis_id").practice_status.map({
        "Full Participation in Practice": 0, "Limited Participation in Practice": 1,
        "Did Not Participate In Practice": 2}).dropna()
    prac = prac[~prac.index.duplicated()]

    teams = set(games.home_team) | set(games.away_team)
    by_gsis = sp.drop_duplicates("gsis_id").set_index("gsis_id")

    # Candidates: recent snaps for a playing team, plus Sleeper depth-chart players.
    pmap = d["players"][["pfr_id", "gsis_id"]].dropna().drop_duplicates("pfr_id")
    sn = d["snaps"]
    sn = sn[(sn.season == season) & (sn.game_type == "REG") & (sn.offense_snaps > 0)
            & sn.position.isin(POSITIONS + ["HB"])]
    recent = sn[sn.week >= week - 3].merge(pmap, left_on="pfr_player_id", right_on="pfr_id")
    cands = recent[["gsis_id", "team"]].drop_duplicates("gsis_id", keep="last")
    depth = sp[sp.team.isin(teams) & sp.position.isin(POSITIONS) & (sp.depth_chart_order <= 3)
               & sp.gsis_id.str.startswith("00-")]
    cands = pd.concat([cands, depth[["gsis_id", "team"]]]).drop_duplicates("gsis_id", keep="last")
    # Sleeper is the source of truth for current team (trades, releases).
    cands["team"] = cands.gsis_id.map(by_gsis.team).fillna(cands.team)
    cands = cands[cands.team.isin(teams)]
    out = cands.gsis_id.map(by_gsis.out).fillna(False).astype(bool) | cands.gsis_id.isin(nfl_out)
    players = cands[~out].copy()
    players["position"] = players.gsis_id.map(d["players"].set_index("gsis_id").position).replace({"HB": "RB"})
    players = players[players.position.isin(POSITIONS)]
    q = players.gsis_id.isin(nfl_q) | (players.gsis_id.map(by_gsis.injury_status) == "Questionable")
    players["inj_report"] = q.astype(int)
    players["inj_practice"] = players.gsis_id.map(prac).fillna(0)
    players = players.rename(columns={"gsis_id": "player_id"}).assign(season=season, week=week)

    # Defensive backs ruled out (or no longer on the team they played for).
    dbs = d["snaps"]
    dbs = dbs[(dbs.season == season) & dbs.position.isin(DB_POSITIONS) & (dbs.defense_snaps > 0)]
    dbs = dbs[["pfr_player_id", "team"]].drop_duplicates("pfr_player_id", keep="last")
    spf = sp.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id")
    gs_by_pfr = pmap.set_index("pfr_id").gsis_id
    db_out = set()
    for pid, team in dbs.itertuples(index=False):
        if pid in spf.index:
            r = spf.loc[pid]
            if r.out or (pd.notna(r.team) and r.team != team):
                db_out.add(pid)
        if gs_by_pfr.get(pid) in nfl_out:
            db_out.add(pid)
    return players, db_out, sp


# ---------------------------------------------------------------- weather

def weather(games):
    """Kickoff forecasts (temp F, wind mph) for outdoor games from Open-Meteo."""
    rows = []
    for g in games.itertuples():
        if g.location == "Neutral" or g.home_team not in OUTDOOR_STADIUMS:
            continue
        lat, lon = OUTDOOR_STADIUMS[g.home_team]
        try:
            r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=30, params={
                "latitude": lat, "longitude": lon, "hourly": "temperature_2m,wind_speed_10m,precipitation",
                "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
                "timezone": "America/New_York", "start_date": g.gameday, "end_date": g.gameday})
            r.raise_for_status()
            h = pd.DataFrame(r.json()["hourly"])
            hour = int(str(g.gametime).split(":")[0])
            # Average over the ~3 game hours starting at kickoff.
            w = h.iloc[hour:hour + 3]
            rows.append({"game_id": g.game_id, "temp": w.temperature_2m.mean(),
                         "wind": w.wind_speed_10m.mean(), "precip": w.precipitation.sum()})
        except Exception as e:  # weather is a nice-to-have; never fail the run
            log.warning("weather failed for %s: %s", g.game_id, e)
    return pd.DataFrame(rows, columns=["game_id", "temp", "wind", "precip"])


# ---------------------------------------------------------------- props

def _norm(name):
    name = re.sub(r"[^a-z ]", "", str(name).lower().replace("-", " "))
    return " ".join(w for w in name.split() if w not in {"jr", "sr", "ii", "iii", "iv", "v"})


def _implied(price):
    return 100 / (price + 100) if price > 0 else -price / (-price + 100)


def fetch_props(season, week, games, allow_fetch):
    """Raw props for the week. Cached per week so reruns don't spend credits."""
    path = PROPS_DIR / f"{season}_w{week:02d}.json"
    PROPS_DIR.mkdir(parents=True, exist_ok=True)
    if path.exists() and not allow_fetch:
        return json.loads(path.read_text())
    key = os.environ.get("ODDS_API_KEY")
    if not key or not allow_fetch:
        return json.loads(path.read_text()) if path.exists() else []
    events = requests.get(f"{ODDS_BASE}/events", params={"apiKey": key}, timeout=30).json()
    start = pd.to_datetime(games.gameday).min() - pd.Timedelta(days=1)
    end = pd.to_datetime(games.gameday).max() + pd.Timedelta(days=2)
    out = []
    for e in events:
        t = pd.to_datetime(e["commence_time"]).tz_convert("America/New_York").tz_localize(None)
        if not (start <= t <= end) or t < pd.Timestamp.now() - pd.Timedelta(hours=3):
            continue
        r = requests.get(f"{ODDS_BASE}/events/{e['id']}/odds", timeout=30, params={
            "apiKey": key, "regions": "us", "markets": ",".join(PROP_MARKETS), "oddsFormat": "american"})
        if r.ok:
            out.append(r.json())
        log.info("props %s @ %s: remaining credits %s", e["away_team"], e["home_team"],
                 r.headers.get("x-requests-remaining"))
    path.write_text(json.dumps(out))
    return out


# Coefficient of variation used to turn an over/under line into a mean.
YARD_CV = {"player_reception_yds": 0.62, "player_rush_yds": 0.58}  # measured from backtest outcomes


def _gamma_mean(line, p_over, cv):
    shape = 1 / cv**2
    f = lambda m: stats.gamma.sf(line, shape, scale=m / shape) - p_over
    return optimize.brentq(f, 0.5, 400)


def _poisson_mean(line, p_over):
    f = lambda lam: stats.poisson.sf(np.floor(line), lam) - p_over
    return optimize.brentq(f, 0.01, 30)


def market_means(raw):
    """Per-player market-implied means: receptions, rec yards, rush yards, TDs."""
    recs = []
    for ev in raw:
        for bk in ev.get("bookmakers", []):
            for m in bk["markets"]:
                by_player = {}
                for o in m["outcomes"]:
                    by_player.setdefault(o.get("description"), {})[o["name"]] = o
                for player, o in by_player.items():
                    try:
                        if m["key"] == "player_anytime_td" and "Yes" in o:
                            p = min(_implied(o["Yes"]["price"]) * 0.9, 0.95)  # strip ~10% hold
                            recs.append((player, "tds", -np.log(1 - p)))
                        elif "Over" in o and "Under" in o:
                            po, pu = _implied(o["Over"]["price"]), _implied(o["Under"]["price"])
                            p_over, line = po / (po + pu), o["Over"]["point"]
                            if m["key"] == "player_receptions":
                                recs.append((player, "receptions", _poisson_mean(line, p_over)))
                            else:
                                stat = "rec_yards" if m["key"] == "player_reception_yds" else "rush_yards"
                                recs.append((player, stat, _gamma_mean(line, p_over, YARD_CV[m["key"]])))
                    except (ValueError, KeyError):
                        continue
    if not recs:
        return pd.DataFrame(columns=["name_key"])
    df = pd.DataFrame(recs, columns=["player", "stat", "value"])
    df = df.groupby(["player", "stat"]).value.median().unstack().reset_index()
    df["name_key"] = df.player.map(_norm)
    return df
