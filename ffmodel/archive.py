"""Immutable per-run forecast archive, plus third-party projections for grading.

Every run writes two files with the same UTC timestamp, never overwritten:

* site/data/history/<season>_wNN/<timestamp>.json   our forecasts (published on the site)
    model-only and published (Vegas-adjusted) numbers kept separately, the
    recent-average baseline, input timestamps, git commit, scoring rules,
    availability assumptions, prop coverage and a reference to the raw props.
* data/archive/<season>_wNN/<timestamp>.json        third-party numbers (repo only, not deployed)
    FantasyPros ECR (from nflverse), Sleeper (Rotowire) and ESPN projections.
    Kept off the site so we don't redistribute other companies' data.

The scorecard (scorecard.py) grades each player on the last file saved before
his own kickoff. The Sleeper and ESPN endpoints are unofficial: any failure is
logged and skipped, never fatal.
"""
import datetime as dt
import hashlib
import json
import logging

import pandas as pd
import requests

from .config import FUMBLE_PT, OUTPUT_DIR, ROOT, SCORING, TD_PT, YD_PT
from .model import BOOM, QUANTILES

log = logging.getLogger(__name__)
FORECAST_DIR = OUTPUT_DIR / "data" / "history"
THIRD_PARTY_DIR = ROOT / "data" / "archive"


def week_dir(base, season, week):
    return base / f"{season}_w{week:02d}"


def stamp(now=None):
    """Filename-safe UTC timestamp, e.g. 2026-10-04T15-47-03Z."""
    now = now or dt.datetime.now(dt.timezone.utc)
    return now.strftime("%Y-%m-%dT%H-%M-%SZ")


def _write_once(path, obj):
    """Create the file; refuse to overwrite (mode 'x')."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:
        json.dump(obj, f, separators=(",", ":"), default=lambda o: None if pd.isna(o) else str(o))
    return path


def _r(x, n=1):
    return None if x is None or pd.isna(x) else round(float(x), n)


def _dist(r, fmt, prefix=""):
    return {"proj": _r(r[f"{prefix}{fmt}_proj"]),
            "q": [_r(r[f"{prefix}{fmt}_q{int(q * 100)}"]) for q in QUANTILES],
            "boom": _r(r[f"{prefix}{fmt}_boom"], 3)}


def forecast_record(pred, model_pred, live, payload, run_meta):
    """The archived forecast: one entry per player, model-only and published side by side."""
    info = {p["id"]: p for p in payload["players"]}
    naive_half = live.r_fp_half.to_numpy()
    naive_ppr = live.r_fp_ppr.to_numpy()
    players = []
    for k, (i, r) in enumerate(pred.iterrows()):
        m = model_pred.loc[i]
        p = info.get(r.player_id, {})
        players.append({
            "id": r.player_id, "name": p.get("name"), "pos": r.position, "team": r.team, "opp": r.opp,
            "kickoff": p.get("kickoff"), "market": bool(r.get("market", False)),
            "model": {fmt: _dist(m, fmt) for fmt in SCORING},
            "published": {fmt: _dist(r, fmt) for fmt in SCORING},
            "naive": {"ppr": _r(naive_ppr[k]), "half": _r(naive_half[k]),
                      "std": _r(2 * naive_half[k] - naive_ppr[k])},
        })
    return {"meta": {**run_meta, "season": payload["season"], "week": payload["week"],
                     "generated": payload["generated"], "quantiles": QUANTILES, "boom_threshold": BOOM,
                     "scoring": {f: {"rec": v["rec"], "yd": YD_PT, "td": TD_PT, "fumble_lost": FUMBLE_PT,
                                     "two_pt": 2} for f, v in SCORING.items()},
                     "freshness": payload.get("freshness")},
            "players": players}


def props_reference(path):
    """Where the raw props live and a hash, so the archive pins exactly what was used."""
    if not path.exists():
        return None
    data = path.read_bytes()
    try:
        fetched = json.loads(data).get("fetched_at")
    except AttributeError:
        fetched = None
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(data).hexdigest(), "fetched_at": fetched}


# ---------------------------------------------------------------- third parties

def _ids(ff_ids, col):
    ids = ff_ids[[col, "gsis_id"]].dropna().copy()
    ids[col] = ids[col].astype(str).str.replace(r"\.0$", "", regex=True)
    return ids.drop_duplicates(col).set_index(col).gsis_id


def ecr_now(ff_ids):
    from .snapshot import current_ecr
    e = current_ecr(ff_ids)
    return json.loads(e[["gsis_id", "player_name", "pos", "team", "ecr", "sd", "scrape_date"]]
                      .to_json(orient="records"))


def sleeper_projections(season, week, ff_ids):
    r = requests.get(f"https://api.sleeper.app/projections/nfl/{season}/{week}", timeout=30,
                     params={"season_type": "regular", "position[]": ["RB", "WR", "TE"]})
    r.raise_for_status()
    gsis = _ids(ff_ids, "sleeper_id")
    out = []
    for x in r.json():
        st = x.get("stats") or {}
        if st.get("pts_ppr") is None:
            continue
        out.append({"gsis_id": gsis.get(str(x["player_id"])), "sleeper_id": str(x["player_id"]),
                    "name": " ".join(filter(None, [x.get("player", {}).get("first_name"),
                                                   x.get("player", {}).get("last_name")])),
                    "pos": x.get("player", {}).get("position"), "team": x.get("team"),
                    "ppr": st.get("pts_ppr"), "half": st.get("pts_half_ppr"), "std": st.get("pts_std"),
                    "updated_at": x.get("updated_at"), "source": x.get("company")})
    return out


ESPN_STATS = {"53": "rec", "42": "rec_yd", "43": "rec_td", "24": "rush_yd", "25": "rush_td",
              "72": "fum_lost", "26": "two_pt_rush", "44": "two_pt_rec"}


def espn_projections(season, week, ff_ids):
    period = f"11{season}{week}"
    flt = {"players": {"filterSlotIds": {"value": [2, 4, 6]},
                       "filterStatsForTopScoringPeriodIds": {"value": 2, "additionalValue": [period]},
                       "sortPercOwned": {"sortPriority": 1, "sortAsc": False}, "limit": 500}}
    r = requests.get(f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}"
                     "/segments/0/leaguedefaults/3", params={"view": "kona_player_info", "scoringPeriodId": week},
                     headers={"X-Fantasy-Filter": json.dumps(flt)}, timeout=30)
    r.raise_for_status()
    gsis = _ids(ff_ids, "espn_id")
    pos = {2: "RB", 3: "WR", 4: "TE"}
    out = []
    for e in r.json().get("players", []):
        p = e.get("player", {})
        if p.get("defaultPositionId") not in pos:
            continue
        proj = next((s for s in p.get("stats", []) if s.get("statSourceId") == 1
                     and s.get("scoringPeriodId") == week), None)
        if proj is None:
            continue
        st = {v: float(proj.get("stats", {}).get(k, 0) or 0) for k, v in ESPN_STATS.items()}
        base = (YD_PT * (st["rec_yd"] + st["rush_yd"]) + TD_PT * (st["rec_td"] + st["rush_td"])
                + FUMBLE_PT * st["fum_lost"] + 2 * (st["two_pt_rush"] + st["two_pt_rec"]))
        out.append({"gsis_id": gsis.get(str(p.get("id"))), "espn_id": str(p.get("id")),
                    "name": p.get("fullName"), "pos": pos[p["defaultPositionId"]],
                    **{f: round(base + SCORING[f]["rec"] * st["rec"], 2) for f in SCORING}})
    return out


def third_party_record(season, week, ff_ids):
    rec = {"meta": {"season": season, "week": week,
                    "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}}
    for name, fn in (("ecr", lambda: ecr_now(ff_ids)),
                     ("sleeper", lambda: sleeper_projections(season, week, ff_ids)),
                     ("espn", lambda: espn_projections(season, week, ff_ids))):
        try:
            rec[name] = fn()
            log.info("archived %d %s rows", len(rec[name]), name)
        except Exception as e:  # unofficial / third-party: never fail the run
            log.warning("third-party %s unavailable: %s", name, e)
            rec[name] = None
            rec["meta"][f"{name}_error"] = str(e)[:300]
    return rec


def save(pred, model_pred, live, payload, run_meta, ff_ids):
    """Write this run's forecast and third-party archive files. Returns the forecast path."""
    season, week = payload["season"], payload["week"]
    ts = stamp()
    fpath = _write_once(week_dir(FORECAST_DIR, season, week) / f"{ts}.json",
                        forecast_record(pred, model_pred, live, payload, run_meta))
    _write_once(week_dir(THIRD_PARTY_DIR, season, week) / f"{ts}.json",
                third_party_record(season, week, ff_ids))
    log.info("archived forecast %s", fpath.relative_to(ROOT))
    return fpath
