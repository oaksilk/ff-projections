"""Frozen pre-game snapshots of our rankings and FantasyPros expert consensus (ECR).

Each file is snapshots/<season>/w<week>/<label>.json and is written once, never
overwritten: it is the record of what we (and the experts) said before kickoff.
ECR is saved alongside because the nflverse feed only keeps occasional history,
so a later download may not match what the experts said at snapshot time.
"""
import datetime as dt
import json
import logging
import os
import subprocess

import nflreadpy as nfl
import pandas as pd

from .config import ROOT, SCORING, SNAPSHOT_DIR

log = logging.getLogger(__name__)
KEEP = ("id", "name", "pos", "team", "opp", "home", "kickoff", "questionable", "market")


def path(season, week, label):
    return SNAPSHOT_DIR / str(season) / f"w{week:02d}" / f"{label}.json"


def current_ecr(ff_ids: pd.DataFrame) -> pd.DataFrame:
    """This week's PPR ECR for RB/WR/TE, keyed to our (gsis) player ids."""
    r = nfl.load_ff_rankings("week").to_pandas()
    r = r[r.page.isin(["ppr-rb", "ppr-wr", "ppr-te"])]
    ids = ff_ids[["fantasypros_id", "gsis_id"]].dropna()
    ids["fantasypros_id"] = ids.fantasypros_id.astype("Int64").astype(str)
    r["fantasypros_id"] = r.fantasypros_id.astype(str)
    r = r.merge(ids, on="fantasypros_id", how="left")
    return r[["gsis_id", "fantasypros_id", "player_name", "pos", "team", "ecr", "sd", "scrape_date"]]


def _git_sha():
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def save(payload, label, ff_ids, force=False):
    """Write the snapshot unless one already exists for this week and label."""
    out = path(payload["season"], payload["week"], label)
    if out.exists() and not force:
        log.info("snapshot %s already exists; leaving it untouched", out)
        return None
    ecr = current_ecr(ff_ids)
    players = [{**{k: p.get(k) for k in KEEP}, **{f: p[f] for f in SCORING}} for p in payload["players"]]
    snap = {
        "meta": {
            "label": label, "season": payload["season"], "week": payload["week"],
            "taken_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
            "rankings_generated": payload["generated"], "model_commit": _git_sha(),
            "quantiles": payload["quantiles"], "boom_threshold": payload["boom_threshold"],
            "freshness": payload.get("freshness"),
            "ecr_scrape_date": str(ecr.scrape_date.max()) if len(ecr) else None,
            "ecr_unmatched": int(ecr.gsis_id.isna().sum()),
        },
        "players": players,
        "ecr": json.loads(ecr.to_json(orient="records")),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snap, separators=(",", ":"), default=str))
    log.info("saved snapshot %s (%d players, %d ECR rows)", out, len(players), len(ecr))
    return out
