"""Raw data loading from nflverse, with a simple per-day parquet cache."""
import datetime as dt
import logging

import nflreadpy as nfl
import pandas as pd

from .config import CACHE_DIR, FIRST_SEASON

log = logging.getLogger(__name__)


def current_season(today: dt.date | None = None) -> int:
    today = today or dt.date.today()
    return today.year if today.month >= 8 else today.year - 1


def _cached(name: str, loader, max_age_hours: float = 12) -> pd.DataFrame:
    """Load from cache if fresh, otherwise call loader and cache the result."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.parquet"
    if path.exists():
        age = dt.datetime.now().timestamp() - path.stat().st_mtime
        if age < max_age_hours * 3600:
            return pd.read_parquet(path)
    log.info("downloading %s", name)
    df = loader()
    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas()
    df.to_parquet(path, index=False)
    return df


def seasons(last: int | None = None) -> list[int]:
    return list(range(FIRST_SEASON, (last or current_season()) + 1))


def load_all(max_age_hours: float = 12) -> dict[str, pd.DataFrame]:
    s = seasons()
    loaders = {
        "player_stats": lambda: nfl.load_player_stats(s),
        "schedules": lambda: nfl.load_schedules(s),
        "snaps": lambda: nfl.load_snap_counts(s),
        "opportunity": lambda: nfl.load_ff_opportunity(s),
        "injuries": lambda: nfl.load_injuries(s),
        # Weekly roster status: ACT, INA (gameday inactive, 2019+), RES (IR/reserve), ...
        "rosters": lambda: nfl.load_rosters_weekly(s).select(
            ["season", "week", "team", "gsis_id", "position", "status"]),
        "players": lambda: nfl.load_players(),
        "ff_ids": lambda: nfl.load_ff_playerids(),
    }
    return {k: _cached(k, f, max_age_hours) for k, f in loaders.items()}


def load_ecr(max_age_hours: float = 12) -> pd.DataFrame:
    """FantasyPros weekly expert consensus rankings (Friday snapshots).

    Position pages (weekly-rb/wr/te) plus the overall offense page (weekly-op),
    which orders RB/WR/TE against each other for flex comparisons.
    """
    def loader():
        r = nfl.load_ff_rankings("all").to_pandas()
        r = r[r.page_type.isin(["weekly-wr", "weekly-rb", "weekly-te", "weekly-op"])]
        return r[["page_type", "id", "player", "pos", "team", "ecr", "scrape_date"]]
    return _cached("ecr_v2", loader, max_age_hours)
