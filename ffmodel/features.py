"""Feature engineering.

Every row is one player in one game. Every feature is computed only from
information available before kickoff: prior games (lagged, exponentially
weighted), Vegas lines, weather, injury reports and who is inactive.

Upcoming games are handled by appending rows with empty outcomes, so the
exact same code path builds training features and live features.
"""
import numpy as np
import pandas as pd

from .config import FUMBLE_PT, OUTDOOR_STADIUMS, POSITIONS, SCORING, TD_PT, YD_PT

DB_POSITIONS = {"CB", "S", "FS", "SS", "DB"}


# ---------------------------------------------------------------- helpers

def _lag_ewm(df, key, cols, hl, prefix, lag=True):
    """Per-key exponentially weighted mean of prior rows (df sorted by key, time)."""
    src = df.groupby(key, sort=False)[cols].shift(1) if lag else df[cols].copy()
    src[key] = df[key].values
    out = src.groupby(key, sort=False)[cols].ewm(halflife=hl, ignore_na=True).mean()
    out = out.reset_index(level=0, drop=True).sort_index()
    return out.add_prefix(prefix)


def score(df, fmt):
    """Fantasy points from component stats for a scoring format."""
    return (
        YD_PT * (df.rec_yards + df.rush_yards)
        + TD_PT * (df.rec_tds + df.rush_tds)
        + SCORING[fmt]["rec"] * df.receptions
        + FUMBLE_PT * df.fumbles_lost
        + 2 * df.two_pt
    )


# ---------------------------------------------------------------- games

def build_games(sched):
    """One row per team per regular-season game, with Vegas and weather context."""
    s = sched[sched.game_type == "REG"].copy()
    rows = []
    for side, opp_side, sign in (("home", "away", 1), ("away", "home", -1)):
        g = pd.DataFrame({
            "game_id": s.game_id, "season": s.season, "week": s.week,
            "gameday": s.gameday, "gametime": s.gametime,
            "team": s[f"{side}_team"], "opp": s[f"{opp_side}_team"],
            "home_team": s.home_team,
            "is_home": int(side == "home") * (s.location != "Neutral").astype(int),
            "spread": sign * s.spread_line,  # positive = this team favored
            "total_line": s.total_line,
            "rest": s[f"{side}_rest"],
            "roof": s.roof, "temp": s.temp, "wind": s.wind,
            "points": s[f"{side}_score"], "opp_points": s[f"{opp_side}_score"],
            "starter_qb_sched": s[f"{side}_qb_id"],
            "location": s.location,
        })
        rows.append(g)
    g = pd.concat(rows, ignore_index=True)
    g["implied"] = (g.total_line + g.spread) / 2
    g["opp_implied"] = (g.total_line - g.spread) / 2
    outdoor = g.home_team.isin(OUTDOOR_STADIUMS.keys())
    g["dome"] = (g.roof.isin(["dome", "closed"]) | (g.roof.isna() & ~outdoor)).astype(int)
    g.loc[g.dome == 1, ["temp", "wind"]] = [70.0, 0.0]
    g["completed"] = g.points.notna()
    g = g.sort_values(["team", "season", "week"]).reset_index(drop=True)
    g["team_game_idx"] = g.groupby("team").cumcount()
    return g


# ---------------------------------------------------------------- players

STAT_MAP = {
    "targets": "targets", "receptions": "receptions",
    "receiving_yards": "rec_yards", "receiving_tds": "rec_tds",
    "receiving_air_yards": "air_yards", "target_share": "target_share",
    "air_yards_share": "air_yards_share", "carries": "carries",
    "rushing_yards": "rush_yards", "rushing_tds": "rush_tds",
}
OPP_MAP = {
    "rec_touchdown_exp": "rec_td_exp", "rush_touchdown_exp": "rush_td_exp",
    "rec_yards_gained_exp": "rec_yards_exp", "rush_yards_gained_exp": "rush_yards_exp",
    "total_fantasy_points_exp": "xfp",
}
LABELS = ["receptions", "rec_yards", "rec_tds", "carries", "rush_yards", "rush_tds",
          "fumbles_lost", "two_pt"]


def build_player_games(d, games):
    """Historical player-game rows for RB/WR/TE (anyone who took an offensive snap)."""
    ps = d["player_stats"]
    ps = ps[ps.season_type == "REG"].copy()
    ps["fumbles_lost"] = ps[["rushing_fumbles_lost", "receiving_fumbles_lost"]].fillna(0).sum(axis=1)
    ps["two_pt"] = ps[["rushing_2pt_conversions", "receiving_2pt_conversions"]].fillna(0).sum(axis=1)
    stats = ps.rename(columns=STAT_MAP)[
        ["player_id", "season", "week", "team", "position"] + list(STAT_MAP.values())
        + ["fumbles_lost", "two_pt"]]

    # Snap counts give us players who played but recorded no stats.
    pmap = d["players"][["pfr_id", "gsis_id"]].dropna().drop_duplicates("pfr_id")
    sn = d["snaps"]
    sn = sn[(sn.game_type == "REG") & (sn.offense_snaps > 0)
            & sn.position.isin(POSITIONS + ["FB", "HB"])]
    sn = sn.merge(pmap, left_on="pfr_player_id", right_on="pfr_id")
    sn = sn.rename(columns={"gsis_id": "player_id", "offense_pct": "snap_pct"})[
        ["player_id", "season", "week", "team", "snap_pct"]]
    sn = sn.drop_duplicates(["player_id", "season", "week"])

    pg = stats.merge(sn, on=["player_id", "season", "week"], how="outer", suffixes=("", "_sn"))
    pg["team"] = pg.team.fillna(pg.team_sn)
    pg = pg.drop(columns="team_sn")

    # Position: prefer the stats feed, fall back to the player directory.
    pos = d["players"].set_index("gsis_id").position
    pg["position"] = pg.position.fillna(pg.player_id.map(pos))
    pg["position"] = pg.position.replace({"HB": "RB"})
    pg = pg[pg.position.isin(POSITIONS)]

    opp = d["opportunity"].rename(columns=OPP_MAP)
    opp = opp[["player_id", "season", "week"] + list(OPP_MAP.values())]
    opp["season"] = opp.season.astype(int)
    opp["week"] = opp.week.astype(int)
    pg = pg.merge(opp, on=["player_id", "season", "week"], how="left")

    pg = pg.merge(games[["season", "week", "team", "team_game_idx", "completed"]],
                  on=["season", "week", "team"], how="inner")
    pg = pg[pg.completed].drop(columns="completed")
    fill0 = list(STAT_MAP.values()) + list(OPP_MAP.values()) + ["fumbles_lost", "two_pt"]
    pg[fill0] = pg[fill0].fillna(0)
    return pg


def build_team_games(d, games):
    """Team offensive volume per game, plus what each defense allowed."""
    ps = d["player_stats"]
    ps = ps[ps.season_type == "REG"]
    t = ps.groupby(["season", "week", "team"]).agg(
        pass_att=("attempts", "sum"), sacks=("sacks_suffered", "sum"),
        carries=("carries", "sum"), targets=("targets", "sum"),
        pass_epa=("passing_epa", "sum"), rush_yards=("rushing_yards", "sum"),
        rec_yards=("receiving_yards", "sum"),
    ).reset_index()
    t["dropbacks"] = t.pass_att + t.sacks
    t["plays"] = t.dropbacks + t.carries
    t["pass_rate"] = t.dropbacks / t.plays
    t["epa_per_db"] = t.pass_epa / t.dropbacks.clip(lower=1)
    t["ypc"] = t.rush_yards / t.carries.clip(lower=1)
    t["ypt"] = t.rec_yards / t.targets.clip(lower=1)
    return t


def starting_qbs(d):
    ps = d["player_stats"]
    qb = ps[(ps.season_type == "REG") & (ps.attempts > 0)].copy()
    qb["dropbacks"] = qb.attempts + qb.sacks_suffered.fillna(0)
    qb["epa_db"] = qb.passing_epa / qb.dropbacks.clip(lower=1)
    starters = qb.sort_values("attempts").drop_duplicates(["season", "week", "team"], keep="last")
    return qb, starters[["season", "week", "team", "player_id"]].rename(columns={"player_id": "qb_id"})


# ---------------------------------------------------------------- absences

def _absences(appear, value_cols, live_present=None, min_value=None, lookback=3):
    """For each team-game, sum the recent-role values of players who were
    around in the last `lookback` team games but are absent from this one.

    appear: rows (pid, team, team_game_idx, <value_cols>) where the values are
            the player's role *as of and including* that game.
    live_present: rows (pid, team, team_game_idx) of players expected to play in
            upcoming games; candidates for those games not in it are absent.
    """
    cands = []
    for k in range(1, lookback + 1):
        c = appear.copy()
        c["target_idx"] = c.team_game_idx + k
        cands.append(c)
    c = pd.concat(cands).sort_values("team_game_idx")
    c = c.drop_duplicates(["pid", "team", "target_idx"], keep="last")
    present = appear[["pid", "team", "team_game_idx"]]
    if live_present is not None:
        present = pd.concat([present, live_present[["pid", "team", "team_game_idx"]]])
    present = present.rename(columns={"team_game_idx": "target_idx"}).assign(here=1)
    c = c.merge(present.drop_duplicates(), on=["pid", "team", "target_idx"], how="left")
    absent = c[c.here.isna()]
    if min_value is not None:
        absent = absent[absent[value_cols[0]] >= min_value]
    out = absent.groupby(["team", "target_idx"])[value_cols].agg(["sum", "count"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index().rename(columns={"target_idx": "team_game_idx"})


# ---------------------------------------------------------------- main

def build_features(d, live_players=None, live_def_out=None, weather_override=None):
    """Return one row per player-game with features and (if played) labels.

    live_players: DataFrame (player_id, team, season, week, position,
                  inj_report, inj_practice) of players expected to play upcoming.
    live_def_out: set of pfr ids of defensive players ruled out upcoming.
    weather_override: DataFrame (game_id, temp, wind) of forecasts.
    """
    games = build_games(d["schedules"])
    if weather_override is not None and len(weather_override):
        w = games.merge(weather_override, on="game_id", how="left", suffixes=("", "_fc"))
        for col in ("temp", "wind"):
            games[col] = np.where(w[f"{col}_fc"].notna() & (games.dome == 0), w[f"{col}_fc"], games[col])

    pg = build_player_games(d, games)
    if live_players is not None and len(live_players):
        lp = live_players.merge(games[["season", "week", "team", "team_game_idx"]],
                                on=["season", "week", "team"])
        pg = pd.concat([pg, lp], ignore_index=True)
    pg = pg.merge(games, on=["season", "week", "team", "team_game_idx"], how="left")

    # Labels
    pg["live"] = pg.receptions.isna()
    for f in SCORING:
        pg[f"fp_{f}"] = score(pg, f)

    # ---- player history features
    pg = pg.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    tm = build_team_games(d, games)
    pg = pg.merge(tm[["season", "week", "team", "carries"]].rename(columns={"carries": "team_carries"}),
                  on=["season", "week", "team"], how="left")
    pg["carry_share"] = pg.carries / pg.team_carries.clip(lower=1)
    pg["opps"] = pg.targets + pg.carries
    pg["rz_td_exp"] = pg.rec_td_exp + pg.rush_td_exp

    hist_cols = ["snap_pct", "target_share", "air_yards_share", "carry_share", "targets",
                 "carries", "receptions", "rec_yards", "air_yards", "rush_yards",
                 "rec_tds", "rush_tds", "rec_td_exp", "rush_td_exp", "rec_yards_exp",
                 "rush_yards_exp", "xfp", "fp_half", "fp_ppr", "opps"]
    feats = [_lag_ewm(pg, "player_id", hist_cols, 3, "r_"),
             _lag_ewm(pg, "player_id", hist_cols, 12, "l_")]
    pg = pd.concat([pg] + feats, axis=1)

    g = pg.groupby("player_id", sort=False)
    pg["n_games"] = g.cumcount()
    pg["n_games_season"] = pg.groupby(["player_id", "season"]).cumcount()
    pg["last_snap_pct"] = g.snap_pct.shift(1)
    pg["last_target_share"] = g.target_share.shift(1)
    pg["last_carry_share"] = g.carry_share.shift(1)
    pg["new_team"] = (g.team.shift(1) != pg.team).astype(int)
    pg.loc[pg.n_games == 0, "new_team"] = 0
    pg["fp_vol"] = g.fp_half.transform(lambda s: s.shift(1).rolling(12, min_periods=3).std())
    pg["fp_p90"] = g.fp_half.transform(lambda s: s.shift(1).rolling(16, min_periods=4).quantile(0.9))
    for p in ("r_", "l_"):
        pg[f"{p}adot"] = pg[f"{p}air_yards"] / pg[f"{p}targets"].clip(lower=0.1)
        pg[f"{p}ypt"] = pg[f"{p}rec_yards"] / pg[f"{p}targets"].clip(lower=0.1)
        pg[f"{p}ypc"] = pg[f"{p}rush_yards"] / pg[f"{p}carries"].clip(lower=0.1)
        pg[f"{p}catch_rate"] = pg[f"{p}receptions"] / pg[f"{p}targets"].clip(lower=0.1)

    # Draft capital as a prior for players with little history.
    pl = d["players"].set_index("gsis_id")
    pg["draft_pick"] = pg.player_id.map(pl.draft_pick).fillna(300)
    pg["years_exp"] = pg.season - pg.player_id.map(pl.rookie_season)

    # ---- own injury report
    inj = d["injuries"]
    inj = inj[inj.game_type == "REG"][["gsis_id", "season", "week", "report_status", "practice_status"]]
    inj = inj.drop_duplicates(["gsis_id", "season", "week"])
    rep = {"Questionable": 1, "Doubtful": 2, "Out": 3}
    prac = {"Full Participation in Practice": 0, "Limited Participation in Practice": 1,
            "Did Not Participate In Practice": 2}
    inj["inj_report_h"] = inj.report_status.map(rep)
    inj["inj_practice_h"] = inj.practice_status.map(prac)
    pg = pg.merge(inj.rename(columns={"gsis_id": "player_id"})[
        ["player_id", "season", "week", "inj_report_h", "inj_practice_h"]],
        on=["player_id", "season", "week"], how="left")
    if "inj_report" in pg:
        pg["inj_report"] = pg.inj_report.fillna(pg.inj_report_h)
        pg["inj_practice"] = pg.inj_practice.fillna(pg.inj_practice_h)
    else:
        pg["inj_report"], pg["inj_practice"] = pg.inj_report_h, pg.inj_practice_h
    pg["inj_report"] = pg.inj_report.fillna(0)
    pg["inj_practice"] = pg.inj_practice.fillna(0)

    # ---- team volume (lagged)
    tg = games.merge(tm, on=["season", "week", "team"], how="left").sort_values(["team", "team_game_idx"])
    tg = tg.reset_index(drop=True)
    tcols = ["plays", "dropbacks", "carries", "pass_rate", "epa_per_db", "points"]
    tg = pd.concat([tg, _lag_ewm(tg, "team", tcols, 4, "tm_")], axis=1)

    # Defense allowed (lagged), keyed by the defense's own game index.
    allowed = pg[~pg.live].groupby(["season", "week", "opp", "position"]).agg(
        fp=("fp_half", "sum"), tg=("targets", "sum")).unstack("position")
    allowed.columns = [f"alw_{a}_{b}" for a, b in allowed.columns]
    allowed = allowed.reset_index().rename(columns={"opp": "team"})
    dg = games[["season", "week", "team", "team_game_idx"]].merge(allowed, on=["season", "week", "team"], how="left")
    off = tm.rename(columns={"team": "opp"})[["season", "week", "opp", "epa_per_db", "ypc", "ypt", "plays", "pass_rate"]]
    dg = dg.merge(games[["season", "week", "team", "opp", "points", "opp_points"]], on=["season", "week", "team"])
    dg = dg.merge(off, on=["season", "week", "opp"], how="left")
    dg = dg.sort_values(["team", "team_game_idx"]).reset_index(drop=True)
    dcols = [c for c in dg.columns if c.startswith("alw_")] + ["epa_per_db", "ypc", "ypt", "plays", "pass_rate", "opp_points"]
    dfe = _lag_ewm(dg, "team", dcols, 6, "def_")
    dg = pd.concat([dg[["season", "week", "team"]], dfe], axis=1).rename(columns={"team": "opp"})

    # ---- QB: who starts and how good he has been
    qb, starters = starting_qbs(d)
    tg = tg.merge(starters, on=["season", "week", "team"], how="left")
    live_qb = tg.qb_id.isna() & tg.starter_qb_sched.notna() & ~tg.completed
    tg.loc[live_qb, "qb_id"] = tg.loc[live_qb, "starter_qb_sched"]
    tg["qb_change"] = (tg.groupby("team").qb_id.shift(1) != tg.qb_id).astype(int)
    qrows = pd.concat([qb[["player_id", "season", "week", "epa_db", "dropbacks"]],
                       tg.loc[live_qb, ["qb_id", "season", "week"]].rename(columns={"qb_id": "player_id"})])
    qrows = qrows.drop_duplicates(["player_id", "season", "week"]).sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    qrows["qb_epa"] = _lag_ewm(qrows, "player_id", ["epa_db"], 10, "")["epa_db"]
    qrows["qb_games"] = qrows.groupby("player_id").cumcount()
    tg = tg.merge(qrows[["player_id", "season", "week", "qb_epa", "qb_games"]].rename(columns={"player_id": "qb_id"}),
                  on=["qb_id", "season", "week"], how="left")

    # ---- absences: teammates (vacated opportunity) and opposing secondary
    appear = pg[~pg.live][["player_id", "team", "team_game_idx"]].rename(columns={"player_id": "pid"})
    role = _lag_ewm(pg, "player_id", ["target_share", "carry_share"], 3, "", lag=False)
    appear = appear.join(role)
    live_present = None
    if live_players is not None and len(live_players):
        live_present = pg[pg.live][["player_id", "team", "team_game_idx"]].rename(columns={"player_id": "pid"})
    vac = _absences(appear, ["target_share", "carry_share"], live_present)
    vac = vac.rename(columns={"target_share_sum": "vac_tgt", "carry_share_sum": "vac_car"})[
        ["team", "team_game_idx", "vac_tgt", "vac_car"]]
    tg = tg.merge(vac, on=["team", "team_game_idx"], how="left")
    tg[["vac_tgt", "vac_car"]] = tg[["vac_tgt", "vac_car"]].fillna(0)

    sn = d["snaps"]
    dbs = sn[(sn.game_type == "REG") & sn.position.isin(DB_POSITIONS) & (sn.defense_snaps > 0)]
    dbs = dbs.merge(games[["season", "week", "team", "team_game_idx"]], on=["season", "week", "team"])
    dbs = dbs.rename(columns={"pfr_player_id": "pid"}).sort_values(["pid", "season", "week"]).reset_index(drop=True)
    dbs["def_pct"] = _lag_ewm(dbs, "pid", ["defense_pct"], 3, "", lag=False)["defense_pct"]
    live_db = None
    if live_def_out is not None:
        # Upcoming games: every recent DB plays unless ruled out.
        nxt = games[~games.completed].groupby("team").team_game_idx.min().rename("nxt").reset_index()
        cand = dbs.merge(nxt, on="team")
        cand = cand[(cand.team_game_idx >= cand.nxt - 3) & ~cand.pid.isin(live_def_out)]
        live_db = cand[["pid", "team", "nxt"]].rename(columns={"nxt": "team_game_idx"}).drop_duplicates()
    dba = _absences(dbs[["pid", "team", "team_game_idx", "def_pct"]], ["def_pct"], live_db, min_value=0.5)
    dba = dba.rename(columns={"def_pct_sum": "db_out_share", "def_pct_count": "db_out_n"})
    dba = dba.merge(games[["team", "team_game_idx", "season", "week"]], on=["team", "team_game_idx"])
    dba = dba.rename(columns={"team": "opp"})[["season", "week", "opp", "db_out_share", "db_out_n"]]

    # ---- assemble
    tkeep = ["season", "week", "team", "qb_change", "qb_epa", "qb_games", "vac_tgt", "vac_car"] + \
            [f"tm_{c}" for c in tcols]
    pg = pg.merge(tg[tkeep], on=["season", "week", "team"], how="left")
    pg = pg.merge(dg, on=["season", "week", "opp"], how="left")
    pg = pg.merge(dba, on=["season", "week", "opp"], how="left")
    pg[["db_out_share", "db_out_n"]] = pg[["db_out_share", "db_out_n"]].fillna(0)
    pg["pos_code"] = pg.position.map({"RB": 0, "WR": 1, "TE": 2})
    # Matchup: what this opponent allows to this player's position.
    for p in POSITIONS:
        m = pg.position == p
        pg.loc[m, "def_alw_fp_pos"] = pg.loc[m, f"def_alw_fp_{p}"]
        pg.loc[m, "def_alw_tg_pos"] = pg.loc[m, f"def_alw_tg_{p}"]
    return pg


def feature_columns(df):
    base = ["pos_code", "is_home", "spread", "total_line", "implied", "opp_implied", "rest",
            "dome", "temp", "wind", "n_games", "n_games_season", "last_snap_pct",
            "last_target_share", "last_carry_share", "new_team", "fp_vol", "fp_p90",
            "draft_pick", "years_exp", "inj_report", "inj_practice", "qb_change", "qb_epa",
            "qb_games", "vac_tgt", "vac_car", "db_out_share", "db_out_n",
            "def_alw_fp_pos", "def_alw_tg_pos"]
    hist = [c for c in df.columns if c.startswith(("r_", "l_", "tm_"))
            or (c.startswith("def_") and not c.startswith("def_alw_"))]
    return [c for c in dict.fromkeys(base + hist) if c in df.columns]
