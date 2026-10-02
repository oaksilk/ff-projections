# Start/Sit Rankings

Weekly fantasy football projections and rankings for RB, WR and TE in PPR,
half-PPR and standard scoring. Built for season-long start/sit decisions.

The system runs itself: GitHub Actions rebuilds the rankings three times a
week and publishes them to a static web page. There is nothing to update by
hand.

## What you see

For each player:

- **Projection**: the average expected fantasy points.
- **Range**: the 10th–90th percentile outcome (floor to ceiling), with the
  middle half highlighted.
- **Boom %**: the chance of 20+ points.
- **Why**: the factors moving the projection most, in points, e.g.
  `+1.5 Game environment` or `−2.2 Own injury status`.
- **Head to head**: pick any two players to see the chance each one outscores
  the other.

## How it works

The model answers two questions for every player and multiplies them together:

1. **How many chances will he get?** Team volume (from Vegas implied totals,
   pace and pass rate) × his share (targets, carries, snap share, air yards,
   red-zone opportunity).
2. **What does he do with a chance?** Yards per target or carry, catch rate
   and TD expectation, pulled toward league norms when the sample is small.

It then adjusts for context:

- Opponent defense
- Injuries to the opposing secondary
- Teammates who are out (vacated targets and carries)
- Who is starting at QB and how good he has been
- Weather at kickoff
- The player's own injury report
- Vegas player props, on Sunday runs

Under the hood:

| Piece | What it is |
|---|---|
| Features | ~90 pre-game signals per player-game, each computed only from information available before kickoff |
| Projection | Average of two gradient-boosted models: one predicts each stat (receptions, yards, TDs…) and scores it; one predicts fantasy points directly |
| Vegas blend | Prop lines (receptions, receiving yards, rushing yards, anytime TD) are converted to expected stats and blended 50/50 with the model |
| Range and boom % | A quantile model fit on out-of-sample backtest predictions: given a projection and how volatile a player's role is, how spread out were real outcomes? |
| Explanations | Each factor group is reset to a neutral value and the model re-run; the difference is that factor's contribution |

## How good is it?

`scripts/backtest.py` replays past seasons week by week. It trains only on
data available at the time and compares rankings against FantasyPros expert
consensus (ECR) on the same players. Results are in `reports/backtest.txt`.

## Schedule

All times are Eastern; the source of truth is `SCHEDULE_ET` in `ffmodel/config.py`.

| When | What updates |
|---|---|
| Nightly 7:05 PM | Injury news; Thu/Sun/Mon-night inactives |
| Tue 10:00 AM | Last week's stats are final |
| Sun 9:00 AM | Morning injury news |
| Sun 11:45 AM | 1pm inactives, kickoff weather, **Vegas props** (the only props pull) |
| Sun 3:00 PM | Late-afternoon inactives |

The page header shows when each data source was last refreshed.

## Data sources (all free)

- [nflverse](https://github.com/nflverse): play-by-play stats, snap counts,
  expected fantasy points, injuries, schedules and betting lines
- [Sleeper API](https://docs.sleeper.com/): current injury status and depth charts
- [Open-Meteo](https://open-meteo.com/): kickoff weather forecasts
- [The Odds API](https://the-odds-api.com/): player props. The free tier is
  500 credits a month, and one Sunday pull uses about 60.

## Operating it

Setup is one time:

1. Add the `ODDS_API_KEY` repository secret.
2. Set Pages to "GitHub Actions" (Settings → Pages → Source).

Run locally:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
echo "ODDS_API_KEY=..." > .env
.venv/bin/python scripts/run_weekly.py          # rankings without props
.venv/bin/python scripts/run_weekly.py --props  # include Vegas props
.venv/bin/python scripts/backtest.py            # ~20 min; refreshes model_data/oos.parquet
```

Once a year, in the offseason, rerun the backtest so the range model learns
from the latest season.
