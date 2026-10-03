# Start/Sit Rankings

Weekly fantasy football projections and rankings for RB, WR and TE in PPR,
half-PPR and standard scoring. Built for season-long start/sit decisions.

The system runs itself: GitHub Actions rebuilds the rankings on a fixed
schedule (and whenever injury news breaks between runs) and publishes them to a
static web page. There is nothing to update by hand.

## What you see

For each player:

- **Projection**: the average expected fantasy points.
- **Range**: the 10th–90th percentile outcome (floor to ceiling), with the
  middle half highlighted.
- **Boom %**: the chance of 20+ points.
- **Why**: the biggest influences on the number, in points for the scoring
  you picked, e.g. `+1.5 Game environment` or `+0.8 Vegas adjustment`. Factors
  are sensitivities (what changes if that one input is reset to neutral), not
  causes, and they don't add up to the projection.
- **Head to head**: pick any two players to see how often each outscores the
  other in simulated games, plus the projected points gap. Flags teammates and
  players in the same game, whose outcomes are linked.
- **Scorecard**: how our rankings, our model alone, FantasyPros experts,
  Sleeper, ESPN and simple baselines have done on finished weeks.

## How it works

The model predicts each player's outcome directly from ~90 pre-game signals.
The most important are opportunity (targets, carries, snap share, air yards,
red-zone and expected-TD opportunity) and efficiency (yards per target or
carry, catch rate), along with the game's Vegas implied total. It learns how
these combine from ten seasons of games. It does **not** project team volume
first and divide it among players, so player projections aren't forced to add
up to a team total (a possible future improvement).

It then adjusts for context:

- Opponent defense
- Injuries to the opposing secondary
- Teammates who are out (vacated targets and carries)
- Who is starting at QB and how good he has been
- Weather at kickoff
- The player's own injury report
- Vegas player props, from Sunday's late-morning pull onward

Under the hood:

| Piece | What it is |
|---|---|
| Features | ~90 pre-game signals per player-game, each computed only from information available before kickoff |
| Projection | Average of two gradient-boosted models: one predicts each stat (receptions, yards, TDs…) and scores it; one predicts fantasy points directly |
| Vegas adjustment | Prop lines (receptions, receiving yards, rushing yards, anytime TD) are converted to expected stats; half of each gap between props and the model's own stat line is added to the projection, so props that agree change nothing |
| Range and boom % | A quantile model fit on out-of-sample backtest predictions: given a projection and how volatile a player's role is, how spread out were real outcomes? |
| Explanations | Each factor group is reset to a neutral value and the model re-run, per scoring format; the difference is that factor's sensitivity |

## How good is it?

`scripts/backtest.py` replays 2022–2025 week by week. It trains only on data
available before each game (including who was known to be out) and compares
rankings against FantasyPros expert consensus (ECR) on the same players.
Results are in `reports/backtest.txt`; the summary is in `docs/DESIGN.md` §5.

In short: the model is **statistically tied with expert consensus** (61.6% vs
61.7% of start/sit pairs ordered correctly), clearly better than a recent-average
baseline (59.3%), and weakest at WR. Averaging our ranking with the experts'
does slightly but measurably better than the experts alone. Close calls are
close to coin flips for everyone. The Vegas adjustment can't be backtested (no
free historical props), so the live scorecard tracks it from 2026 Week 4 on.

## Schedule

All times are Eastern; the source of truth is `SCHEDULE_ET` in `ffmodel/config.py`.

| When | What updates |
|---|---|
| Nightly 7:05 PM | Injury news; Thu/Sun/Mon-night inactives (Friday's run also saves a snapshot) |
| Tue 10:00 AM | Last week's stats are final; last week's report card is emailed |
| Sun 9:00 AM | Morning injury news |
| Sun 11:15 AM | First **Vegas props** attempt |
| Sun 11:45 AM | 1pm inactives, kickoff weather, props retry (reuses 11:15's pull if it worked); saves a snapshot |
| Sun 3:00 PM | Late-afternoon inactives |
| Hourly, 8 AM–11 PM | If injury statuses changed since the last build, rebuild |

GitHub's scheduler often runs late, so a check fires twice an hour and runs any
scheduled update that is due but hasn't happened yet (up to 6 hours late).

The page header shows when each data source was last refreshed.

Runs happen only during the regular season, starting about 9 days before the
opener. They pause through the playoffs, off-season and preseason, and restart
on their own. Details are in [`docs/OPERATIONS.md`](docs/OPERATIONS.md).

## Track record

Every build saves an immutable copy of its forecasts
(`site/data/history/<season>_wNN/<time>.json`), with the model-only and published
numbers kept separately. FantasyPros, Sleeper and ESPN numbers from the same
moment are kept in the repo (`data/archive/`) but not published on the site.
Once a week is final, the scorecard grades every source on its last forecast
before each player's kickoff (`site/data/scorecard.json`, shown on the page and
in the weekly email). The Friday-evening and Sunday-morning snapshots
(`snapshots/`) and their reports (`reports/live/`) continue as before.

## Data sources (all free)

- [nflverse](https://github.com/nflverse): play-by-play stats, snap counts,
  expected fantasy points, injuries, schedules and betting lines
- [Sleeper API](https://docs.sleeper.com/): current injury status and depth charts
- [Open-Meteo](https://open-meteo.com/): kickoff weather forecasts
- ESPN: injury statuses for the hourly change check, and projections for the
  scorecard (unofficial endpoints; failures never stop a run)
- Sleeper projections (Rotowire) for the scorecard (unofficial endpoint)
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
.venv/bin/python scripts/backtest.py            # ~1.5 h; refreshes model_data/ and reports/backtest.txt
```

The backtest reruns itself every Aug 1 so the range model learns from the
latest season.
