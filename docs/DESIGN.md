# Start/Sit Rankings: Design Record

This document preserves the *why* behind this tool: its goals, philosophy, the
decisions made while building it, and how each piece works. Read it before
changing anything. The README covers day-to-day operation; this covers intent.

Originally built 2026-10-02, during the 2026 NFL season (Week 4).

---

## 1. Purpose

**Goal:** rank fantasy football players every week with accuracy, for
**season-long start/sit decisions**. The output is a ranked list that users can
compare across, not a binary start/sit verdict. The motivating question:
"Should I start Jameson Williams or another WR this week, and where does he
rank among the top 50 receivers?"

**Product decisions (from the owner):**

| Decision | Choice |
|---|---|
| Scoring formats | All three: PPR, half-PPR, standard (half is the page default) |
| Use case | Season-long start/sit only. **Not DFS**: no salaries, ownership or lineup optimization |
| Positions | RB, WR, TE first. QB is the natural next addition |
| Upkeep | Near zero. It must run itself; no CSVs to update by hand |
| Cost | Free. Every data source and the hosting are free tier |
| Output | A ranked list per position and flex, with a range, boom odds, reasons and head-to-head comparison |

**The owner is not a data scientist.** Explanations on the page must be
plain-English and honest. The "Why" factors are the most loved feature;
keep them accurate and interpretable above all.

## 2. Philosophy

These principles came from researching how sportsbooks and the best
projection shops work. Keep them.

1. **Opportunity over efficiency.** Fantasy points = chances × what a player
   does per chance. Volume (targets, carries, snap share, air-yards share,
   red-zone opportunity) is stable week to week; efficiency (yards per target,
   TD rate) is noisy. The model leans on volume and pulls efficiency toward
   league norms.
2. **TDs are mostly luck.** Model them from *expected* TDs (the red-zone and
   goal-line opportunity in nflverse's `ffopportunity` data), not from past TD
   counts.
3. **Top-down structure.** Team volume (Vegas implied total, pace, pass rate)
   → player share → efficiency → distribution.
4. **The betting market is the strongest single signal.** Vegas player props
   are blended 50/50 with the model when available (Sunday runs). The market
   is hard to beat; the model's job is to be close to it on its own, and to
   explain *why*.
5. **Matchups matter less than people think.** Defense-vs-position effects
   are real but small. The data confirmed this: WRs facing secondaries missing
   3+ regular DBs scored the same as usual (12.4 vs 12.4 PPR points per game,
   2016–2025). Don't hand-tune matchup boosts the data doesn't support.
6. **Output a distribution, not a number.** Every player gets a median,
   floor/ceiling (10th–90th percentile), and boom odds (P ≥ 20 points). Boom
   vs. bust profile is a real start/sit input (e.g. needing upside as an
   underdog).
7. **No information leaks.** Every feature uses only what was knowable before
   kickoff. Backtests train only on data before the week being predicted.
   This is the most common way hobby models fool themselves.
8. **Honest validation.** Measure against a real benchmark (FantasyPros
   expert consensus) and report results faithfully, including where the model
   loses.

## 3. Architecture

```
nflverse (stats, snaps, xFP, injuries, schedules+lines)  ─┐
Sleeper API (live injury status, depth charts, teams)    ─┤
Open-Meteo (kickoff weather)                              ─┼─► features.py ─► model.py ─► publish.py ─► site/data/rankings.json
The Odds API (player props, Sunday only)                 ─┘      (≈90 pre-game     (projection   (Vegas blend,     site/index.html renders it
                                                                   features)        + range)      reasons, JSON)
```

| File | Role |
|---|---|
| `ffmodel/config.py` | Seasons, scoring rules, outdoor stadium coordinates, "out" injury statuses |
| `ffmodel/data.py` | nflverse loaders with a 12-hour parquet cache (`data/cache/`, gitignored) |
| `ffmodel/features.py` | Builds one row per player-game. Upcoming games are appended as rows with empty outcomes, so **training and live features use the exact same code path** |
| `ffmodel/model.py` | `Projector` (the average outcome) and `Distribution` (the range) |
| `ffmodel/live.py` | The upcoming week: who plays (Sleeper + nflverse injuries), weather, props |
| `ffmodel/publish.py` | Vegas blend, factor explanations, plain-English context, JSON output |
| `ffmodel/backtest.py` | Walk-forward evaluation against FantasyPros ECR |
| `scripts/run_weekly.py` | The weekly job (what GitHub Actions runs) |
| `scripts/backtest.py` | Full backtest (~20 min); also regenerates `model_data/oos.parquet` |
| `site/index.html` | Single static page; all logic is client-side JS over `rankings.json` |

### 3.1 Features (`features.py`)

Every row is a player in a game. Feature families:

- **Player history:** exponentially weighted averages of *prior* games at two
  horizons (`r_` = recent, half-life 3 games; `l_` = long, half-life 12):
  snap %, target share, air-yards share, carry share, targets, carries,
  yards, TDs, expected TDs, expected fantasy points (xFP), fantasy points.
  Plus aDOT, yards per target and carry, catch rate, games played, last-game
  role, volatility (`fp_vol`), 90th-percentile outcome, a team-change flag,
  and draft pick plus years of experience (priors for players with little history).
- **Game environment:** spread, total, implied team and opponent points,
  home, rest days.
- **Team volume:** lagged team plays, dropbacks, carries, pass rate, EPA per
  dropback, points (`tm_` prefix).
- **Opponent defense:** lagged fantasy points and targets allowed to each
  position, EPA per dropback allowed, yards per carry and per target allowed,
  plays faced, pass rate faced, points allowed (`def_` prefix).
- **QB:** starting QB's EWMA EPA per dropback (`qb_epa`), career games,
  `qb_change` (different starter than last game), and `qb_upgrade`
  (`qb_epa` minus the team's recent passing EPA, i.e. "is this QB better than
  the QB play behind this player's recent stats").
- **Absences:** `vac_tgt` and `vac_car` are the recent target and carry
  shares of teammates who are out. `db_out_n` and `db_out_share` count
  opposing regular defensive backs (≥50% of snaps recently) who are out.
  Training infers "out" from snap counts (on the team in the last 3 games but
  didn't play); live uses injury reports and Sleeper status.
- **Weather:** temperature and wind (domes set to 70°F with no wind).
- **Own health:** injury report status and practice participation.

### 3.2 Projection (`Projector`)

The projection is an **equal average of two gradient-boosted model families**
(scikit-learn `HistGradientBoostingRegressor`, regularized: learning rate
0.03, 500 iterations, 15 leaves, min leaf 300):

1. **Component models**, one per stat: receptions, receiving yards,
   receiving TDs, carries, rushing yards, rushing TDs, fumbles lost and 2-pt
   conversions. Poisson loss for counts. Scoring them gives any format and
   lets Vegas props substitute stat by stat.
2. **Direct models**, one per scoring format, predicting fantasy points.

Seasons are weighted by `0.88^(years ago)`. The model retrains from scratch
on every run, using all completed games including earlier weeks of the
current season (in-season data measurably helps; see §5).

### 3.3 Vegas blend (`publish.blend_market`)

Props pulled: receptions, receiving yards, rushing yards, anytime TD. Each is
converted to an expected value:

- **Yards:** solve for the mean of a gamma distribution whose probability of
  going over the line matches the vig-free over price. The coefficient of
  variation (CV) is 0.62 for receiving and 0.58 for rushing, *measured* from
  backtest outcomes. An earlier guess of 0.85 inflated yards ~15–20%.
- **Receptions:** Poisson mean matching the over probability.
- **Anytime TD:** strip about 10% hold from the implied probability; expected
  TDs = −ln(1 − p).

The "market view" is the model's stat line with available prop stats
substituted in. The published projection is 50% model + 50% market view.
`model_half` in the JSON keeps the model-only number for transparency.

### 3.4 Range and boom (`Distribution`)

Quantile models (sklearn `GradientBoostingRegressor`, quantile loss) predict
the 10th/25th/50th/75th/90th percentiles of **outcome ÷ projection**, from:
projection, position, the player's volatility, aDOT, TD share of the
projection, and snap %. They're trained on **out-of-sample backtest
predictions** (`model_data/oos.parquet`), so the ranges reflect how wrong the
model really is, not how well it fits its own training data. Predicting the
ratio (not raw points) lets ranges scale correctly for stars.

An earlier version fit raw-point quantiles on training data. The ranges
collapsed (many WRs showed a 25th percentile of 0, and stars all got identical
ranges). Don't go back to that.

Boom % = P(points > 20), interpolated between the quantiles with an
exponential tail above the 90th percentile.

### 3.5 Explanations ("Why")

For each factor group, the model re-predicts the player with that group's
features set to a **neutral baseline**; the difference in half-PPR points is
the factor's effect. Groups and baselines live in `publish.GROUPS` and
`publish.explain`; definitions shown to users are in `publish.FACTOR_KEY`.

| Factor | Features | Neutral baseline |
|---|---|---|
| Game environment | spread, total, implied points | Position median over last 2 seasons |
| Opponent defense | all `def_*` | Position median |
| Opponent DB injuries | `db_out_*` | 0 (healthy) |
| Teammate absences | `vac_*` | 0 |
| QB vs. recent QB play | `qb_epa`, `qb_change`, `qb_upgrade` | `qb_epa` = team's recent passing EPA, no change, upgrade 0 |
| Weather | temp, wind | 65°F, 5 mph |
| Own injury status | report, practice | Healthy, full practice |

**Lesson learned (the QB baseline):** the first version compared the starter
against a hypothetical league-median veteran with 40 starts. That made Drake
London show "−0.3 Starting QB" in a week when Michael Penix Jr. (good, but only
15 career games) replaced the backups (Cooper Rush, Jack Strand) who
depressed London's recent stats. Users read a factor as "compared to what this
player has been dealing with." **Baselines should match that intuition**
wherever possible.

Effects are not strictly additive (the model has interactions), and
everything not listed, mainly the player's own role, is the baseline
projection itself. The page says so.

### 3.6 The page (`site/index.html`)

Static HTML plus vanilla JS, no build step. It reads `data/rankings.json` and
offers:

- Flex/RB/WR/TE tabs, a PPR/Half/Std toggle and search
- Each player's range bar, boom %, Why chips, and a detail row with the stat
  line, Vegas lines and a factor table with this week's inputs
- Head-to-head: samples each player's distribution 20,000 times (assumes
  independence; teammates are actually correlated)
- A "What the factors mean" key
- A data-freshness panel under the title: "Last rebuilt" (red) with the next
  scheduled run, then one cell per source with its own timestamp: injuries
  (Sleeper fetch time), weather (forecast time), game lines (nflverse's actual
  publish time, from its GitHub release asset), Vegas props (pull time, or a
  hollow dot and "arriving" with the next pull time) and stats (latest game
  included). Hovering a cell explains the source. Deliberately restrained:
  red only on the rebuild time, two dot states, no traffic-light colors.

Preferences are stored in `localStorage` (wrapped in try/catch). Light and
dark themes come from CSS tokens.

## 4. Operations

- **Hosting:** public GitHub repo `oaksilk/ff-projections`; GitHub Pages at
  https://oaksilk.github.io/ff-projections/. The repo must stay public for
  free Pages.
- **Schedule:** defined once, in Eastern time, as `SCHEDULE_ET` in
  `ffmodel/config.py`. There are 11 runs a week:
  - Nightly 7:05 PM: injury news, plus Thursday/Sunday/Monday-night inactives
  - Tue 10:00 AM: last week's stats are final
  - Sun 9:00 AM: morning injury news
  - Sun **11:45 AM**: just after 1pm inactives; the **only Vegas props pull**
  - Sun 3:00 PM: after late-afternoon inactives

  GitHub cron is UTC-only, so `weekly.yml` lists every slot at both its
  daylight and standard-time UTC offset. `scripts/gate.py` lets through only
  the firing that lands within 50 minutes after a real ET slot. The same list
  drives the page's "Next" time. **If you change the schedule, change both
  `SCHEDULE_ET` and the crons.**
- Each run commits `site/data/` (including `history/{season}_wNN.json`, every
  week's published projections, for future accuracy tracking) and
  `data/props/`.
- **Odds API budget:** 500 credits/month free. One full-slate pull is ~4
  credits per game (~60 per week). Only the Sunday run fetches. Raw props are
  committed in `data/props/` with a `fetched_at` time; any run within 3 hours of
  a pull reuses it instead of re-spending. One free key is enough (~260 of 500
  credits/month). Don't rotate extra free keys to exceed the limit (likely
  against The Odds API's terms); use their paid tier if more pulls are needed.
  Never add more prop markets or runs without redoing this math.
- **Secrets:** `ODDS_API_KEY` is a repo secret; locally it lives in `.env`
  (gitignored). Never commit the key.
- **Yearly:** after each season, rerun `scripts/backtest.py` (and add the new
  season to its `seasons` tuple) to refresh `model_data/oos.parquet` and
  `reports/backtest.txt`.

## 5. Validation (how we know it works)

`scripts/backtest.py` replays 2022–2025, retraining every 6 weeks on
everything before that week, and compares against FantasyPros weekly expert
consensus rankings (ECR, the Friday snapshot archived by nflverse). Only
players who were ranked by ECR *and* played are compared, so the model gets no
credit for knowing about inactives.

Main metric: **pairwise accuracy**. Over all pairs of top players (top 48 WR,
36 RB, 18 TE by ECR) in a week, the share where the ranking ordered them the
same as actual PPR points. This is literally start/sit accuracy.

Results at launch (`reports/backtest.txt` has the latest; adding `qb_upgrade`
moved the overall figure to 61.7%):

| | Model | Expert consensus | Recent-average baseline |
|---|---|---|---|
| All | 61.8% | 61.6% | ~58% |
| RB | 63.2% | 63.0% | |
| TE | 61.0% | 60.0% | |
| WR | 61.1% | 61.9% | |

Boom % is well calibrated (predicted 22% → actual 21%; 38% → 39%). The
10/25/50/75/90 percentiles hold within about 1 point for players projected
8+ points.

Experiments that informed the design (2025 holdout): in-season retraining
helped (+0.7 pts pairwise); stronger regularization +0.3; averaging the
component and direct models +0.2. Fewer features hurt.

**Rule for changes:** any change to features or models must be followed by a
backtest. Compare against the table above; don't ship regressions.

## 6. Known limitations and ideas

- **Routes run per dropback** (best WR/TE usage stat) isn't available
  in-season from free sources (nflverse participation data lags a season).
  Snap share is the proxy.
- **WR is the weakest position** against experts.
- **Head-to-head ignores correlation** between teammates.
- **The prop blend weight (50%) is a judgment call**; there's no free
  historical prop data to tune it.
- **Neutral-site games** get no weather. Retractable roofs are treated as
  closed.
- **Mapping quirks:** Sleeper team codes are normalized (`LAR→LA`, etc.) in
  `live.SLEEPER_TEAMS`. A missing mapping silently drops a team's players;
  check counts in the run log ("N skill players expected active").
- **Roadmap ideas:**
  - Add QB
  - "How did last week go" accuracy panel (history JSON already saved)
  - Weekly scorecard vs. FantasyPros consensus
  - Comparison against individual top experts (see below)

### Expert comparison: what's available

- **FantasyPros consensus (ECR):** free via nflverse, archived weekly since
  2020. This is the current control.
- **Individual experts:** Yahoo's weekly rankings are a FantasyPros partner
  widget, so FantasyPros is the hub for nearly all named experts. Their
  individual weekly rankings need FantasyPros API access or a premium
  account. Don't scrape the partner widget without permission.
- **FantasyPros accuracy awards:** the most accurate in-season rankers were
  Justin Boone (Yahoo, 2025), Tyler Orginski (2024) and Sean Koerner (2023).
  Patrick Thorman (Establish the Run) and Jeff Ratcliffe (FTN) were top 10 in
  all three years.

## 7. Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-10-02 | Python + scikit-learn, no LightGBM | LightGBM needs a system OpenMP library on macOS; HGB is equivalent here |
| 2026-10-02 | Separate public repo, GitHub Actions + Pages | Free, zero-maintenance hosting; the parent CloakandKernel repo had no remote |
| 2026-10-02 | Props only on the Sunday run | Fits the 500-credit free tier with headroom |
| 2026-10-02 | Distribution from out-of-sample ratio quantiles | In-sample raw quantiles collapsed (see §3.4) |
| 2026-10-02 | Yards prop CV 0.62/0.58 | Measured, replacing a guessed 0.85 |
| 2026-10-02 | QB factor baseline = team's recent QB play; added `qb_upgrade` | Owner feedback on Drake London / Penix (see §3.5). Backtest was neutral (61.7% vs 61.8%, within noise); kept because it gives the model the right signal for QB returns |
| 2026-10-02 | Sunday run moved 11:15 → 11:45 ET; page shows "updated" (red) and "next update" | Earlier run fired before inactives were posted (owner noticed via Jadarian Price going Out between runs) |
| 2026-10-02 | Schedule moved to ET slots (11/week, nightly injury refresh, extra Sunday runs) + per-source freshness panel | Owner noted one "updated" time hid that props are weekly while injuries change hourly; showing per-source timestamps also surfaces the rigor under the hood |
| 2026-10-02 | Factor key + per-factor "input this week" text on the page | Owner asked what e.g. "+1.4 Game environment" means; definitions live in `publish.FACTOR_KEY` |
