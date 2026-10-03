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
3. **Opportunity and efficiency as features, not a pipeline.** The model
   predicts each player's outcome *directly*, using team volume (Vegas implied
   total, pace, pass rate), player share and efficiency as inputs. There is no
   explicit team-volume projection divided among players, and nothing forces a
   team's player projections to add up to a team total. (An earlier version of
   this document described a top-down pipeline the code never had; corrected
   2026-10-03 after the external review. Team reconciliation is listed in §6
   as possible future work.)
4. **The betting market is the strongest single signal.** When Vegas player
   props are available (Sunday's pull onward), half of each gap between the
   props and the model's own stat line is added to the projection. The market
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
| `ffmodel/publish.py` | Vegas adjustment, factor explanations, plain-English context, JSON output |
| `ffmodel/archive.py` | Immutable per-run forecast archive + third-party projections (ECR, Sleeper, ESPN) |
| `ffmodel/scorecard.py` | Grades every archived source on finished weeks (`site/data/scorecard.json`) |
| `ffmodel/metrics.py` | Shared metrics: pairwise (ties half credit), points lost, bootstrap CIs, pinball, H2H |
| `ffmodel/backtest.py` | Walk-forward evaluation against FantasyPros ECR, two information sets |
| `scripts/run_weekly.py` | The weekly job (what GitHub Actions runs) |
| `scripts/backtest.py` | Full backtest (~1.5 h); regenerates `model_data/oos.parquet` and `h2h_calibration.json` |
| `scripts/gate.py` | Decides whether a twice-hourly check should build (catch-up slots, injury changes) |
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
- **QB:** the schedule's listed starter (`home_qb_id`/`away_qb_id`), never
  "who threw the most passes" (that can be an in-game replacement). Note
  nflverse fills this with whoever actually started, so a surprise start is
  still known in hindsight; that is rare and usually public by Friday.
  Features: the starter's EWMA EPA per dropback (`qb_epa`), career games,
  `qb_change` (different starter than last game), and `qb_upgrade`
  (`qb_epa` minus the team's recent passing EPA, i.e. "is this QB better than
  the QB play behind this player's recent stats").
- **Absences:** `vac_tgt` and `vac_car` are the recent target and carry
  shares of teammates who are out. `db_out_n` and `db_out_share` count
  opposing regular defensive backs (≥50% of snaps recently) who are out.
  "Out" means known unavailable **before kickoff**, never "took no snaps".
  Historical rows use one of two information sets (`features.INFO_SETS`):
  - **friday:** injury report Out/Doubtful, or not on the team's weekly
    roster as active/inactive (IR, released, traded, suspended...).
  - **sunday:** friday plus gameday inactives (`INA`, nflverse 2019+; earlier
    seasons fall back to friday).
  Live rows use injury reports and Sleeper status at run time. A player
  flagged out who played anyway doesn't count his own share as vacated.
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

### 3.3 Vegas adjustment (`publish.blend_market`)

Props pulled: receptions, receiving yards, rushing yards, anytime TD. Each is
converted to an expected value:

- **Yards:** solve for the mean of a gamma distribution whose probability of
  going over the line matches the vig-free over price. The coefficient of
  variation (CV) is 0.62 for receiving and 0.58 for rushing, *measured* from
  backtest outcomes. An earlier guess of 0.85 inflated yards ~15–20%.
- **Receptions:** Poisson mean matching the over probability.
- **Anytime TD:** strip about 10% hold from the implied probability; expected
  TDs = −ln(1 − p).

The "market view" is the model's own component stat line with available prop
stats substituted in. The published projection is

    full model projection + 0.5 × (score(market view) − score(model component view))

so **props that agree with the model change nothing**; only disagreement moves
the number. (Until 2026-10-03 it was 50% full projection + 50% market view,
which also shifted weight from the direct model to the component model, so
merely having a matching prop could move a player, e.g. 12 → 11 points.) The
per-format adjustment is published as `<fmt>.vegas` and shown as the "Vegas
adjustment" line; model-only numbers are `<fmt>.model` and archived separately.

These conversions rest on assumptions (fixed CVs, Poisson counts, a flat 10%
TD hold, a 0.5 weight) that can't be tuned without historical props. The live
scorecard compares "model alone" with "model + Vegas" to test the adjustment
prospectively.

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
features set to a **neutral baseline**; the difference is the factor's effect,
computed **separately for PPR, half-PPR and standard** (the page shows the one
you picked). Groups and baselines live in `publish.GROUPS` and
`publish.explain`; definitions shown to users are in `publish.FACTOR_KEY`.

| Factor | Features | Neutral baseline |
|---|---|---|
| Game environment | spread, total, implied points | Position median over last 2 seasons |
| Opponent defense | all `def_*` | Position median |
| Opponent DB injuries | `db_out_*` | 0 (healthy) |
| Teammate absences | `vac_*` | 0 |
| QB vs. recent QB play | `qb_epa`, `qb_change`, `qb_upgrade` | `qb_epa` = team's recent passing EPA, no change, upgrade 0 |
| Weather | temp, wind | 65°F, 5 mph (0 effect when no forecast) |
| Own injury status | report, practice | Healthy, full practice |

These are **sensitivities, not causes**: one input reset, everything else held
fixed. The model has interactions, so effects don't sum to the projection and
can overlap. The page says so. Everything not listed, mainly the player's own
role, is the baseline projection itself.

The **Vegas adjustment** is shown as its own line (and chip when ≥0.3 points).
It is not a sensitivity: it is the actual change made to the model's number
(§3.3), with the props and the model's matching stats in the "input" column.

**Lesson learned (the QB baseline):** the first version compared the starter
against a hypothetical league-median veteran with 40 starts. That made Drake
London show "−0.3 Starting QB" in a week when Michael Penix Jr. (good, but only
15 career games) replaced the backups (Cooper Rush, Jack Strand) who
depressed London's recent stats. Users read a factor as "compared to what this
player has been dealing with." **Baselines should match that intuition**
wherever possible.

**Lesson learned (same starter):** Jaguars receivers showed a small negative
QB factor though Trevor Lawrence had started all season. The factor compares
the starter's longer-run efficiency (+0.13) with the offense's recent passing
(+0.23): he had been playing above his norm, and the model expects some
regression. That's a reasonable bet but read like a QB change. The input text
now says which case applies: "New starter X" vs. "Same starter, X, playing
above/below his usual level lately; the model expects some regression /
a bounce-back."

### 3.6 The page (`site/index.html`)

Static HTML plus vanilla JS, no build step. It reads `data/rankings.json` and
offers:

- Flex/RB/WR/TE tabs, a PPR/Half/Std toggle and search
- Each player's range bar, boom %, Why chips, and a detail row with the stat
  line, Vegas lines, model-alone number and a factor table (plus the Vegas
  adjustment line) with this week's inputs. On phones the range bar and the
  top Why chip move into a compact line under the player's name.
- Head-to-head: samples each player's distribution 20,000 times, then applies
  the backtest's calibration (`h2h_slope`, §5) so a stated 80% means 80%.
  Shows "X scores more in N% of simulated games" and the projected points
  gap. No "Start X" verdict: highest projection, highest chance to outscore
  one player, and highest chance your lineup wins can disagree, and it doesn't
  know your lineup. Teammates and same-game pairs get a warning (the
  simulation assumes independence).
- A "What the factors mean" key
- A Scorecard section (season to date, from `data/scorecard.json`)
- A data-freshness panel under the title: "Last rebuilt" with the next
  scheduled run, then one cell per source with its own timestamp: injuries
  (Sleeper fetch time), weather (forecast time), game lines (nflverse's actual
  publish time, from its GitHub release asset), Vegas props (pull time, or the
  next pull time while waiting) and stats (latest game included). Each has a
  status dot, computed in the browser at view time so an open tab ages
  honestly:
  - **Green (filled):** current for that source's cadence. Run-refreshed
    sources are green if under 26h old, i.e. a nightly run happened.
  - **Yellow (hollow ring):** waiting/on schedule. Props before Sunday's pull;
    games played but stats not yet ingested; a source 26–50h old; a run a
    little late.
  - **Red (filled):** stale. Over 50h old, a missed scheduled run, a missed
    props pull, or every weather forecast failing.

  A missing timestamp for data that *was* loaded (game lines, injuries) shows
  **"Unverified"** in yellow, not "unavailable". Weather says "Indoors" only
  when no game needs a forecast; failed or partial forecasts say so, and those
  games show "Forecast unavailable" (never nan°F).

  Hovering a cell gives the state and an explanation. There's no other color
  in the panel and no source labels (the owner removed them for clarity).
  "Last rebuilt" text is neutral; its dot carries the status (red text would
  read as an alarm). Thresholds live in `sourceStates()` in `index.html`.

Preferences are stored in `localStorage` (wrapped in try/catch). Light and
dark themes come from CSS tokens.

## 4. Operations

- **Hosting:** public GitHub repo `oaksilk/ff-projections`; GitHub Pages at
  https://oaksilk.github.io/ff-projections/. The repo must stay public for
  free Pages.
- **Schedule:** defined once, in Eastern time, as `SCHEDULE_ET` in
  `ffmodel/config.py`. There are 12 slots a week:
  - Nightly 7:05 PM: injury news, plus Thursday/Sunday/Monday-night inactives
  - Tue 10:00 AM: last week's stats are final
  - Sun 9:00 AM: morning injury news
  - Sun 11:15 AM: first **Vegas props** attempt
  - Sun **11:45 AM**: just after 1pm inactives; props retry (reuses a pull from
    the last 8 hours, so still one pull a week)
  - Sun 3:00 PM: after late-afternoon inactives

  Plus an hourly injury check (8 AM–11 PM) that rebuilds only if ESPN's
  injury statuses changed. GitHub cron fires late and sometimes not at all,
  so `weekly.yml` fires a cheap gate check at :17 and :47 every hour and
  `scripts/gate.py` builds any slot that has passed but wasn't served (up to
  6 hours late), recording served slots in `data/run_state.json`. The same
  list drives the page's "Next" time. Details in OPERATIONS.md.
- **Information set:** runs from Sunday 11:30 ET onward train on the SUNDAY
  information set (gameday inactives known); every other run uses FRIDAY
  (`config.info_set`, §3.1).
- Each run commits `site/data/` (rankings, the immutable per-run archive
  `history/{season}_wNN/{time}.json`, the scorecard), `data/archive/`
  (third-party projections, not deployed), `data/props/` and
  `data/run_state.json`. The Friday 7:05 PM and Sunday 11:45 AM runs also
  write frozen snapshots (`snapshots/`, see OPERATIONS.md).
- Scheduled runs only proceed in season: from 9 days before the first
  regular-season game to 3 days after the last (`gate.py`).
- **Odds API budget:** 500 credits/month free. One full-slate pull is ~4
  credits per game (~60 per week). Only the Sunday 11:15/11:45 slots fetch.
  Raw props are committed in `data/props/` with a `fetched_at` time; any run
  within 8 hours of a pull reuses it instead of re-spending. One free key is enough (~260 of 500
  credits/month). Don't rotate extra free keys to exceed the limit (likely
  against The Odds API's terms); use their paid tier if more pulls are needed.
  Never add more prop markets or runs without redoing this math.
- **Secrets:** `ODDS_API_KEY` is a repo secret; locally it lives in `.env`
  (gitignored). Never commit the key.
- **Yearly:** automatic. `maintenance.yml` re-runs the backtest every Aug 1 on
  the four most recent completed seasons and emails the before/after results.
- **Live evaluation, snapshots, season on/off, keep-alive:** see
  [`OPERATIONS.md`](OPERATIONS.md), the runbook for cadence and failure handling.

## 5. Validation (how we know it works)

`scripts/backtest.py` replays 2022–2025, retraining every 6 weeks on
everything before that week, and compares against FantasyPros weekly expert
consensus rankings (ECR, the Friday snapshot archived by nflverse). Only
players who were ranked by ECR *and* played are compared. It runs twice, once
per information set (§3.1): **FRIDAY** is the fair comparison with Friday
ECR; **SUNDAY** matches the live 11:45 run but knows inactives the experts
didn't. Full output: `reports/backtest.txt` (~80–90 min to regenerate).

**Main metric: pairwise accuracy.** Over all pairs of the experts' top 48 WR,
36 RB and 18 TE in a week, the share where the forecast put the higher PPR
scorer first. Tied forecasts get half credit. Each position-week counts
equally. Differences come with 95% intervals from resampling whole weeks.

Results (rerun 2026-10-03 after removing postgame information, §7):

| PPR, 2022–2025 | Model | Experts | Recent average | Model+ECR blend |
|---|---|---|---|---|
| All (FRIDAY info) | 61.6% | 61.7% | 59.3% | 62.0% |
| All (SUNDAY info) | 61.7% | 61.7% | 59.3% | 62.1% |
| RB (FRIDAY) | 62.8% | 63.1% | | |
| TE (FRIDAY) | 60.8% | 60.0% | | |
| WR (FRIDAY) | 61.2% | 62.0% | | |
| Close calls, within 5 expert places (FRIDAY) | 54.6% | 53.9% | 51.0% | 54.6% |

**What this supports, in plain words:**

- The model is **statistically tied with expert consensus** (FRIDAY: −0.08
  points, 95% interval −0.66 to +0.46). It does not beat the experts. It does
  clearly beat a recent-average baseline (+2.3 points).
- **WR is genuinely behind** the experts (−0.84, interval −1.52 to −0.17).
- **The model + ECR blend** (equal-weight average of the two ranks, fixed
  before this run) is slightly but measurably ahead of the experts: +0.32
  (interval +0.04 to +0.60) on FRIDAY. Caveat: the external review had
  already computed this on the old features, so it is not a pristine
  pre-registration. The live scorecard tests it going forward.
- **Close calls are close to coin flips for everyone** (51–55%). The 61–62%
  headline is not the chance of getting a tough lineup question right.
- Same picture in half-PPR and standard, and on cross-position flex pairs.
- Removing the leaks barely moved the numbers (old report: 61.7% vs 61.6%).
- Inactive information adds little: SUNDAY vs FRIDAY is +0.1 point.

**Ranges** (now validated chronologically: each season's range model learns
only from earlier seasons, with 2020–2021 as warm-up): for players projected
8+ points the 10–90% range holds 80.0% of outcomes (target 80%) and the
median is unbiased (49.8% below). For all players, including fringe ones,
ranges are a bit wide (86%).

**Boom %** is slightly optimistic in the middle (says 24%, happens ~22%).

**Head-to-head %** is overconfident for lopsided pairs: when it says 74%,
the favorite wins ~71%; when it says 92%, ~72–74%. Close pairs are well
calibrated. The page now applies a one-parameter correction,
P = sigmoid(0.87 × logit(raw)), fit on these pairs (`model_data/h2h_calibration.json`;
fit on 2022–24 it gives 0.85 and slightly improves 2025).

**Not tested:** the Vegas prop blend (no free historical props) and the
timing of historical game lines and weather (no timestamps). The live
numbers include the prop blend, so they are *not* validated by this table.

**Rule for changes:** any change to features or models must be followed by a
backtest. Compare against the table above; don't ship regressions.

## 6. Known limitations and next steps

- **Model is tied with expert consensus, not ahead** (§5). WR is the weakest
  position and is measurably behind.
- **The Vegas adjustment is unvalidated.** No free historical props; its 0.5
  weight, CVs, Poisson assumption and 10% TD hold are judgment calls. The live
  scorecard ("model alone" vs. "model + Vegas") is the test.
- **Head-to-head assumes independence.** Teammates and same-game players are
  correlated; the page warns. Calibration was fit on model-only backtest
  ranges; live ranges include the Vegas adjustment.
- **Historical game lines and weather have no timestamps**, so the backtest
  can't prove they match a Friday information cutoff. nflverse fills the
  schedule's starting QB after the fact, so surprise starts are known in
  hindsight (rare).
- **Routes run per dropback** (best WR/TE usage stat) isn't available
  in-season from free sources (nflverse participation data lags a season).
  Snap share is the proxy.
- **Neutral-site games** get no weather. Retractable roofs are treated as
  closed.
- **Mapping quirks:** Sleeper team codes are normalized (`LAR→LA`, etc.) in
  `live.SLEEPER_TEAMS`. A missing mapping silently drops a team's players;
  check counts in the run log ("N skill players expected active").
- **Hourly injury rebuilds fetch Sleeper's full player list each build**
  (Sleeper asks for about once a day). Typical in-season volume is a handful
  of builds a day; if Sleeper objects, cache it and rely on ESPN/nflverse for
  intra-day status.

**Deferred model work (next steps, in rough priority):**

1. **Ablations.** Measure what each feature family adds (backtest with it
   removed), so later changes are judged against known contributions.
2. **Role-change handling.** Detect promotions/demotions (snap share and
   route proxies jumping) faster than the EWMA half-lives allow; the
   biggest misses cluster here.
3. **Rookie and low-sample priors.** Better priors than draft pick + years of
   experience for players with few games (e.g. college production, depth chart).
4. **Team reconciliation.** Project team plays and pass/rush split, then
   scale player projections so teammates add up to a coherent team total
   (the structure §2.3 used to claim).
5. **WR-specific work**, since WR is where we trail the experts.
6. Better usage signals as free sources allow (routes, red-zone targets).
7. Add QB.
8. Tune or replace the Vegas weight once the scorecard has a season of data.

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
| 2026-10-02 | Freshness dots use green/yellow/red status instead of blue; source labels removed | Owner: blue dots were meaningless; status colors are intuitive. Shape (ring vs. filled) backs up color for accessibility |
| 2026-10-02 | Factor key + per-factor "input this week" text on the page | Owner asked what e.g. "+1.4 Game environment" means; definitions live in `publish.FACTOR_KEY` |
| 2026-10-02 | Frozen Friday 7:05 PM + Sunday 11:45 AM snapshots in `snapshots/`, never overwritten, with ECR saved alongside | Owner wants live accuracy tracking. Friday evening is after the final injury report (a morning snapshot would mostly test guessing the report). 11:45 rather than 11:59 because GitHub cron often starts 5–30 min late. ECR is saved at snapshot time because the nflverse feed doesn't keep every version, so a later download may not match what experts said |
| 2026-10-02 | Evaluation drops players whose game started before the snapshot was taken | Handles London games and late cron starts without leaking in-game information |
| 2026-10-02 | Weekly report emailed as a GitHub issue assigned to the owner | Free, no new service; owner wants it pushed to them. Self-heals: any in-season run scores unscored finished weeks |
| 2026-10-02 | Runs active only from 9 days before the opener to 3 days after the last regular-season game; no preseason or playoffs | Model is regular-season only, fantasy leagues end by week 17/18, preseason stats are noise. Dates come from the nflverse schedule, so nothing to update yearly |
| 2026-10-02 | Monthly keep-alive re-enables workflows | GitHub disables schedules in public repos after 60 idle days and doesn't re-enable them, which would stop the season from starting on its own |
| 2026-10-02 | Aug 1 automatic backtest refresh on the last four completed seasons | Replaces a manual yearly step; owner wants zero upkeep |
| 2026-10-03 | External review saved (`docs/reviews/2026-10-02-external-review.md`); fixing in phases | Graded C+ (modeling B). Owner agreed with nearly all of it |
| 2026-10-03 | Historical starting QB = schedule's listed starter, not most pass attempts | Most-attempts picked in-game replacements: postgame information (105 of 2,174 team-games, 2022–2025) |
| 2026-10-03 | Absences use pregame status (injury report, weekly roster status, gameday inactives), not "took no snaps"; two information sets, FRIDAY and SUNDAY | "Didn't play" isn't knowable before kickoff. FRIDAY is the fair comparison with Friday ECR; SUNDAY matches the live 11:45 run |
| 2026-10-03 | Pairwise metric gives tied forecasts half credit; added close calls, points lost, flex, all formats, week-bootstrap 95% intervals, pinball loss, H2H calibration | Zero credit for ties unfairly penalized blends; headline accuracy alone overstated how useful the model is on hard calls |
| 2026-10-03 | Range model validated chronologically (train on earlier seasons only, 2020–2021 warm-up) | Leave-one-season-out let 2022 ranges learn from 2023–2025 |
| 2026-10-03 | Model+ECR equal-weight rank blend added as a graded candidate (not published) | Defined before the rerun; it is the only candidate measurably ahead of the experts |
| 2026-10-03 | Backtest wording changed to "statistically tied with consensus" | The 95% interval for model − experts includes zero |
| 2026-10-03 | Explanations computed per scoring format; labeled sensitivities, not causes; explicit "Vegas adjustment" line | Effects were half-PPR only and shown for every format; users read them as additive causes |
| 2026-10-03 | Vegas blend changed to an adjustment: full projection + 0.5 × (market view − model's component view) | Old 50/50 mix also reweighted component vs. direct model, so a prop matching the model still moved the number (12 → 11 in the review's test) |
| 2026-10-03 | QB factor text distinguishes "new starter" from "same starter, above/below his usual level" | Owner saw a negative QB factor for Jaguars receivers with Trevor Lawrence starting all season |
| 2026-10-03 | Head-to-head: "X scores more in N% of simulated games" + points gap; no "Start X"; teammate/same-game warning; calibration slope 0.87 from the backtest | Verdict implied more than the math supports; stated 92% won ~73% in the backtest; independence ignores shared games |
| 2026-10-03 | Every run writes an immutable archive (`site/data/history/<season>_wNN/<time>.json`) with model-only and published numbers; third-party ECR/Sleeper/ESPN in `data/archive/` (not deployed) | The old weekly history file was overwritten each run; third-party numbers stay out of the public site to avoid redistributing them |
| 2026-10-03 | Prospective scorecard: all sources graded on their last forecast before each player's kickoff; recommended non-players score 0 | Owner wants the live product (including the Vegas adjustment) graded as a system, against experts and other free projections |
| 2026-10-03 | Weather: failed or neutral-site forecasts show "Forecast unavailable" with zero weather effect; all-failed never reads as "Indoors". Freshness: missing timestamp = "Unverified" (yellow) | 25 players showed nan°F; a failed fetch looked like an indoor week |
| 2026-10-03 | Phones show a compact range bar and the top Why chip under each name | Range and Why (the most-loved features) were hidden on narrow screens |
| 2026-10-03 | Gate check twice an hour at :17/:47; builds any due-but-unserved slot (≤6 h late) via `data/run_state.json`; Sun 11:15 props attempt backs up 11:45; props reuse window 8 h | Crons fired 2h46m and 4h41m late on 2026-10-02 and the on-time-only gate rejected them. Top-of-hour crons are GitHub's most delayed. One props pull per Sunday is preserved |
| 2026-10-03 | Hourly injury check (8 AM–11 PM ET): rebuild if ESPN injury statuses for QB/RB/WR/TE/DB changed (≤1 per 55 min, no props) | Breaking news appears within about an hour instead of at the next slot, at no cost |
| 2026-10-03 | Live runs train on the SUNDAY information set from Sun 11:30 ET, FRIDAY otherwise | Training history must match what the run itself can know (gameday inactives) |

