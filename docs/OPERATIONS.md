# Operations

How the system runs week to week and year to year, and what to do when something
breaks. Everything here is automatic; the owner's only job is reading the weekly
email. The *why* behind these choices is in `DESIGN.md` (§4 and the decision log).

Exact run times live in one place: `SCHEDULE_ET` (plus `SNAPSHOT_SLOTS`, the
catch-up and injury-check settings, and the season-window settings) in
`ffmodel/config.py`. The workflow crons don't encode slot times any more (see
"How runs are triggered"), so changing a slot only means editing `SCHEDULE_ET`,
then the README schedule table and this file.

## In-season week

| When (ET) | Run | Produces |
|---|---|---|
| Nightly 7:05 PM | Injury news; Thu/Sun/Mon-night inactives | Updated rankings on the site |
| **Fri 7:05 PM** | Same, after the final injury report | **Friday snapshot** `snapshots/<season>/wNN/fri.json` |
| Sun 9:00 AM | Morning injury news | Updated rankings |
| **Sun 11:15 AM** | First **Vegas props** attempt | Updated rankings with props |
| **Sun 11:45 AM** | 1pm inactives, weather, props retry (reuses 11:15's pull) | **Sunday snapshot** `.../sun.json` |
| Sun 3:00 PM | Late-afternoon inactives | Updated rankings |
| **Tue 10:00 AM** | Last week's stats final | Rankings for the new week; scorecard graded; **evaluation report emailed** |
| Hourly, 8 AM–11 PM | Only if injury statuses changed since the last build | Updated rankings (no props) |

Every build is the `build` job of `weekly.yml` (rankings → archive → scorecard →
commit → deploy to Pages), followed by the `evaluate` job.

### How runs are triggered

GitHub's cron often fires late (we saw 2h46m and 4h41m on 2026-10-02) and sometimes
not at all, worst at the top of the hour. So:

- `weekly.yml` fires a cheap **gate check** at :17 and :47 past every hour (`CHECK_MINUTES`).
- `scripts/gate.py` builds when **a slot in `SCHEDULE_ET` has passed and hasn't been
  served yet**, up to `MAX_LATE_HOURS` (6) late. Several missed slots are served by one
  build (props if any of them pulls props, the latest snapshot label).
- Between slots, from 8 AM to 11 PM ET, it fetches ESPN's injury list (~360 KB) and
  builds if statuses for QB/RB/WR/TE/DBs changed since the last build (at most once per
  55 min, never pulls props).
- After a build succeeds, `gate.py --record` writes `data/run_state.json` (last served
  slot, last build time, injury fingerprint), committed with the rankings. A failed
  build doesn't record, so the next check (≤30 min) retries it.
- Off-season, the gate exits immediately.

**Odds API budget:** props are pulled by the Sun 11:15 slot; 11:45 and any catch-up run
reuse a pull from the last 8 hours (`PROPS_REUSE_HOURS`), so it's still one pull
(~60 credits) per Sunday.

### Forecast archive

- Every build writes `site/data/history/<season>_wNN/<UTC time>.json` (created with
  exclusive mode, never overwritten): model-only and published (Vegas-adjusted)
  projections, ranges and boom for all three formats, the recent-average baseline, the
  information set, git commit, slot/trigger, scoring rules, who was ruled out, prop
  coverage, a hash of the raw props file, and input freshness.
- The same build writes `data/archive/<season>_wNN/<same time>.json` with FantasyPros
  ECR (nflverse), Sleeper (Rotowire) and ESPN projections. **Repo only, not deployed**:
  we don't republish other companies' numbers. Sleeper and ESPN are unofficial
  endpoints; if they fail the file records the error and the run continues.
- The old single `site/data/history/<season>_wNN.json` (overwritten each run) is no
  longer written; the one from 2026 Week 4 stays as-is.

### Scorecard

- `ffmodel/scorecard.py` runs at the end of every build and grades any archived week
  that is final and not yet graded. Output: `site/data/scorecard.json` (accuracy
  figures only), the page's Scorecard section, and a section in the weekly email.
- Sources: our published rankings, our model alone, the model + expert blend,
  FantasyPros experts, Sleeper, ESPN, recent average.
- Each player is graded on each source's **last archive file before his own
  kickoff**. The decision pool is every player in any source's top 48 WR / 36 RB /
  18 TE; recommended players who didn't play count as 0.
- Metrics: start/sit pairwise accuracy (ties half credit), close calls, points lost,
  flex, MAE, range coverage, pinball loss, boom and head-to-head calibration; season
  to date with 95% intervals vs. the experts.
- **Re-grade a week:** remove its entry from `site/data/scorecard.json` `weeks`; the
  next build regrades it. Never edit the archive files.

### Snapshots

- A snapshot is our published rankings (projection, 10–90% range, boom % for PPR,
  half and std) **plus FantasyPros expert consensus (ECR) downloaded at the same
  moment**, plus metadata: when it was taken, the model commit, and data freshness.
- Snapshots are **written once and never overwritten or regenerated**. If a run
  repeats, the existing file wins. They are the record of what we said before
  kickoff; recomputing one later would leak post-game information.
- ECR is saved with the snapshot because the nflverse ECR feed only keeps
  occasional history, so a later download may not match what the experts said at
  snapshot time. `meta.ecr_scrape_date` shows how fresh the feed was. If nflverse
  hasn't refreshed between Friday and Sunday, both snapshots carry the same ECR.
- GitHub cron can start late. A snapshot records when it was actually taken, and
  the evaluation drops any player whose game had already kicked off by then
  (e.g. London games, or the whole 1pm slate if the Sunday run slipped past 1:00).

### Weekly evaluation

- The `evaluate` job runs after every in-season build. It scores any week that
  has snapshots but no report, once every game that week is final and in the
  nflverse stats. Normally that's Tuesday 10 AM. If stats are late or the run
  fails, the next nightly run picks it up.
- Output: `reports/live/<season>_wNN.md` (the report), `reports/live/scores.csv`
  (one row per week × snapshot × position), and `reports/live/SUMMARY.md` (season
  to date).
- **Delivery:** each new report is opened as a GitHub issue assigned to the repo
  owner, so GitHub emails it to the owner's notification address. Closing the
  issues is optional.
- **Metrics** (same rules as the backtest): pairwise start/sit accuracy and
  Spearman correlation vs. actual PPR points on the experts' top 48 WR / 36 RB /
  18 TE who played, for us and for ECR. Point accuracy (average miss, lean),
  10–90% range coverage (target 80%), and boom calibration for all three formats.
  One week is noise; judge on the season average.
- The last regular-season week's report includes a season wrap-up.

## The year

| When | What happens | How |
|---|---|---|
| ~9 days before the opener → 3 days after the last regular-season game | Scheduled runs active | `gate.py` reads the nflverse schedule; nothing to update yearly |
| Rest of the year (playoffs, off-season, preseason) | Scheduled runs skip themselves (no commits, no props, no emails) | Same gate. If the schedule can't be fetched, it fails open and runs |
| 1st of every month | Re-enable both workflows | `maintenance.yml` → `keepalive`. GitHub disables schedules in public repos after 60 days without activity and does not re-enable them; this resets the clock |
| Aug 1 | Re-run the backtest on the last four completed seasons; refresh `model_data/oos.parquet` and `reports/backtest.txt`; email before/after results | `maintenance.yml` → `refresh` (~20–40 min) |

The model itself retrains on all history every run, so new-season data is picked up
automatically. The August refresh only updates the range/boom calibration and the
reference accuracy numbers.

Playoffs are skipped on purpose: the model only projects regular-season games, and
season-long leagues end by week 17/18. Preseason is skipped because box scores are
noise. Depth-chart and injury changes arrive via Sleeper once runs resume.

## What the owner does

Nothing routine. Read the Tuesday email if you like. If GitHub emails about a
**failed** run, paste it into a Claude session.

## When something breaks

- **Failed run email:** open the run link and read the failing step's log. A
  `build` failure means no new rankings that slot. An `evaluate` failure doesn't
  block rankings; fix it and the next run catches up automatically.
- **Runs late or skipped:** normal; the gate catches up within 30 minutes of the next
  check. `gh run list --workflow weekly.yml` shows each check; the build step's commit
  message and the gate log line say why it ran (`slot Sun 11:45 ET (35 min after)`,
  `injury statuses changed`) or didn't.
- **Missed snapshot:** it can't be recreated after kickoff. The report notes the
  gap and scores whichever snapshot exists. To take one manually before kickoff:
  Actions → Weekly rankings → Run workflow → `snapshot: fri|sun`.
- **Test that emails arrive:** Actions → Maintenance → Run workflow →
  `test-email`. Issues opened by the bot email the owner; issues the owner opens
  themselves do not.
- **Scorecard or third-party archive failed:** never blocks rankings. A warning in the
  build log says why; the next build retries grading.
- **Season didn't start:** check `gh run list --workflow weekly.yml`. If the
  workflow shows as disabled, run Maintenance → `keepalive` or enable it in the
  Actions tab.
- **Re-score a week** (e.g. after an evaluation bug fix): delete its
  `reports/live/<season>_wNN.md` and the next in-season run regenerates it.
  Never edit snapshots.

## Commands

```bash
.venv/bin/python scripts/run_weekly.py --snapshot fri   # rankings + snapshot (refuses to overwrite)
.venv/bin/python scripts/evaluate.py --pending          # any unscored weeks?
.venv/bin/python scripts/evaluate.py                    # score finished weeks
gh workflow run maintenance.yml -f task=test-email      # send a test email
```
