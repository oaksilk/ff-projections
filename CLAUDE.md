# CLAUDE.md

Weekly fantasy football start/sit rankings (RB/WR/TE; PPR, half and std),
published by GitHub Actions to https://oaksilk.github.io/ff-projections/.

**Read `docs/DESIGN.md` before changing anything.** It holds the goals,
philosophy, architecture, validation results and decision log. Update it
(especially the decision log) when you change behavior.

## Rules

- **Validate model changes.** Any change to features or models: run
  `.venv/bin/python scripts/backtest.py` (~20 min) and compare
  `reports/backtest.txt` against the results in DESIGN.md §5. Don't ship
  regressions without the owner's OK.
- **No leakage.** Features may only use information available before
  kickoff. Training and live features share one code path in `features.py`;
  keep it that way.
- **Odds API budget.** 500 credits/month; ~60 per Sunday pull. Don't add
  markets, runs or `--props` calls without redoing the math. Cached props live
  in `data/props/` (committed).
- **Never commit `.env`** or the API key. In CI it's the `ODDS_API_KEY` secret.
- **Explanations must match intuition.** Factor baselines should answer "compared to
  what this player has been dealing with" (see DESIGN.md §3.5). The owner is
  non-technical; page copy should be plain English.
- **Keep it free and hands-off.** No paid services or manual weekly steps.
- **Read `docs/OPERATIONS.md` before touching the schedule, snapshots,
  evaluation or off-season logic**, and update it in the same change.
- **Snapshots are immutable.** Never overwrite, edit or regenerate files in
  `snapshots/`; they are the pre-kickoff record. Expert comparisons use the ECR
  saved *inside* each snapshot, never a fresh download.

## Commands

```bash
.venv/bin/python scripts/run_weekly.py           # build this week's rankings (no new props)
.venv/bin/python scripts/run_weekly.py --props   # also fetch Vegas props (spends credits)
.venv/bin/python scripts/backtest.py             # walk-forward validation vs FantasyPros ECR
.venv/bin/python scripts/evaluate.py             # score finished weeks' snapshots (reports/live/)
```

Preview locally: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`,
then serve `site/` with any static server.
