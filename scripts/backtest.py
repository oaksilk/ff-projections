"""Run the walk-forward backtest and write reports/backtest.txt.

Usage: python scripts/backtest.py [--info friday|sunday] [--seasons 2022-2025] [--retrain-every 6]
Both information sets by default (~80–90 min on a laptop).
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ffmodel import backtest
from ffmodel.config import ROOT
from ffmodel.data import load_all
from ffmodel.features import INFO_SETS

ap = argparse.ArgumentParser()
ap.add_argument("--info", choices=INFO_SETS, action="append")
ap.add_argument("--seasons", help="e.g. 2022-2025")
ap.add_argument("--retrain-every", type=int, default=6)
ap.add_argument("--out", default=str(ROOT / "reports" / "backtest.txt"))
args = ap.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
seasons = None
if args.seasons:
    a, b = map(int, args.seasons.split("-"))
    seasons = tuple(range(a, b + 1))
d = load_all()
p, weekly, oos = backtest.run(d, seasons, tuple(args.info or INFO_SETS), args.retrain_every)

cache = ROOT / "data" / "cache"
p.to_parquet(cache / "backtest_preds.parquet", index=False)
weekly.to_parquet(cache / "backtest_weekly.parquet", index=False)
# The live range model learns from out-of-sample errors under the information
# set the Sunday run uses (all predicted seasons, including warm-up).
if "sunday" in oos and args.out == str(ROOT / "reports" / "backtest.txt"):
    (ROOT / "model_data").mkdir(exist_ok=True)
    oos["sunday"].to_parquet(ROOT / "model_data" / "oos.parquet", index=False)
text = backtest.summarize(p, weekly)
print(text)
Path(args.out).parent.mkdir(exist_ok=True)
Path(args.out).write_text(text + "\n")
