"""Run the walk-forward backtest and write reports/backtest.txt."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ffmodel import backtest
from ffmodel.config import ROOT
from ffmodel.data import load_all
from ffmodel.features import build_features

logging.basicConfig(level=logging.INFO)
d = load_all()
df = build_features(d)
p, weekly, oos = backtest.run(df, d)
(ROOT / "model_data").mkdir(exist_ok=True)
oos.to_parquet(ROOT / "model_data" / "oos.parquet", index=False)
p.to_parquet(ROOT / "data" / "cache" / "backtest_preds.parquet", index=False)
weekly.to_parquet(ROOT / "data" / "cache" / "backtest_weekly.parquet", index=False)
text = backtest.summarize(p, weekly)
print(text)
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / "backtest.txt").write_text(text + "\n")
