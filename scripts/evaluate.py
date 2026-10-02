"""Score finished weeks that have snapshots but no report yet.

Usage:
  python scripts/evaluate.py --pending   list unscored weeks (standard library only; prints pending=true/false)
  python scripts/evaluate.py             score every finished, unscored week; writes reports/live/
                                         and lists new report paths in reports/live/.new (for the workflow)
"""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    # Import lazily so --pending works before dependencies are installed.
    import importlib.util
    spec = importlib.util.spec_from_file_location("cfg", ROOT / "ffmodel" / "config.py")
    if "--pending" in sys.argv:
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        live = ROOT / "reports" / "live"
        weeks = [w for w in sorted(cfg.SNAPSHOT_DIR.glob("*/w*")) if any(w.glob("*.json"))
                 and not (live / f"{w.parent.name}_{w.name}.md").exists()]
        print(f"pending={'true' if weeks else 'false'}")
        return

    from ffmodel import evaluate
    from ffmodel.data import load_all

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("evaluate")
    todo = evaluate.pending()
    new = []
    if todo:
        d = load_all()
        for season, week in todo:
            if not evaluate.week_final(d, season, week):
                log.info("%s week %s not final yet", season, week)
                continue
            out = evaluate.evaluate_week(d, season, week)
            if out:
                log.info("wrote %s", out)
                new.append(str(out.relative_to(ROOT)))
    evaluate.LIVE_DIR.mkdir(parents=True, exist_ok=True)
    (evaluate.LIVE_DIR / ".new").write_text("\n".join(new))


if __name__ == "__main__":
    main()
