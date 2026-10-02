"""Static configuration: seasons, scoring, stadiums."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"
OUTPUT_DIR = ROOT / "site"
PROPS_DIR = ROOT / "data" / "props"  # committed, so reruns never re-spend API credits

FIRST_SEASON = 2016  # training history start
POSITIONS = ["RB", "WR", "TE"]

# Points per stat. Receptions differ by format; everything else is shared.
SCORING = {
    "ppr": {"rec": 1.0},
    "half": {"rec": 0.5},
    "std": {"rec": 0.0},
}
YD_PT = 0.1
TD_PT = 6.0
FUMBLE_PT = -2.0

# Outdoor stadiums keyed by home team -> (lat, lon). Teams not listed play
# under a dome or retractable roof, which we treat as weather-neutral.
OUTDOOR_STADIUMS = {
    "BUF": (42.7738, -78.7870), "NE": (42.0909, -71.2643),
    "NYG": (40.8135, -74.0745), "NYJ": (40.8135, -74.0745),
    "PHI": (39.9008, -75.1675), "WAS": (38.9078, -76.8645),
    "BAL": (39.2780, -76.6227), "PIT": (40.4468, -80.0158),
    "CLE": (41.5061, -81.6995), "CIN": (39.0955, -84.5161),
    "CHI": (41.8623, -87.6167), "GB": (44.5013, -88.0622),
    "KC": (39.0489, -94.4839), "DEN": (39.7439, -105.0201),
    "SEA": (47.5952, -122.3316), "SF": (37.4033, -121.9694),
    "TEN": (36.1665, -86.7713), "JAX": (30.3239, -81.6373),
    "MIA": (25.9580, -80.2389), "TB": (27.9759, -82.5033),
    "CAR": (35.2258, -80.8528),
}

# Injury designations that mean a player will not play.
OUT_STATUSES = {"Out", "IR", "PUP", "Sus", "NFI", "Doubtful", "COV", "DNR"}

# Scheduled runs, in US Eastern time: (weekday Mon=0..Sun=6, hour, minute, fetch_props).
# GitHub cron is UTC-only, so .github/workflows/weekly.yml lists each slot at both
# its daylight (UTC-4) and standard (UTC-5) offset, and scripts/gate.py lets only
# the one landing within GATE_MINUTES after a slot proceed. This list also drives
# the page's "next update" time. Keep the workflow crons in sync with it.
SCHEDULE_ET = [
    (1, 10, 0, False),   # Tue 10:00  last week's stats are final
    (6, 9, 0, False),    # Sun 9:00   morning injury news
    (6, 11, 45, True),   # Sun 11:45  just after 1pm inactives; Vegas props
    (6, 15, 0, False),   # Sun 3:00   after late-afternoon inactives
] + [(d, 19, 5, False) for d in range(7)]  # nightly 7:05: injuries; TNF/SNF/MNF inactives
GATE_MINUTES = 50
