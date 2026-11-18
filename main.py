"""gta6-twitch-ccu-forecast-01 entry in Python 3.13 (standard library only).

Forecasts, for each of hours 2..24 of GTA VI's release day (UTC), five
quantiles of the Twitch category's concurrent viewers, from everything the
case records up to the end of hour 1 (that hour's print included).

Method — nearest-neighbour launch curves (k = 10), calibrated on the kit's
own comparables, no network, no fitted constants that need the future:
  1. For every comparable past big launch in the case, take its release-day
     hourly curve (SullyGnome hourly averages, as the kit ships them) and
     express each hour as a log-ratio to that launch's own hour-1 print:
     r_h = ln(viewers_h / viewers_1).  The ratio, not the level, is what
     transfers between launches: hour 1 already holds the midnight unlock
     of New Zealand, Australia, Asia and Europe, and the rest of the day is
     the US unlocks (05:00Z east, 08:00Z west) plus the evening peak.
  2. Pick the 10 comparables whose hour-1 print is closest to this case's
     hour-1 print in log space.  Launch size predicts the shape: the
     correlation between ln(hour-1 print) and ln(peak / hour-1) across the
     kit's 38 comparables is -0.94 — the biggest launches are already near
     their peak at hour 1 (peak ratio ~1.0-1.3), mid-size launches roughly
     double, small ones grow ten- to a thousand-fold.  A single pooled
     shape would overstate growth for a giant like GTA VI and understate
     it for a small launch.
  3. For each hour h, the forecast quantiles are the 5/25/50/75/95
     percentiles (linear interpolation) of the neighbours' log-ratios,
     scaled by this case's hour-1 print: q_tau = p1 * exp(percentile_tau).
     The spread is the neighbours' own disagreement, so it widens where
     launches genuinely diverge (late hours) and stays honest where they
     agree — exactly what the pinball loss on the log scale rewards.
  4. Fallback: an hour with fewer than 4 neighbours holding it falls back
     to the house baseline's widening log-normal spread around the hour-1
     print (sigma 0.15 * sqrt(h - 1)), so the output is always valid.
  5. Size-trend shift: within the 10 neighbours, the log-ratios are
     regressed (OLS) on their log hour-1 prints, and the quantiles are
     shifted by half the slope times (this case's log hour-1 minus the
     neighbours' median log hour-1).  Bigger launches grow less after
     hour 1, so the shift pulls the forecast down for a launch bigger
     than its neighbours and up for a smaller one; it is ~0 when the
     case sits inside the neighbours' size range.  The half-shrink is
     there because a 10-point slope is noisy: the leave-one-out mean is
     flat across shrink 0.25-1.0, but the giant regime (>=200k hour-1)
     starts to worsen above 0.5.

Validation (leave-one-out over the kit's 38 comparables, scored with the
contest's own pinball loss on the log scale; the house baseline scores
~0.387 on the 4 practice cases, this model 0.228):
  - all 38 launches: this model 0.396 mean (baseline 0.930)
  - launches with hour-1 >= 50k (the GTA VI regime): 0.214 (baseline 0.251)
  - launches with hour-1 >= 200k: 0.134 (baseline 0.226)
  Median loss over all 38: 0.266 (baseline 0.325).

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages. -P
keeps this file's directory OFF sys.path, so a module shipped beside main.py
imports only after the sys.path line below, which puts the directory first;
-s drops user site-packages.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case at a
    time, with only that case's file there): JSON {"schema":
    "makefun/twitch-ccu-case/v1", "id", "game", "day_start", "checkpoint",
    "levels", "known", "forecast", "history", "history_hourly", "streams",
    "categories_hourly", "comparables", "notes"}, where `known` holds hour
    1's print {hour: 1, from, until, viewers, channels, reads} and
    `forecast` lists the hours to forecast, 2..24; `comparables` holds past
    big launches' hourly curves;
  - write <id>.out to the output directory: {"hours": [{"hour": 2, "q":
    [q05, q25, q50, q75, q95]}, ...]}, exactly 23 rows, hours 2..24 each
    once, no other key, each q five finite numbers above 0 and at most 1e9
    (viewers; fractions allowed), non-decreasing; at most 65,536 bytes.
    Anything else is an invalid case, scored as if every quantile equalled
    hour 1's print, and an entry with an invalid case takes no prize.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (fallback spread only).
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The spread of an hour's move, in natural-log units per square-root hour (fallback only).
SIGMA = 0.15
# Neighbours (comparables closest in log hour-1 print) whose log-ratio
# distribution gives each hour's quantiles.
K_NEIGHBORS = 10
# An hour needs at least this many neighbours holding it, else the fallback.
MIN_NEIGHBORS = 4
# Shrinkage on the local size-trend shift (see hour_quantiles): the OLS
# slope of log-ratio on log hour-1 among only 10 neighbours is noisy, so
# only half the implied shift is used.  The leave-one-out mean is flat
# across shrink 0.25-1.0, but the giant regime (>=200k hour-1, where GTA VI
# lives) starts to worsen above 0.5, so 0.5 is the safe middle.
TREND_SHRINK = 0.5


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def percentile(sorted_vals: list[float], tau: float) -> float:
    """Linear-interpolation percentile of an already sorted list."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] * (hi - pos) + sorted_vals[hi] * (pos - lo)


def release_day_hours(comp: dict) -> dict[int, float]:
    """{hour: viewers} for hours 1..24 of a comparable's release day."""
    try:
        day_start = parse_ts(comp["day_start"])
    except (KeyError, ValueError):
        return {}
    out: dict[int, float] = {}
    for pt in comp.get("hourly") or []:
        v = pt.get("viewers")
        if v is None or v <= 0:
            continue
        try:
            t = parse_ts(pt["from"])
        except (KeyError, ValueError):
            continue
        h = int((t - day_start).total_seconds() // 3600) + 1
        if 1 <= h <= 24:
            out[h] = float(v)
    return out


def hour_quantiles(p1: float, curves: list[dict[int, float]], h: int) -> list[float | int]:
    """Five quantiles for hour h: empirical quantiles of the neighbours'
    log-ratios to their own hour-1 print, scaled by this case's p1, then
    shifted by the local size trend (see below)."""
    lp1 = math.log(p1)
    ranked = sorted(
        (abs(math.log(cur[1]) - lp1), cur)
        for cur in curves
        if 1 in cur
    )
    near = [cur for _, cur in ranked[:K_NEIGHBORS]]
    pairs = sorted(
        (math.log(cur[h] / cur[1]), math.log(cur[1]))
        for cur in near
        if h in cur
    )
    if len(pairs) < MIN_NEIGHBORS:
        # Fallback: the house baseline's widening spread around hour 1.
        qs = [p1 * math.exp(z * SIGMA * math.sqrt(h - 1)) for z in Z]
    else:
        ratios = [r for r, _ in pairs]
        qs = [p1 * math.exp(percentile(ratios, tau)) for tau in LEVELS]
        # Size-trend shift.  Launch size predicts the shape: across the
        # kit's comparables the correlation between ln(hour-1) and
        # ln(peak / hour-1) is -0.94, so a launch bigger than its
        # neighbours grows less after hour 1, and a smaller one grows
        # more.  The neighbours' own log-ratios therefore sit at the
        # wrong level when this case's hour-1 print is far from the
        # neighbours' typical size — most of all for a launch bigger
        # than every comparable, which is exactly GTA VI's regime.
        # Correct it with the local OLS slope of log-ratio on log
        # hour-1 among the neighbours, evaluated at this case's size
        # minus the neighbours' median size, half-shrunk (TREND_SHRINK)
        # because a 10-point slope is noisy.  The shift is ~0 when the
        # case sits inside the neighbours' size range, so it only acts
        # where the size mismatch is real.
        xs = [x for _, x in pairs]
        ys = [r for r, _ in pairs]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx > 0:
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
            med_x = percentile(sorted(xs), 0.5)
            delta = TREND_SHRINK * slope * (lp1 - med_x)
            qs = [q * math.exp(delta) for q in qs]
    # Clamp to the contest's bounds and keep the row non-decreasing.
    out: list[float | int] = []
    prev = 0.0
    for q in qs:
        q = min(max(q, 0.01), 1e9)
        q = max(q, prev)
        out.append(round2(q))
        prev = float(out[-1])
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    curves = [release_day_hours(c) for c in case.get("comparables") or []]
    return {
        "hours": [
            {"hour": row["hour"], "q": hour_quantiles(p1, curves, row["hour"])}
            for row in case["forecast"]
        ]
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
