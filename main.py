"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

A program entry for make.fun's gta6-twitch-ccu-forecast-01: it reads one case
from /input/<id>.in and writes /output/<id>.out with, for each of hours 2..24
of the release day (UTC), five quantiles (5%, 25%, 50%, 75%, 95%) of that
hour's Twitch concurrent viewers in the "Grand Theft Auto VI" category.

The method, fitted at run time from the case's own `comparables` (past big
game launches' hourly Twitch curves, 14 days before to 3 days after their
release day):

1. Every comparable launch is reduced to a launch-shape curve: the log-ratio
   r_h = ln(viewers_h / viewers_1) for hours h = 2..24, where viewers_1 is
   that launch's hour-1 print. A launch's shape is mostly decided by when its
   big markets unlock relative to UTC midnight (rolling midnight: a small
   hour 1, a dip in hours 2-4, a jump at the US unlock around hour 5-6, then
   a climb to a US-evening peak), and the hour-1 print is the visible trace
   of that: a big hour 1 means the launch was already unlocked in the big
   markets, a small one means the unlock is still coming. So the shape is
   predicted from the hour-1 level.
2. Median: a local-linear (kernel-weighted least squares) fit of r_h on
   ln(hour-1 print), Gaussian kernel with bandwidth 2.0 in log units, fitted
   separately for each hour. Local, not global: the log-ratio/log-level
   relation bends sharply at the top end, and a single global line badly
   mispredicts the biggest launches (checked on past launches: training only
   on launches with hour-1 <= 500k and predicting the three above it, the
   global line scores ~0.40 pinball loss while this local fit scores ~0.17).
3. Spread: honest leave-one-out residuals of the same local fit. For each
   comparable launch, refit without it and record the residual at its own
   hour-1 level; the spread for the target hour is the standard deviation of
   the residuals of the 12 comparables nearest in log hour-1 level to the
   target, scaled by 0.9 (leave-one-out residuals slightly overstate the
   error of a fit that uses all launches). Quantiles are
   q_tau = p1 * exp(median_h + z_tau * sigma_h) with z the standard normal
   quantiles, rounded half-up to 2 decimals.

Everything is fitted from the case file at run time, so the same program
works on the practice cases and on the final GTA VI case (whose comparables
also include the four practice launches). Deterministic: fixed iteration
order, no randomness, no network, Python 3.13 standard library only.

Local measurement (pinball loss on the log scale, lower is better; the
practice answers are the four past launches' release-day counts, verified
against each case's known hour-1 print):

  practice cases p00-p03: 0.178 (house baseline 0.387 on the same cases)
  leave-one-out over the 16 comparables with hour-1 >= 100k: 0.211
  top-edge check (train on hour-1 <= 500k, test the three above): 0.173

How the grader runs it: `python3 -P -s /bot/main.py` in python:3.13-slim,
standard library only, no network. Outside the grader's container,
MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output:

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file (none is shipped; this stays).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Gaussian kernel bandwidth, in natural-log units of the hour-1 print.
BANDWIDTH = 2.0
# How many of the nearest (in log hour-1 level) comparables' leave-one-out
# residuals set an hour's spread.
NEAREST_K = 12
# Leave-one-out residuals slightly overstate the error of the full fit.
SIGMA_SCALE = 0.9
# The spread never goes below this, in log units.
SIGMA_FLOOR = 0.05
# The house baseline's spread, used only when a case carries no usable
# comparable launch at all.
FALLBACK_SIGMA = 0.15


def parse_time(text):
    """An ISO 8601 UTC timestamp into a datetime (accepts a trailing Z)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def hour_of(day_start, stamp):
    """Which hour of the release day (1..24) a timestamp falls in, else None."""
    delta = (parse_time(stamp) - day_start).total_seconds()
    hour = int(delta // 3600) + 1
    return hour if 1 <= hour <= 24 else None


def training_rows(case):
    """One row per comparable launch with a complete release day.

    Each row is (log hour-1 print, {hour: log-ratio to hour 1}). A launch is
    usable only when every hour 1..24 has a positive viewer count; the GTA VI
    category's own pre-release history has no release day yet, so it drops
    out here on its own.
    """
    rows = []
    for comp in case.get("comparables", []):
        try:
            day_start = parse_time(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        hours = {}
        for point in comp.get("hourly", []):
            viewers = point.get("viewers")
            if viewers is None or viewers <= 0:
                continue
            try:
                hour = hour_of(day_start, point["from"])
            except (KeyError, TypeError, ValueError):
                continue
            if hour is not None and hour not in hours:
                hours[hour] = viewers
        if len(hours) < 24 or 1 not in hours:
            continue
        p1 = hours[1]
        rows.append((math.log(p1), {h: math.log(hours[h] / p1) for h in range(2, 25)}))
    return rows


def local_linear(rows, target_logp1, hour):
    """Kernel-weighted least squares of log-ratio on log hour-1 print.

    Returns (intercept, slope) of the local line at the target's level, fitted
    on `rows` (each (logp1, {hour: log-ratio})). A flat fit when the local
    spread of levels is too small to support a slope.
    """
    weights = [math.exp(-((lp - target_logp1) / BANDWIDTH) ** 2) for lp, _ in rows]
    total = sum(weights)
    if total <= 0:
        return 0.0, 0.0
    mean_x = sum(w * lp for w, (lp, _) in zip(weights, rows)) / total
    mean_y = sum(w * r[hour] for w, (_, r) in zip(weights, rows)) / total
    sxx = sum(w * (lp - mean_x) ** 2 for w, (lp, _) in zip(weights, rows))
    sxy = sum(w * (lp - mean_x) * (r[hour] - mean_y)
              for w, (lp, r) in zip(weights, rows))
    slope = sxy / sxx if sxx > 1e-9 else 0.0
    return mean_y - slope * mean_x, slope


def pstdev(values):
    """Population standard deviation."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    logp1 = math.log(p1)

    rows = training_rows(case)
    hours = []
    if not rows:
        # No usable comparable: fall back to the house baseline's persistence
        # with its widening spread.
        for row in case["forecast"]:
            h = row["hour"]
            hours.append({"hour": h, "q": [round2(p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(h - 1))) for z in Z]})
        return {"hours": hours}

    # Median log-ratio per hour from the local fit at the target's level.
    median = {}
    for h in range(2, 25):
        intercept, slope = local_linear(rows, logp1, h)
        median[h] = intercept + slope * logp1

    # Honest spread: leave-one-out residuals of the same local fit, one per
    # comparable launch, then the spread of the nearest launches' residuals.
    loo_residuals = {h: [] for h in range(2, 25)}
    for i, (lp_i, r_i) in enumerate(rows):
        rest = rows[:i] + rows[i + 1:]
        for h in range(2, 25):
            intercept, slope = local_linear(rest, lp_i, h)
            loo_residuals[h].append(r_i[h] - (intercept + slope * lp_i))
    nearest = sorted(range(len(rows)), key=lambda i: abs(rows[i][0] - logp1))[:NEAREST_K]
    sigma = {}
    for h in range(2, 25):
        nearby = [loo_residuals[h][i] for i in nearest]
        sigma[h] = max(SIGMA_FLOOR, pstdev(nearby) * SIGMA_SCALE)

    for row in case["forecast"]:
        h = row["hour"]
        if h not in median or h not in sigma:
            # A forecast hour outside 2..24 (the rules promise none): persist.
            quantiles = [round2(p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(max(h - 1, 0)))) for z in Z]
            hours.append({"hour": h, "q": quantiles})
            continue
        quantiles = []
        for z in Z:
            q = p1 * math.exp(median[h] + z * sigma[h])
            q = min(q, 1e9)
            quantiles.append(round2(max(q, 0.01)))
        # Rounding can only equalise neighbours, never reverse them, but
        # enforce non-decreasing explicitly so the output is always valid.
        for j in range(1, 5):
            if quantiles[j] < quantiles[j - 1]:
                quantiles[j] = quantiles[j - 1]
        hours.append({"hour": h, "q": quantiles})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
