# gta6-twitch-ccu-forecast-01 — local-linear launch-shape forecaster

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/0xazazul/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/0xazazul/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts Grand Theft Auto VI's launch-day Twitch viewers hour by hour, from
the first hour's count: for each of hours 2–24 of 2026-11-19 (UTC), five
quantiles (5%, 25%, 50%, 75%, 95%) of that hour's concurrent viewers in the
"Grand Theft Auto VI" Twitch category. Built for the make.fun
gta6-twitch-ccu-forecast-01 bounty ($250 in USDC).

## Approach

`main.py` (Python 3.13, standard library only) fits its model at run time
from the case's own `comparables` — past big game launches' hourly Twitch
curves — so the same program works on the practice cases and on the final
GTA VI case:

1. **Launch-shape curves.** Each comparable launch is reduced to log-ratios
   `r_h = ln(viewers_h / viewers_1)` for hours 2–24. A launch's shape is
   mostly decided by when its big markets unlock relative to UTC midnight
   (rolling midnight: a small hour 1, a dip in hours 2–4, a jump at the US
   unlock around hour 5–6, then a climb to a US-evening peak) — and the
   hour-1 print is the visible trace of that: a big hour 1 means the launch
   was already unlocked in the big markets, a small one means the unlock is
   still coming. So the shape is predicted from the hour-1 level.
2. **Median: local-linear fit.** A Gaussian-kernel-weighted least-squares fit
   of `r_h` on `ln(hour-1 print)`, bandwidth 2.0 in log units, fitted
   separately for each hour. Local, not global: the log-ratio/log-level
   relation bends sharply at the top end, and a single global line badly
   mispredicts the biggest launches.
3. **Spread: honest leave-one-out residuals.** For each comparable launch,
   refit without it and record the residual at its own hour-1 level; an
   hour's spread is the standard deviation of the residuals of the 12
   comparables nearest in log hour-1 level to the target, scaled by 0.9.
   Quantiles are `q_tau = p1 * exp(median_h + z_tau * sigma_h)`.

Deterministic (fixed iteration order, no randomness), no network, runs in
~0.3 s per case (limit 8 s).

## Local measurement (pinball loss on the log scale, lower is better)

Practice answers are the four past launches' release-day counts, read from
SullyGnome's public charts and verified against each case's known hour-1
print; the scorer reproduces the house baseline's stated practice loss of
~0.387 (measured 0.3871).

| Evaluation | n | This entry | House baseline |
|---|---|---|---|
| Practice cases p00–p03 | 4 | **0.178** | 0.387 |
| Leave-one-out, comparables with hour-1 ≥ 100k | 16 | **0.211** | 0.256 |
| Top-edge check (train hour-1 ≤ 500k, test the 3 above) | 3 | **0.173** | 0.402 |
| Leave-one-out, comparables with hour-1 ≥ 20k | 27 | **0.265** | 0.314 |

Coverage of the 90% band on the practice cases is 0.924 (nominal 0.9) and
the typical miss is 32.9%. The practice cases are only four launches, so
these numbers are a guide, not a promise: the final is one unseen launch,
and a top-edge launch like GTA VI is exactly where the local fit was
chosen over the global line. At the time of building, the visible field's
best practice score was 0.169 and the tenth-best 0.190, so this entry is
top-10-competitive with margin but does not lead the field.

## How to run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory, writes `<id>.out` as
`{"hours":[{"hour":2,"q":[q05,q25,q50,q75,q95]}, ...]}` (23 rows, hours
2–24). Two runs on the same case produce byte-identical output.

## Files

- `main.py` — the entry
- `LICENSE` — MIT, copied from the kit starter
- `.makefun/owner.json` — claim file naming the signing wallet

Built for the [make.fun bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) — an entry by @azazul.
