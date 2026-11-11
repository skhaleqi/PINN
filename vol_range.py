"""
Measures the range of annualized REALIZED volatility in your price data, to choose the
sigma range of the parametric PINN.

    python vol_range.py yourfile.csv [window_days]

The file needs a date (or date+time) column and a close-price column. Works with common layouts
(Date/Time/Close, MetaTrader '<DATE>' '<TIME>' '<CLOSE>', Yahoo Finance, etc.). Intraday data is first
reduced to the daily last price. Volatility = standard deviation of daily log returns over a rolling
window (default 21 trading days, about one month) times sqrt(252).
Prints min / percentiles / max of that rolling volatility and the overall full-sample volatility.
"""
import sys
import numpy as np
import pandas as pd

path = sys.argv[1]
win = int(sys.argv[2]) if len(sys.argv) > 2 else 21
raw = pd.read_csv(path, sep=None, engine="python")
cols = {c: str(c).lower().strip().strip("<>").strip() for c in raw.columns}
date_c = [c for c, l in cols.items() if l in ("date", "datetime", "timestamp", "time stamp", "gmt time", "local time")]
time_c = [c for c, l in cols.items() if l == "time"]
close_c = [c for c, l in cols.items() if l in ("close", "adj close", "price", "last")]
if not date_c or not close_c:
    sys.exit(f"Could not find date/close columns. Columns are: {list(raw.columns)}")
dt = raw[date_c[0]].astype(str)
if time_c and time_c[0] != date_c[0]:
    dt = dt + " " + raw[time_c[0]].astype(str)
raw["_dt"] = pd.to_datetime(dt, errors="coerce", dayfirst=False)
raw = raw.dropna(subset=["_dt"]).set_index("_dt").sort_index()
close = pd.to_numeric(raw[close_c[0]], errors="coerce").dropna()
daily = close.resample("1D").last().dropna()
ret = np.log(daily).diff().dropna()
roll = ret.rolling(win).std() * np.sqrt(252)
roll = roll.dropna()
print(f"{path}: {len(daily)} daily closes from {daily.index[0].date()} to {daily.index[-1].date()}")
print(f"full-sample annualized volatility: {ret.std() * np.sqrt(252):.3f}")
print(f"rolling {win}-day annualized volatility ({len(roll)} windows):")
for name, q in [("min", 0), ("1%", .01), ("5%", .05), ("25%", .25), ("median", .5), ("75%", .75), ("95%", .95), ("99%", .99), ("max", 1)]:
    print(f"  {name:>6}: {roll.quantile(q):.3f}")
