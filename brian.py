
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


NAV0  = 10000
START = "2010-01-01"  #this is ur start date of the backtest

# Portfolio weights (example: 100% QQQ. jst add the same thing below for more assets, eg "AAPL": 0.2)
weights = {
    "QQQ": 1.00,
}

# Hedge settings
HEDGE_H = 0.60                 # short 60% of QQQ exposure when overlay is ON
BORROW_RATE_ANNUAL = 0.02      # 2% annual borrow/financing cost on short notional, cos moomoo got interest rate on shorting
SLIPPAGE_BPS = 5               # 5 bps cost per regime switch (enter/exit) -> basically assuming u wont get the best bid/ask

# Regime settings
MA_WINDOW = 50                 # moving average window on QQQ
VIX_THRESHOLD = 20             # overlay requires VIX > threshold

# Rolling stats window
WINDOW = 63                    # ~3 months


# this is our mini database from yahoo finance
# Ensure QQQ and ^VIX included
tickers = sorted(set(list(weights.keys()) + ["QQQ", "^VIX"]))
px = yf.download(tickers, start=START, auto_adjust=True, progress=False)["Close"].dropna()
rets = px.pct_change().dropna()

# Align everything
idx = rets.index


# Portfolio return series (if weights don't sum to 1, remainder is assumed cash at 0% return)
w_sum = sum(weights.values())
if w_sum > 1.000001:
    raise ValueError(f"Weights sum to {w_sum:.4f} > 1.0. Fix your weights.")

port = pd.Series(0.0, index=idx)
for t, w in weights.items():
    port = port.add(rets[t] * w, fill_value=0.0)


# REGIME (overlay ON/OFF)
qqq = px["QQQ"].loc[idx]
vix = px["^VIX"].loc[idx]

#this is gonna be our strategy logic, u can add more conditions if u want, eg risk3=....| and forget add it int eh oeverlay line also
qqq_ma = qqq.rolling(100).mean() 
risk1 = qqq < qqq_ma
risk2 = vix > 20
overlay_on = (risk1 & risk2).astype(int)  # 1 = hedge on, 0 = hedge off
overlay_series =  overlay_on.reindex(idx).shift(1).fillna(0).astype(int) # this is our time series where we are gonna risk on mode


# SHORT QQQ OVERLAY HEDGE
# ----------------------------
qqq_ret = rets["QQQ"].loc[idx]

# Short overlay return: when QQQ falls, short gains (+), when QQQ rises, short loses (-)
short_overlay_return = -HEDGE_H * qqq_ret * overlay_series # returns from the short

# Borrow/financing cost applies only when short is active (proportional to hedge notional)
daily_borrow_cost = (BORROW_RATE_ANNUAL / 252.0) #252 trading days in a year 
borrow_drag = daily_borrow_cost * (HEDGE_H * overlay_series)

# Slippage cost on regime switches (enter/exit)
switch = overlay_series.diff().abs().fillna(0)  # 1 on OFF->ON or ON->OFF
slip_cost = (SLIPPAGE_BPS / 10000.0) * switch

# Hedged return series
hedged = port + short_overlay_return - borrow_drag - slip_cost #this is our final hedged return series

# PERFORMANCE STATS
# ----------------------------
def perf_stats(x: pd.Series):
    x = x.dropna()
    ann_ret = (1 + x).prod() ** (252/len(x)) - 1
    ann_vol = x.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    return ann_ret, ann_vol, sharpe

u = perf_stats(port)
h = perf_stats(hedged)

print("Unhedged:  ann_ret %.2f%% | ann_vol %.2f%% | sharpe %.2f" % (u[0]*100, u[1]*100, u[2]))
print("Hedged:    ann_ret %.2f%% | ann_vol %.2f%% | sharpe %.2f" % (h[0]*100, h[1]*100, h[2]))
print("\nOverlay ON days (%):", overlay_series.mean()*100)


# When ON, you are short HEDGE_H of QQQ notional. If your portfolio beta ≈ 1, that’s the hedge fraction.
print("\nEffective hedge when ON (fraction of QQQ):")
print("  Hedge fraction (HEDGE_H):", HEDGE_H)



#these are matplotlib settings for the plots, dont need to change anything here


# EQUITY CURVES + DRAWDOWN
# ----------------------------
equity_unhedged = (1 + port).cumprod()
equity_hedged   = (1 + hedged).cumprod()

def drawdown(equity_curve: pd.Series) -> pd.Series:
    peak = equity_curve.cummax()
    return equity_curve / peak - 1

dd_unhedged = drawdown(equity_unhedged)
dd_hedged   = drawdown(equity_hedged)

# Rolling Sharpe
roll_ret_u = (1 + port).rolling(WINDOW).apply(lambda x: x.prod()**(252/WINDOW)-1, raw=False)
roll_ret_h = (1 + hedged).rolling(WINDOW).apply(lambda x: x.prod()**(252/WINDOW)-1, raw=False)
roll_vol_u = port.rolling(WINDOW).std() * np.sqrt(252)
roll_vol_h = hedged.rolling(WINDOW).std() * np.sqrt(252)
roll_sharpe_u = roll_ret_u / roll_vol_u
roll_sharpe_h = roll_ret_h / roll_vol_h

# Regime shading helper
def shade_overlay(ax):
    on = overlay_series == 1
    if on.any():
        idx2 = overlay_series.index
        starts = idx2[(~on.shift(1, fill_value=False)) & on]
        ends   = idx2[(on) & (~on.shift(-1, fill_value=False))]
        for s, e in zip(starts, ends):
            ax.axvspan(s, e, alpha=0.15)

# ----------------------------
# PLOTS
# ----------------------------
# Plot 1: Equity curves
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(equity_unhedged.index, equity_unhedged.values, label="Unhedged")
ax.plot(equity_hedged.index, equity_hedged.values, label="Hedged (Short QQQ ON)")
shade_overlay(ax)
ax.set_title("Equity Curve")
ax.set_xlabel("Date")
ax.set_ylabel("Asset value")
ax.legend()
plt.tight_layout()
plt.show()





# Plot 2: Drawdowns
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dd_unhedged.index, dd_unhedged.values, label="Unhedged DD")
ax.plot(dd_hedged.index, dd_hedged.values, label="Hedged DD")
shade_overlay(ax)
ax.set_title("Drawdowns (Shaded = hedge ON)")
ax.set_xlabel("Date")
ax.set_ylabel("Drawdown")
ax.legend()
plt.tight_layout()
plt.show()

# Plot 3: Rolling Sharpe
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(roll_sharpe_u.index, roll_sharpe_u.values, label=f"Unhedged {WINDOW}d Sharpe")
ax.plot(roll_sharpe_h.index, roll_sharpe_h.values, label=f"Hedged {WINDOW}d Sharpe")
shade_overlay(ax)
ax.set_title(f"Rolling Sharpe ({WINDOW}d) (Shaded = Overlay ON)")
ax.set_xlabel("Date")
ax.set_ylabel("Sharpe")
ax.legend()
plt.tight_layout()
plt.show()



# ----------------------------
# REGIME PERFORMANCE TABLE (ON vs OFF)
# ----------------------------
def stats(x):
    x = x.dropna()
    ann_ret = (1 + x).prod() ** (252/len(x)) - 1
    ann_vol = x.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    return ann_ret, ann_vol, sharpe

mask_on  = overlay_series == 1
mask_off = overlay_series == 0

u_on  = stats(port[mask_on])
h_on  = stats(hedged[mask_on])
u_off = stats(port[mask_off])
h_off = stats(hedged[mask_off])

table = pd.DataFrame({
    "Unhedged (ON)":  [u_on[0], u_on[1], u_on[2]],
    "Hedged (ON)":    [h_on[0], h_on[1], h_on[2]],
    "Unhedged (OFF)": [u_off[0], u_off[1], u_off[2]],
    "Hedged (OFF)":   [h_off[0], h_off[1], h_off[2]],
}, index=["Ann Return", "Ann Vol", "Sharpe"])

print("\n=== Regime split (Overlay ON vs OFF) ===")
table_fmt = table.copy()
table_fmt.loc["Ann Return"] *= 100
table_fmt.loc["Ann Vol"] *= 100
print(table_fmt.round(2))
print("\n(Ann Return/Vol shown in %, Sharpe unitless)")

# ----------------------------
# CALENDAR-YEAR RETURNS
# ----------------------------
unhedged_eq = (1 + port).cumprod()
hedged_eq   = (1 + hedged).cumprod()

un_y = unhedged_eq.resample("YE").last()
hd_y = hedged_eq.resample("YE").last()

un_cal = un_y.pct_change().dropna()
hd_cal = hd_y.pct_change().dropna()

annual_table = pd.DataFrame({
    "Unhedged": (un_cal * 100).round(2),
    "Hedged":   (hd_cal * 100).round(2),
})
annual_table.index = annual_table.index.year

print("\n=== Calendar-year returns (%) ===")
print(annual_table.to_string())

print("\nBest/Worst calendar year:")
print("Unhedged best:", annual_table["Unhedged"].max(), "worst:", annual_table["Unhedged"].min())
print("Hedged   best:", annual_table["Hedged"].max(), "worst:", annual_table["Hedged"].min())
