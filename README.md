# Systematic Trading Strategy Backtest 

Problem: Retail traders often lack flexible and free tools to test complex trading rules over long historical periods. There are very limited free options available online and those that are available, have very limited strategy modifications. I strongly feel that most retail traders are unable to effectively test their strategy. I built a Python-based backtesting engine that allows users to define and modify market stress signals, then evaluate how a hedging overlay would affect portfolio performance.

The system uses indicators such as VIX spikes, cross-asset correlation spikes, sector correlation spikes, and widening credit spreads to turn a risk overlay on or off. It then compares hedged and unhedged performance using metrics such as annual return, volatility, Sharpe ratio, Sortino ratio, and calendar-year returns.
Users are able to modify these indicators too, they are not limited to the ones mentioned above. 

in-progress: I am currently building a front-end to let users edit trade rules and test strategies more easily.

Data: yfinance 
Model: Short (sell) when risk overlay is ON
How to use: For changing/adding more rules, pls edit line 1-59, the rest is graph plotting and statistics. Model will hedge  Short QQQ during risk-on regimes, identified by risk1,risk2,etc.

Example of Performance metrics:

=== Overall Performance ===
Unhedged:  annual returns: 19.09% | annual volatility 22.25% | sharpe 0.86 | sortino 1.09
Hedged:    annual returns: 14.47% | annual volatility 16.49% | sharpe 0.88 | sortino 1.14

RISK ON days (%): 25.928822839264765
Overlay ON today (2026-03-08): NO
Latest signal date (2026-03-06): OFF

Risk3 (cross-asset correlation) spike days (%): 22.60
Risk4 (sector correlation) spike days (%): 11.54
Risk5 (V VIX) spike days (%): 9.78
Risk6 (credit spread) spike days (%): 65.23


