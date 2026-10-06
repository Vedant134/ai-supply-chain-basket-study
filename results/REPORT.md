# AI versus infrastructure baskets: research results

Data snapshot retrieved 2026-10-06T05:22:16.583726+00:00 from Yahoo Finance via yfinance.
Six stocks: NVDA, AMD, AVGO, VRT, ETN, PWR. Complete sessions: 1672 (2020-02-10 to 2026-10-05); 0 incomplete provider sessions excluded.

## What we tested

Each basket resets to equal dollar weights daily. The log spread is log(tech basket) minus 0.954 times log(infrastructure basket); that hedge ratio was fitted only on the first 504 sessions (2020-02-10 to 2022-02-07). The later period, 2022-02-10 to 2026-10-05, is held out for walk-forward evaluation. This is a fixed present-day stock selection and remains subject to selection bias.

The training-only cointegration test p-value was 0.7165; the training-only spread ADF p-value was 0.4809. A low p-value is evidence against a unit root/no cointegration under each test's assumptions. It does not guarantee the relationship will persist. The ML label asks whether the spread falls over five sessions starting at the next close. The classifier predicts that direction; it filters entries of a simple z-score rule. It refits every 63 sessions, with a 6-session purge before each test block.

Out-of-sample classifier accuracy was 50.78%, against a majority-class baseline of 50.86%; ROC AUC was 0.4949449895926257. These prediction scores are separate from trading performance.

## Simulated performance

| Strategy | Total return | CAGR | Max drawdown | Sharpe (zero cash rate) | Entries/reversals |
|---|---:|---:|---:|---:|---:|
| Simple spread rule | 2.75% | 0.59% | -24.41% | 0.11 | 17 |
| Random Forest filter | 23.81% | 4.72% | -23.65% | 0.41 | 14 |
| 50/50 long-only | 681.29% | 55.94% | -39.01% | 1.35 | 1 |

The pair strategies use a one-session delay from close-based signal to next-close fill. Returns begin after that fill. They pay 2 bps fees plus 5 bps slippage per traded dollar, and 100 bps annualized on the short leg. The long-only reference has no costs in this comparison. No order book, borrow locate, intraday execution, tax, or market-impact model is included.

Calendar-year returns show where performance came from (2022 and 2026 are partial years):

| Year | Simple spread rule | Random Forest filter |
|---|---:|---:|
| 2022 | -6.22% | -9.75% |
| 2023 | 3.40% | 4.98% |
| 2024 | 13.29% | 25.34% |
| 2025 | -4.30% | 0.73% |
| 2026 | -2.27% | 3.50% |

With the same positions and borrow charge, the Random Forest result changes as fee plus slippage per traded dollar changes:

| Fee + slippage | ML total return |
|---|---:|
| 0 bps | 26.17% |
| 7 bps | 23.81% |
| 20 bps | 19.53% |

## What this means

This sample does not support the proposed stable mean-reverting relationship: the training tests do not reject their null hypotheses. The classifier also fails to beat a majority-class prediction and has ROC AUC below 0.50. Its higher simulated return could be sampling luck or a particular pattern of skipped trades. Treat this as a research result, not a deployable trading signal.

The long-only reference is not risk-matched to the pair trades. The chosen six names were selected using today's knowledge, the hedge ratio and thresholds were fixed from one sample, and the Yahoo-derived adjusted closes are research data rather than an executable bid/ask record. Before any paper-trading decision, repeat the analysis with independent data, alternative baskets, and predeclared parameters. The frozen data, daily decisions, and code are saved beside this report so the calculation can be reproduced.
