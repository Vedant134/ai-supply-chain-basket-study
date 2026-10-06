# AI and infrastructure basket spread: research brief

## Goal

Build a reproducible research and paper-trading prototype that tests whether a basket of AI semiconductor stocks and a basket of AI infrastructure stocks have a tradeable mean-reverting relationship. A positive backtest is evidence to investigate, not proof of future returns or readiness to manage real money.

## Initial universe and data window

- Tech basket: NVDA, AMD, AVGO.
- Infrastructure basket: VRT, ETN, PWR. PWR is an initial substitute for GE because today's GE ticker represents GE Aerospace after the 2024 GE Vernova separation. Record this choice in every report; assess alternatives separately.
- Use the common available history of all six stocks, starting no earlier than VRT's public listing on 2020-02-10. Do not describe this as a ten-year six-stock backtest.
- Save source, retrieval date, raw data snapshot, corporate-action adjustments, and missing-data decisions. The data provider is replaceable and must stay separate from strategy logic.

## Research sequence

1. Build adjusted, normalized basket indexes from prices and documented weights. Decide and record rebalancing and financing assumptions.
2. Define a spread using a hedge ratio estimated only from past training data. Test whether it actually tends to revert; do not infer that from the companies' business relationship.
3. Establish a simple rules-based spread strategy as a baseline before fitting a Random Forest. Define the prediction horizon and labels strictly from future data, with features known at signal time.
4. Evaluate with chronological walk-forward splits, including a gap appropriate to the label horizon. Refit all preprocessing, spread parameters, and model on each training window only.
5. Simulate next-session execution with short-sale availability, borrow/financing costs, transaction fees, slippage, and position/risk limits. Never treat same-day closing prices as executable after a close-based signal.
6. Compare the model with the baseline and a no-trade benchmark on out-of-sample data. Report return, drawdown, turnover, trade count, exposure, and sensitivity to costs and parameter choices.
7. Produce a dated report with charts, assumptions, failure cases, and a clear go/no-go conclusion. Paper trade only after the historical test is reproducible.

## Deliverables

- Versioned data pipeline and frozen sample dataset or fetch instructions.
- Basket/spread research notebook or script with diagnostic results.
- Baseline and Random Forest backtests sharing one execution simulator.
- Automated checks for data alignment, look-ahead leakage, position accounting, and transaction-cost application.
- One-command reproducible run and concise results report.

## Scope

The market-data pipeline, statistical tests, strategy, and simulator live in this Python project. The first complete research run is documented in `results/REPORT.md`; it does not establish a reliable trading edge. Yahoo Finance supplies market data through `yfinance`; it is independent of any AI-agent harness.
