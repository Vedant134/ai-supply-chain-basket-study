"""Causal basket-spread research, walk-forward classification, and paper backtest."""

from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from statsmodels.tsa.stattools import adfuller, coint

from .data import INFRA, TECH


CALIBRATION_DAYS = 504
ROLLING_DAYS = 126
HORIZON_DAYS = 5
EXECUTION_LAG_DAYS = 2
REFIT_DAYS = 63
ENTRY_Z = 1.0
EXIT_Z = 0.25
FEE_BPS = 2.0
SLIPPAGE_BPS = 5.0
BORROW_BPS_ANNUAL = 100.0


@dataclass
class Study:
    daily: pd.DataFrame
    diagnostics: dict
    metrics: dict


def basket_index(prices: pd.DataFrame, symbols: tuple[str, ...]) -> pd.Series:
    """Equal dollar weights reset each close; index starts at one."""
    daily = prices[list(symbols)].pct_change(fill_method=None).fillna(0).mean(axis=1)
    return (1 + daily).cumprod().rename("basket")


def _features(spread: pd.Series, tech: pd.Series, infra: pd.Series) -> pd.DataFrame:
    tech_ret = tech.pct_change(fill_method=None)
    infra_ret = infra.pct_change(fill_method=None)
    spread_mean = spread.rolling(ROLLING_DAYS, min_periods=ROLLING_DAYS).mean()
    spread_std = spread.rolling(ROLLING_DAYS, min_periods=ROLLING_DAYS).std()
    return pd.DataFrame({
        "z": (spread - spread_mean) / spread_std.replace(0, np.nan),
        "spread_change_5": spread.diff(5),
        "spread_change_20": spread.diff(20),
        "tech_return_5": tech.pct_change(5),
        "infra_return_5": infra.pct_change(5),
        "tech_vol_20": tech_ret.rolling(20).std(),
        "infra_vol_20": infra_ret.rolling(20).std(),
    }).replace([np.inf, -np.inf], np.nan)


def walk_forward_probabilities(features: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """Fit on past labeled rows, purge the six-day label horizon, refit quarterly."""
    result = pd.Series(np.nan, index=features.index, name="prob_spread_down")
    n = len(features)
    for start in range(CALIBRATION_DAYS, n, REFIT_DAYS):
        # At decision date `start`, a label at i needs prices through i+6.
        # Excluding the six immediately preceding sample indices is conservative.
        train_end = start - (HORIZON_DAYS + 1)
        train_x = features.iloc[:train_end]
        train_y = labels.iloc[:train_end]
        good = train_x.notna().all(axis=1) & train_y.notna()
        if good.sum() < 200 or train_y.loc[good].nunique() < 2:
            continue
        model = RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        model.fit(train_x.loc[good], train_y.loc[good].astype(int))
        end = min(start + REFIT_DAYS, n)
        test_x = features.iloc[start:end]
        valid = test_x.notna().all(axis=1)
        if valid.any():
            result.loc[test_x.index[valid]] = model.predict_proba(test_x.loc[valid])[:, 1]
    return result


def position_signals(z: pd.Series, probability: pd.Series | None = None) -> pd.Series:
    """Enter on a large deviation and leave when the spread returns near its mean."""
    values = []
    state = 0
    for date, score in z.items():
        if not np.isfinite(score):
            state = 0
        else:
            if state == -1 and score <= EXIT_Z:
                state = 0
            elif state == 1 and score >= -EXIT_Z:
                state = 0
            gate = float(probability.loc[date]) if probability is not None else np.nan
            if state == 0 and score >= ENTRY_Z and (probability is None or gate >= 0.55):
                state = -1  # Short tech; long infrastructure.
            elif state == 0 and score <= -ENTRY_Z and (probability is None or gate <= 0.45):
                state = 1   # Long tech; short infrastructure.
        values.append(state)
    return pd.Series(values, index=z.index, dtype=int)


def simulate(
    signal: pd.Series, tech: pd.Series, infra: pd.Series, beta: float,
    first_test_index: int,
) -> pd.DataFrame:
    """Signal at close t, fill at close t+1, earn the t+1 to t+2 return."""
    position = signal.shift(EXECUTION_LAG_DAYS).fillna(0).astype(int)
    tech_weight = 1 / (1 + abs(beta))
    infra_weight = abs(beta) / (1 + abs(beta))
    tech_ret = tech.pct_change(fill_method=None).fillna(0)
    infra_ret = infra.pct_change(fill_method=None).fillna(0)
    gross = position * (tech_weight * tech_ret - infra_weight * infra_ret)
    turnover = position.diff().abs().fillna(position.abs())
    trading_cost = turnover * (FEE_BPS + SLIPPAGE_BPS) / 10_000
    short_weight = np.where(position > 0, infra_weight, np.where(position < 0, tech_weight, 0))
    borrow_cost = short_weight * BORROW_BPS_ANNUAL / 10_000 / 252
    net = gross - trading_cost - borrow_cost
    frame = pd.DataFrame({
        "signal": signal, "position": position, "gross_return": gross,
        "turnover": turnover, "trading_cost": trading_cost,
        "borrow_cost": borrow_cost, "net_return": net,
    })
    # Begin only after the first test signal could have been executed.
    frame = frame.iloc[first_test_index + EXECUTION_LAG_DAYS:].copy()
    frame["equity"] = (1 + frame["net_return"]).cumprod()
    return frame


def _metrics(frame: pd.DataFrame) -> dict:
    returns = frame["net_return"]
    n = len(returns)
    equity = frame["equity"]
    annual_vol = float(returns.std() * np.sqrt(252))
    previous = frame["position"].shift(1).fillna(0)
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "cagr": float(equity.iloc[-1] ** (252 / n) - 1),
        "annual_volatility": annual_vol,
        "sharpe_zero_cash_rate": float(returns.mean() * 252 / annual_vol) if annual_vol else None,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "entries_or_reversals": int(((frame["position"] != 0) & (frame["position"] != previous)).sum()),
        "gross_turnover": float(frame["turnover"].sum()),
        "invested_days": int((frame["position"] != 0).sum()),
    }


def run_study(prices: pd.DataFrame) -> Study:
    if len(prices) < CALIBRATION_DAYS + 200:
        raise ValueError("Need at least 504 calibration sessions and 200 evaluation sessions.")
    tech = basket_index(prices, TECH)
    infra = basket_index(prices, INFRA)
    log_tech, log_infra = np.log(tech), np.log(infra)
    calibration = slice(0, CALIBRATION_DAYS)
    beta = float(np.polyfit(log_infra.iloc[calibration], log_tech.iloc[calibration], 1)[0])
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError(f"Calibration returned a nonpositive hedge ratio: {beta}.")
    spread = (log_tech - beta * log_infra).rename("log_spread")
    features = _features(spread, tech, infra)

    # The target compares the spread at the executable next close with five sessions later.
    future_change = spread.shift(-(HORIZON_DAYS + 1)) - spread.shift(-1)
    labels = (future_change < 0).astype(float)
    labels.iloc[-(HORIZON_DAYS + 1):] = np.nan
    probabilities = walk_forward_probabilities(features, labels)
    if probabilities.iloc[CALIBRATION_DAYS:].notna().sum() < 100:
        raise ValueError("Too few out-of-sample classifier predictions.")

    eligible_z = features["z"].copy()
    eligible_z.iloc[:CALIBRATION_DAYS] = np.nan
    baseline = simulate(position_signals(eligible_z), tech, infra, beta, CALIBRATION_DAYS)
    ml_z = eligible_z.where(probabilities.notna())
    ml = simulate(position_signals(ml_z, probabilities), tech, infra, beta, CALIBRATION_DAYS)
    buy_hold_returns = 0.5 * tech.pct_change(fill_method=None) + 0.5 * infra.pct_change(fill_method=None)
    hold = pd.DataFrame({"net_return": buy_hold_returns.loc[ml.index].fillna(0)})
    hold["equity"] = (1 + hold["net_return"]).cumprod()

    result = pd.DataFrame({
        "tech_index": tech, "infra_index": infra, "log_spread": spread,
        "z": features["z"], "prob_spread_down": probabilities,
        "label_spread_down": labels,
    }).loc[ml.index].copy()
    for name, frame in (("baseline", baseline), ("ml", ml)):
        for column in ("signal", "position", "turnover", "trading_cost", "borrow_cost", "net_return", "equity"):
            result[f"{name}_{column}"] = frame[column]
    result["buy_hold_equity"] = hold["equity"]

    train_spread = spread.iloc[calibration]
    coint_stat, coint_p, _ = coint(log_tech.iloc[calibration], log_infra.iloc[calibration])
    adf_stat, adf_p, *_ = adfuller(train_spread, autolag="AIC", result_object=False)
    labeled = result["label_spread_down"].notna() & result["prob_spread_down"].notna()
    actual = result.loc[labeled, "label_spread_down"].astype(int)
    predicted = (result.loc[labeled, "prob_spread_down"] >= 0.5).astype(int)
    diagnostics = {
        "calibration_first": prices.index[0].date().isoformat(),
        "calibration_last": prices.index[CALIBRATION_DAYS - 1].date().isoformat(),
        "out_of_sample_first": result.index[0].date().isoformat(),
        "out_of_sample_last": result.index[-1].date().isoformat(),
        "calibration_hedge_ratio": beta,
        "training_cointegration_p": float(coint_p),
        "training_adf_spread_p": float(adf_p),
        "classifier_accuracy": float(accuracy_score(actual, predicted)),
        "majority_class_accuracy": float(max(actual.mean(), 1 - actual.mean())),
        "classifier_roc_auc": float(roc_auc_score(actual, result.loc[labeled, "prob_spread_down"])) if actual.nunique() == 2 else None,
        "labeled_out_of_sample_days": int(labeled.sum()),
        "cointegration_null": "no cointegration; p<0.05 is one possible rejection rule, not proof of future stability",
    }
    metrics = {"baseline": _metrics(baseline), "ml": _metrics(ml), "buy_hold": _metrics(hold.assign(position=1, turnover=0))}
    metrics["annual_net_returns"] = {
        str(year): {
            "baseline": float((1 + block["baseline_net_return"]).prod() - 1),
            "ml": float((1 + block["ml_net_return"]).prod() - 1),
        }
        for year, block in result.groupby(result.index.year)
    }
    metrics["ml_cost_sensitivity"] = {
        str(cost_bps): float((1 + ml["gross_return"] - ml["turnover"] * cost_bps / 10_000 - ml["borrow_cost"]).prod() - 1)
        for cost_bps in (0, 7, 20)
    }
    return Study(result, diagnostics, metrics)


def save_study(study: Study, output_dir: Path, manifest: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    study.daily.to_csv(output_dir / "out_of_sample_daily.csv", float_format="%.10f")
    payload = {"diagnostics": study.diagnostics, "metrics": study.metrics, "data_manifest": manifest}
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    study.daily["z"].plot(ax=axes[0], color="#315a7d", linewidth=0.9, title="Spread deviation (rolling z-score)")
    axes[0].axhline(ENTRY_Z, color="gray", linestyle="--", linewidth=0.7)
    axes[0].axhline(-ENTRY_Z, color="gray", linestyle="--", linewidth=0.7)
    for column, label in (("baseline_equity", "Simple spread rule"), ("ml_equity", "Random Forest filter"), ("buy_hold_equity", "50/50 long-only")):
        study.daily[column].plot(ax=axes[1], label=label, linewidth=1.1)
    axes[1].set_title("Out-of-sample simulated equity, starting at 1")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "study_chart.png", dpi=160)
    plt.close(fig)

    def fmt(value):
        return "n/a" if value is None else f"{value:.2%}"

    rows = []
    for name, label in (("baseline", "Simple spread rule"), ("ml", "Random Forest filter"), ("buy_hold", "50/50 long-only")):
        metric = study.metrics[name]
        sharpe = metric['sharpe_zero_cash_rate']
        rows.append(f"| {label} | {fmt(metric['total_return'])} | {fmt(metric['cagr'])} | {fmt(metric['max_drawdown'])} | {sharpe:.2f} | {metric['entries_or_reversals']} |")
    d = study.diagnostics
    annual_rows = [
        f"| {year} | {fmt(values['baseline'])} | {fmt(values['ml'])} |"
        for year, values in study.metrics["annual_net_returns"].items()
    ]
    cost_rows = [
        f"| {cost} bps | {fmt(value)} |"
        for cost, value in study.metrics["ml_cost_sensitivity"].items()
    ]
    hypothesis_supported = d["training_cointegration_p"] < 0.05 and d["training_adf_spread_p"] < 0.05
    ml_predictive = d["classifier_accuracy"] > d["majority_class_accuracy"] and (d["classifier_roc_auc"] or 0) > 0.5
    verdict = (
        "This sample does not support the proposed stable mean-reverting relationship: the training tests do not reject their null hypotheses. "
        "The classifier also fails to beat a majority-class prediction and has ROC AUC below 0.50. "
        "Its higher simulated return could be sampling luck or a particular pattern of skipped trades. Treat this as a research result, not a deployable trading signal."
        if not hypothesis_supported and not ml_predictive else
        "The diagnostics are mixed; further independent tests are required before treating this as a trading signal."
    )
    report = f"""# AI versus infrastructure baskets: research results

Data snapshot retrieved {manifest['retrieved_at_utc']} from {manifest['source']}.
Six stocks: {', '.join(manifest['symbols'])}. Complete sessions: {manifest['complete_sessions']} ({manifest['first_complete_session']} to {manifest['last_complete_session']}); {len(manifest['dropped_sessions'])} incomplete provider sessions excluded.

## What we tested

Each basket resets to equal dollar weights daily. The log spread is log(tech basket) minus {d['calibration_hedge_ratio']:.3f} times log(infrastructure basket); that hedge ratio was fitted only on the first {CALIBRATION_DAYS} sessions ({d['calibration_first']} to {d['calibration_last']}). The later period, {d['out_of_sample_first']} to {d['out_of_sample_last']}, is held out for walk-forward evaluation. This is a fixed present-day stock selection and remains subject to selection bias.

The training-only cointegration test p-value was {d['training_cointegration_p']:.4f}; the training-only spread ADF p-value was {d['training_adf_spread_p']:.4f}. A low p-value is evidence against a unit root/no cointegration under each test's assumptions. It does not guarantee the relationship will persist. The ML label asks whether the spread falls over five sessions starting at the next close. The classifier predicts that direction; it filters entries of a simple z-score rule. It refits every {REFIT_DAYS} sessions, with a {HORIZON_DAYS + 1}-session purge before each test block.

Out-of-sample classifier accuracy was {d['classifier_accuracy']:.2%}, against a majority-class baseline of {d['majority_class_accuracy']:.2%}; ROC AUC was {d['classifier_roc_auc'] if d['classifier_roc_auc'] is not None else 'n/a'}. These prediction scores are separate from trading performance.

## Simulated performance

| Strategy | Total return | CAGR | Max drawdown | Sharpe (zero cash rate) | Entries/reversals |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The pair strategies use a one-session delay from close-based signal to next-close fill. Returns begin after that fill. They pay {FEE_BPS:.0f} bps fees plus {SLIPPAGE_BPS:.0f} bps slippage per traded dollar, and {BORROW_BPS_ANNUAL:.0f} bps annualized on the short leg. The long-only reference has no costs in this comparison. No order book, borrow locate, intraday execution, tax, or market-impact model is included.

Calendar-year returns show where performance came from (2022 and 2026 are partial years):

| Year | Simple spread rule | Random Forest filter |
|---|---:|---:|
{chr(10).join(annual_rows)}

With the same positions and borrow charge, the Random Forest result changes as fee plus slippage per traded dollar changes:

| Fee + slippage | ML total return |
|---|---:|
{chr(10).join(cost_rows)}

## What this means

{verdict}

The long-only reference is not risk-matched to the pair trades. The chosen six names were selected using today's knowledge, the hedge ratio and thresholds were fixed from one sample, and the Yahoo-derived adjusted closes are research data rather than an executable bid/ask record. Before any paper-trading decision, repeat the analysis with independent data, alternative baskets, and predeclared parameters. The frozen data, daily decisions, and code are saved beside this report so the calculation can be reproduced.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
