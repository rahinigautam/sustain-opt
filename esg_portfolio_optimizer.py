from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


TRADING_DAYS = 252


@dataclass(frozen=True)
class OptimizationConfig:
    max_weight: float = 0.30
    max_esg_risk: float = 22.0
    risk_aversion: float = 1.0
    esg_preference: float = 0.05
    num_monte_carlo: int = 100_000
    seed: int = 42


def load_esg_data(path: str = "data/ESGrisk.csv") -> pd.DataFrame:
    esg = pd.read_csv(path)
    esg.columns = esg.columns.str.strip()
    esg["Ticker"] = esg["Ticker"].str.strip()
    esg = esg.set_index("Ticker")
    esg = esg.rename(columns={"ESG Score": "ESG_Risk"})

    required_columns = {"ESG_Risk", "Environment", "Social", "Governance", "Sector"}
    missing = required_columns.difference(esg.columns)
    if missing:
        raise ValueError(f"ESG file is missing columns: {sorted(missing)}")

    return esg


def load_close_prices(path: str = "data/stock_prices_auto_adjusted.csv") -> pd.DataFrame:
    prices = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    if "Close" not in prices.columns.get_level_values(0):
        raise ValueError("Price file must contain a 'Close' level from yfinance output.")

    close = prices["Close"].copy()
    close.columns = close.columns.str.strip()
    close = close.sort_index()
    close = close.dropna(how="all")

    return close


def align_assets(close: pd.DataFrame, esg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_tickers = close.columns.intersection(esg.index)
    missing_in_prices = sorted(set(esg.index) - set(close.columns))
    missing_in_esg = sorted(set(close.columns) - set(esg.index))

    if missing_in_prices:
        print(f"Warning: ESG tickers missing from prices: {missing_in_prices}")
    if missing_in_esg:
        print(f"Warning: price tickers missing from ESG data: {missing_in_esg}")
    if len(common_tickers) < 2:
        raise ValueError("Need at least two aligned assets to optimize a portfolio.")

    close = close.loc[:, common_tickers]
    esg = esg.loc[common_tickers]

    return close, esg


def calculate_return_inputs(close: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    daily_returns = close.pct_change(fill_method=None).dropna(how="any")
    mean_returns = daily_returns.mean() * TRADING_DAYS
    cov_matrix = daily_returns.cov() * TRADING_DAYS

    return daily_returns, mean_returns, cov_matrix


def normalize_lower_is_better(series: pd.Series) -> pd.Series:
    denominator = series.max() - series.min()
    if denominator == 0:
        return pd.Series(1.0, index=series.index)

    return 1 - ((series - series.min()) / denominator)


def portfolio_metrics(
    weights: np.ndarray,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    esg_risk: pd.Series,
    esg_quality: pd.Series,
) -> dict[str, float]:
    annual_return = float(np.dot(weights, mean_returns))
    annual_volatility = float(np.sqrt(weights.T @ cov_matrix.values @ weights))
    portfolio_esg_risk = float(np.dot(weights, esg_risk))
    portfolio_esg_quality = float(np.dot(weights, esg_quality))

    return {
        "Return": annual_return,
        "Risk": annual_volatility,
        "ESG_Risk": portfolio_esg_risk,
        "ESG_Quality": portfolio_esg_quality,
    }


def portfolio_utility(metrics: dict[str, float], config: OptimizationConfig) -> float:
    return (
        metrics["Return"]
        - config.risk_aversion * metrics["Risk"]
        + config.esg_preference * metrics["ESG_Quality"]
    )


def optimize_portfolio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    esg_risk: pd.Series,
    esg_quality: pd.Series,
    config: OptimizationConfig,
) -> tuple[pd.Series, dict[str, float]]:
    num_assets = len(mean_returns)

    def objective(weights: np.ndarray) -> float:
        metrics = portfolio_metrics(weights, mean_returns, cov_matrix, esg_risk, esg_quality)
        return -portfolio_utility(metrics, config)

    constraints = [
        {"type": "eq", "fun": lambda weights: np.sum(weights) - 1},
        {"type": "ineq", "fun": lambda weights: config.max_esg_risk - np.dot(weights, esg_risk)},
    ]
    bounds = [(0, config.max_weight) for _ in range(num_assets)]
    initial_guess = np.ones(num_assets) / num_assets

    result = minimize(
        objective,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1_000, "ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError(f"SLSQP failed: {result.message}")

    weights = pd.Series(result.x, index=mean_returns.index, name="Weight")
    metrics = portfolio_metrics(result.x, mean_returns, cov_matrix, esg_risk, esg_quality)
    metrics["Utility"] = portfolio_utility(metrics, config)

    return weights, metrics


def simulate_monte_carlo(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    esg_risk: pd.Series,
    esg_quality: pd.Series,
    config: OptimizationConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    records = []
    num_assets = len(mean_returns)

    for _ in range(config.num_monte_carlo):
        weights = rng.random(num_assets)
        weights = weights / weights.sum()

        if weights.max() > config.max_weight:
            continue

        metrics = portfolio_metrics(weights, mean_returns, cov_matrix, esg_risk, esg_quality)
        if metrics["ESG_Risk"] > config.max_esg_risk:
            continue

        metrics["Utility"] = portfolio_utility(metrics, config)
        records.append(metrics | {ticker: weight for ticker, weight in zip(mean_returns.index, weights)})

    if not records:
        raise RuntimeError("No Monte Carlo portfolios satisfied the constraints.")

    return pd.DataFrame(records)


def main() -> None:
    config = OptimizationConfig()

    esg = load_esg_data()
    close = load_close_prices()
    close, esg = align_assets(close, esg)
    daily_returns, mean_returns, cov_matrix = calculate_return_inputs(close)

    esg_risk = esg["ESG_Risk"]
    esg_quality = normalize_lower_is_better(esg_risk)

    optimal_weights, optimal_metrics = optimize_portfolio(
        mean_returns, cov_matrix, esg_risk, esg_quality, config
    )
    monte_carlo = simulate_monte_carlo(
        mean_returns, cov_matrix, esg_risk, esg_quality, config
    )
    best_mc = monte_carlo.loc[monte_carlo["Utility"].idxmax()]

    print("Aligned tickers:")
    print(list(mean_returns.index))
    print(f"\nDaily return observations used: {len(daily_returns)}")

    print("\nSLSQP optimal metrics:")
    print(pd.Series(optimal_metrics).round(4))

    print("\nSLSQP optimal weights:")
    print(optimal_weights[optimal_weights > 0.001].sort_values(ascending=False).round(4))

    print("\nBest Monte Carlo metrics:")
    print(best_mc[["Return", "Risk", "ESG_Risk", "ESG_Quality", "Utility"]].round(4))


if __name__ == "__main__":
    main()
