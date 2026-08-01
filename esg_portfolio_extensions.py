"""Sensitivity analysis and honest out-of-sample tests for ESG portfolios.

This module deliberately does not alter the project's completed optimisation
pipeline.  It is an add-on: it reads the same prices/ESG inputs and builds
new, train-only portfolio estimates for robustness and validation studies.

Run the complete extension from the repository root with::

    python esg_portfolio_extensions.py

Figures and CSV summaries are written to ``results/`` by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


TRADING_DAYS = 252


@dataclass(frozen=True)
class PortfolioParameters:
    """Constraints and preferences shared by all extension experiments."""

    max_weight: float = 0.30
    min_esg_score: float = 22.0
    risk_aversion: float = 0.50
    wasserstein_radius: float = 0.02


def load_project_inputs(
    prices_path: str | Path = "data/stock_prices_auto_adjusted.csv",
    esg_path: str | Path = "data/ESGrisk.csv",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load aligned adjusted-close prices and numeric ESG scores."""
    prices = pd.read_csv(prices_path, header=[0, 1], index_col=0, parse_dates=True)
    close = prices["Close"].copy()
    close.columns = close.columns.str.strip()
    close = close.sort_index().ffill().dropna(axis=1)

    esg = pd.read_csv(esg_path)
    esg.columns = esg.columns.str.strip()
    esg["Ticker"] = esg["Ticker"].str.strip()
    esg["ESG Score"] = pd.to_numeric(esg["ESG Score"], errors="coerce")
    esg = esg.dropna(subset=["Ticker", "ESG Score"]).drop_duplicates("Ticker").set_index("Ticker")

    tickers = close.columns.intersection(esg.index, sort=False)
    if len(tickers) < 2:
        raise ValueError("At least two price series with ESG scores are required.")
    return close.loc[:, tickers], esg.loc[tickers, "ESG Score"]


def daily_returns(close_prices: pd.DataFrame) -> pd.DataFrame:
    """Create a complete, date-indexed daily simple-return matrix."""
    returns = close_prices.pct_change(fill_method=None).dropna(how="any")
    if len(returns) < 30:
        raise ValueError("Insufficient return history after cleaning prices.")
    return returns


def _annual_inputs(returns: pd.DataFrame, covariance: str) -> tuple[np.ndarray, np.ndarray]:
    """Estimate annualised means and either sample or Ledoit--Wolf covariance."""
    means = returns.mean().to_numpy() * TRADING_DAYS
    if covariance == "sample":
        matrix = returns.cov().to_numpy() * TRADING_DAYS
    elif covariance == "ledoit_wolf":
        matrix = LedoitWolf().fit(returns.to_numpy()).covariance_ * TRADING_DAYS
    else:
        raise ValueError("covariance must be 'sample' or 'ledoit_wolf'.")
    return means, matrix


def _risk_factor(covariance: np.ndarray) -> np.ndarray:
    """Return L with LL' equal to a numerically PSD covariance matrix."""
    symmetric = (covariance + covariance.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    return eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0, None)))


def _solver() -> str:
    available = cp.installed_solvers()
    for candidate in ("CLARABEL", "ECOS", "SCS"):
        if candidate in available:
            return candidate
    raise RuntimeError("No compatible CVXPY conic solver is installed.")


def solve_training_portfolio(
    train_returns: pd.DataFrame,
    esg_scores: pd.Series,
    parameters: PortfolioParameters,
    covariance: str = "ledoit_wolf",
    robust: bool = False,
) -> pd.Series:
    """Solve a long-only ESG portfolio using *only* the supplied training data.

    ``robust=True`` adds the Wasserstein dual L2 penalty.  This is the same
    robust formulation as the completed DRO notebook, expressed here as a
    reusable function for experiments and out-of-sample validation.
    """
    expected_returns, covariance_matrix = _annual_inputs(train_returns, covariance)
    factor = _risk_factor(covariance_matrix)
    n_assets = train_returns.shape[1]
    if len(esg_scores) != n_assets:
        raise ValueError("ESG scores must align with the return columns.")

    weights = cp.Variable(n_assets)
    volatility = cp.norm(factor.T @ weights, 2)
    objective = expected_returns @ weights - parameters.risk_aversion * volatility
    if robust:
        objective -= parameters.wasserstein_radius * cp.norm(weights, 2)

    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights <= parameters.max_weight,
        esg_scores.to_numpy() @ weights >= parameters.min_esg_score,
    ]
    problem = cp.Problem(cp.Maximize(objective), constraints)
    if not problem.is_dcp():
        raise RuntimeError("Portfolio problem unexpectedly violates DCP rules.")
    problem.solve(solver=_solver(), verbose=False)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Optimisation failed: {problem.status}")
    return pd.Series(np.asarray(weights.value).ravel(), index=train_returns.columns, name="weight")


def _portfolio_summary(weights: pd.Series, evaluation_returns: pd.DataFrame, esg_scores: pd.Series) -> dict[str, float]:
    """Compute requested metrics using realised/estimated daily returns."""
    series = evaluation_returns @ weights
    annual_return = float(series.mean() * TRADING_DAYS)
    annual_volatility = float(series.std(ddof=1) * np.sqrt(TRADING_DAYS))
    wealth = (1 + series).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": annual_return / annual_volatility if annual_volatility else np.nan,
        "portfolio_esg_score": float(weights @ esg_scores),
        "max_weight": float(weights.max()),
        "l2_weight_norm": float(np.linalg.norm(weights.to_numpy(), 2)),
        "active_holdings": int((weights > 1e-6).sum()),
        "cumulative_return": float(wealth.iloc[-1] - 1),
        "maximum_drawdown": float(drawdown.min()),
    }


def run_sensitivity_analysis(
    returns: pd.DataFrame,
    esg_scores: pd.Series,
    baseline: PortfolioParameters = PortfolioParameters(),
    wasserstein_radii: Iterable[float] = (0.00, 0.01, 0.02, 0.03, 0.05),
    esg_thresholds: Iterable[float] = (18.0, 20.0, 22.0, 24.0, 26.0),
    risk_aversions: Iterable[float] = (0.10, 0.25, 0.50, 0.75, 1.00),
) -> dict[str, pd.DataFrame]:
    """Perform one-way DRO sensitivity sweeps, retaining other inputs fixed."""
    sweeps = {
        "wasserstein_radius": wasserstein_radii,
        "min_esg_score": esg_thresholds,
        "risk_aversion": risk_aversions,
    }
    output: dict[str, pd.DataFrame] = {}
    for parameter_name, values in sweeps.items():
        rows = []
        for value in values:
            changed = {parameter_name: float(value)}
            params = PortfolioParameters(**{**baseline.__dict__, **changed})
            weights = solve_training_portfolio(returns, esg_scores, params, covariance="ledoit_wolf", robust=True)
            rows.append({parameter_name: value, **_portfolio_summary(weights, returns, esg_scores)})
        output[parameter_name] = pd.DataFrame(rows).sort_values(parameter_name).reset_index(drop=True)
    return output


def plot_sensitivity_results(results: dict[str, pd.DataFrame], output_dir: str | Path) -> list[Path]:
    """Save clean six-panel, publication-ready sensitivity figures."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("annual_return", "Annualised return"), ("annual_volatility", "Annualised volatility"),
        ("sharpe_ratio", "Sharpe ratio (rf = 0)"), ("portfolio_esg_score", "Portfolio ESG score"),
        ("max_weight", "Maximum weight"), ("l2_weight_norm", "L2 weight norm"),
        ("active_holdings", "Active holdings"),
    ]
    paths = []
    plt.style.use("seaborn-v0_8-whitegrid")
    for parameter, frame in results.items():
        fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), constrained_layout=True)
        for axis, (metric, label) in zip(axes.flat, metrics):
            axis.plot(frame[parameter], frame[metric], marker="o", color="#0B5D7A", linewidth=2)
            axis.set_xlabel(parameter.replace("_", " ").title())
            axis.set_ylabel(label)
            axis.spines[["top", "right"]].set_visible(False)
        axes.flat[-1].axis("off")
        fig.suptitle(f"DRO Sensitivity Analysis: {parameter.replace('_', ' ').title()}", fontsize=15, fontweight="bold")
        path = output / f"sensitivity_{parameter}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def out_of_sample_validation(
    returns: pd.DataFrame,
    esg_scores: pd.Series,
    parameters: PortfolioParameters = PortfolioParameters(),
    train_fraction: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    """Train on the early period, freeze weights, and evaluate on later dates.

    The test period is never used in estimating expected returns, covariance,
    shrinkage, or DRO weights.  Returned wealth curves are indexed by test date.
    """
    if not 0.5 <= train_fraction < 1:
        raise ValueError("train_fraction must be in [0.5, 1).")
    split = int(len(returns) * train_fraction)
    train, test = returns.iloc[:split], returns.iloc[split:]
    if len(test) < 20:
        raise ValueError("Test period is too short for a meaningful evaluation.")

    weights = {
        "Sample covariance": solve_training_portfolio(train, esg_scores, parameters, covariance="sample"),
        "Ledoit-Wolf": solve_training_portfolio(train, esg_scores, parameters, covariance="ledoit_wolf"),
        "Wasserstein DRO": solve_training_portfolio(train, esg_scores, parameters, covariance="ledoit_wolf", robust=True),
        "Equal weight": pd.Series(1 / train.shape[1], index=train.columns, name="weight"),
    }
    summaries, wealth = {}, {}
    for name, portfolio_weights in weights.items():
        summaries[name] = _portfolio_summary(portfolio_weights, test, esg_scores)
        wealth[name] = (1 + test @ portfolio_weights).cumprod()
    table = pd.DataFrame(summaries).T
    table.index.name = "portfolio"
    return table, pd.DataFrame(wealth), weights


def plot_out_of_sample_wealth(wealth: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a publication-quality cumulative wealth comparison."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(11, 6))
    for column in wealth:
        axis.plot(wealth.index, wealth[column], linewidth=2.2, label=column)
    axis.set_title("Out-of-Sample Cumulative Wealth (Weights Frozen After Training)", fontweight="bold")
    axis.set_xlabel("Date")
    axis.set_ylabel("Growth of $1")
    axis.legend(frameon=True)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    """Run all add-on studies and persist reproducible tables and figures."""
    close, esg = load_project_inputs()
    returns = daily_returns(close)
    output = Path("results")
    output.mkdir(parents=True, exist_ok=True)

    sensitivity = run_sensitivity_analysis(returns, esg)
    for name, table in sensitivity.items():
        table.to_csv(output / f"sensitivity_{name}.csv", index=False)
    plot_sensitivity_results(sensitivity, output)

    performance, wealth, _ = out_of_sample_validation(returns, esg)
    performance.to_csv(output / "out_of_sample_performance.csv")
    wealth.to_csv(output / "out_of_sample_wealth.csv")
    plot_out_of_sample_wealth(wealth, output / "out_of_sample_wealth.png")
    print("Out-of-sample performance (test period only):")
    print(performance.round(4))


if __name__ == "__main__":
    main()
