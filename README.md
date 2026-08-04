Overview

This project develops an ESG-constrained portfolio optimization framework by combining classical portfolio theory with modern quantitative finance techniques. The objective is to construct portfolios that balance expected return, risk, and environmental, social, and governance (ESG) considerations while improving robustness against estimation uncertainty.

Features

* Historical price and ESG data collection
* Monte Carlo portfolio simulation
* Markowitz mean-risk optimization
* Ledoit–Wolf covariance shrinkage
* Convex optimization using CVXPY
* Random Matrix Theory (RMT) diagnostics
* Distributionally Robust Optimization (Wasserstein regularization)
* Sensitivity analysis
* Out-of-sample portfolio validation

Methodology

The project follows the workflow:
Data Collection
        ↓
Monte Carlo Portfolio Simulation
        ↓
Markowitz Portfolio Optimisation
        ↓
Ledoit–Wolf Covariance Estimation
        ↓
CVXPY Convex Reformulation
        ↓
Random Matrix Theory Diagnostics
        ↓
Distributionally Robust Optimisation
        ↓
Sensitivity Analysis
        ↓
Out-of-Sample Validation

Results

The project compares multiple portfolio construction techniques under ESG constraints and evaluates their robustness using sensitivity analysis and out-of-sample testing. Results illustrate how covariance estimation and robust optimization influence portfolio stability, diversification, and risk-return characteristics.

Future Improvements

* Dynamic portfolio rebalancing
* Transaction cost modelling
* Time-varying ESG scores
* Multi-period portfolio optimization
* Black–Litterman expected return estimation
* Real market deployment and backtesting
