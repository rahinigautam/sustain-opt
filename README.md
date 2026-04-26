# ESG Portfolio Optimization using Constrained Mean-Variance Framework
Extension of Mean-Variance Portfolio Theory: an ESG based optimization approach 

#Overview 
This project proposes the integration of Environmental, Social, and Governance (ESG) considerations into classical mean-variance portfolio optimisation, aiming to evaluate whether sustainability metrics add a meaningful aspect alongside risk and return in portfolio selection.

#Problem statement 
Traditional portfolio optimization focuses on maximizing returns for a given level of risk. However, modern investors increasingly care about ESG performance.
This project addresses:
- How can ESG scores be incorporated into portfolio construction?
- What trade-offs exist between return, risk, and ESG constraints?

#Methodology
This project uses a constrained optimization framework.
- Objective: Maximize expected returns / optimize risk-return trade-off
- Constraints:
  -Weights sum to 1 (full investment)
  -No short selling (weights ≥ 0)
  -Diversification constraint (weight ≤ 0.3)
  -ESG ≥ 0.3 (minimum sustainability threshold)
Approach:
- Mean-Variance Optimization
- ESG constraint added as an additional filter or penalty
  Tools and technologies
- Python  
- pandas, numpy (data handling & computation)  
- matplotlib (visualization)  
- scipy.optimize (SLSQP for constrained optimization)
The portfolio return is computed as a weighted sum of asset returns, while risk is measured using portfolio variance derived from the covariance matrix of returns.

#Data
- 5 years' worth of stock data for 15 companies, extracted from yfinance
- ESG scores for each stock.
Dataset: [Google Sheet](https://docs.google.com/spreadsheets/d/1lfa_dgxDnIRKC2bfhYTmGpLdjmJJ43LEzKt1yUT0Yuw/edit?gid=0#gid=0)

#Results
The project visualizes the feasible portfolio space by simulating multiple portfolio weight combinations and plotting their corresponding risk and return.
- Each point represents a possible portfolio
- The distribution illustrates the trade-off between risk and return
An optimal portfolio is obtained using constrained optimization (SLSQP) and highlighted on the plot.
Visualizations include:
- Risk vs Return scatter plot of simulated portfolios
- Highlighted optimal portfolio allocation
- ![Portfolio Space](results/portfolio_plot.png)

#Key Insights 
- The portfolio space demonstrates the inherent trade-off between risk and return
- Not all high-return portfolios are optimal due to increased risk
- The optimized portfolio achieves a balanced trade-off under given constraints, including ESG considerations
- Imposing ESG constraints restricts the feasible portfolio space, leading to different optimal allocations compared to unconstrained optimization

#Limitations
- Limited dataset size
- Simplified ESG scoring model
- Assumes static market conditions

#Future Improvements 
- Use larger and more realistic datasets
- Dynamic portfolio rebalancing
- Advanced models (multi-factor, stochastic methods)

#Author
Rahini Gautam, Mathematics Major @ Lady Shri Ram College for Women

#Notes
This project was presented at the International Conference on Exploring Mathematics and Applied Areas (ICEMAA), conducted by NMRC, Hindu College, University of Delhi

