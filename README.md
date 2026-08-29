# Australian Multi-Asset Portfolio Construction

Black-Litterman portfolio construction, constrained optimisation, rolling out-of-sample testing, and factor attribution for an Australian investor.

## Research question

Can a transparent Black-Litterman framework improve the risk-adjusted performance of diversified Australian ETF portfolios while respecting realistic allocation constraints?

The analysis uses ten ASX-listed ETFs spanning Australian and global equities, listed property, bonds, and gold. It constructs conservative, balanced, and growth profiles and compares benchmark, equal-weight, mean-variance, and maximum-Sharpe portfolios.

## What the project demonstrates

- End-to-end financial data ingestion and cleaning.
- Australian-dollar portfolio construction using the RBA cash-rate target as the local risk-free proxy.
- Currency-consistent Kenneth French factor regressions: AUD returns are converted to USD before being compared with USD-denominated factors.
- Black-Litterman posterior return estimation under absolute, relative, and combined views.
- Long-only SLSQP optimisation with asset caps and minimum bond allocations.
- 504-trading-day rolling estimation windows and monthly out-of-sample rebalancing.
- HAC/Newey-West inference, risk contribution analysis, turnover monitoring, and transaction-cost sensitivity.
- Numerical validation of posterior estimates, portfolio feasibility, optimisation convergence, and risk-contribution sums.

## Repository structure

```text
.
|-- README.md
|-- requirements.txt
|-- src/
|   `-- australian_multi_asset_black_litterman.py
|-- report/
|   `-- Ziyi_Qiu_Black_Litterman_Project_Summary.pdf
|-- data/
|   |-- raw/
|   `-- processed/
`-- output/
    |-- figures/
    `-- tables/
```

The `data/` and `output/` directories are created automatically when the analysis is run.

## Reproduce the analysis

Python 3.10 or later is recommended.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python src/australian_multi_asset_black_litterman.py
```

The script downloads market data from Yahoo Finance, the Reserve Bank of Australia, and the Kenneth R. French Data Library. Internet access is therefore required for a clean run.

## Portfolio design

The investable universe contains:

- Australian equities: VAS, VSO, and VHY.
- Australian listed property: VAP.
- Global developed equities: VGS and VGAD.
- Emerging markets: VGE.
- Australian and global bonds: VAF and VBND.
- AUD-hedged gold: QAU.

The optimiser is long-only. Individual asset caps, a lower bond allocation for each risk profile, and tighter limits on small companies, property, and gold are imposed to keep portfolios investable and diversified.

## Black-Litterman assumptions

The project uses four illustrative views:

- VGS expected return: 8%.
- QAU expected return: 5%.
- VGS expected to outperform VAS by 2%.
- VHY expected to outperform VAS by 1%.

These are research scenarios rather than forecasts claimed to be known with certainty. The analysis therefore tests absolute-only, relative-only, and combined views, varies confidence levels, and reports sensitivity to tau and covariance shrinkage. In a production setting, the views would be generated from a documented forecasting model or an investment committee process.

## Backtest and transaction costs

At each month-end, the strategy estimates its inputs using the previous 504 trading days and applies the resulting weights from the next trading day. This avoids using the rebalance day's return in model estimation.

Headline backtest tables are net of a 10-basis-point cost per unit of one-way turnover. Separate outputs report gross returns and 0/5/10/20-basis-point sensitivity. Taxes, market impact, fund fees, borrowing costs, and execution slippage beyond the selected cost assumption are not modelled.

## Key outputs

- `backtest_summary_metrics.csv`: net performance, risk, drawdown, turnover, and gross return comparison.
- `transaction_cost_sensitivity.csv`: results under 0/5/10/20 bps trading costs.
- `backtest_rebalance_weights.csv`: month-end target weights.
- `portfolio_factor_regression.csv`: factor exposures and HAC t-statistics.
- `robustness_metrics.csv`: sensitivity to tau, view confidence, and covariance shrinkage.
- `validation_checks.csv`: auditable numerical checks.

Representative figures are saved in `output/figures/`.

## Headline out-of-sample results

The table below reports the balanced profile from 2021 to 2026. Returns are annualised arithmetic means and all headline strategy results are net of 10 bps per unit of one-way turnover.

| Portfolio | Annual return | Annual volatility | Excess Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|
| Balanced benchmark | 9.77% | 9.81% | 0.714 | -15.82% |
| Mean-variance | 11.10% | 10.89% | 0.766 | -17.22% |
| Maximum Sharpe | 9.03% | 7.65% | 0.819 | -14.47% |

The maximum-Sharpe portfolio improves risk-adjusted performance primarily by reducing volatility and drawdown, not by maximising raw return. Its excess Sharpe declines only from 0.822 before costs to 0.816 under the 20 bps sensitivity case.

## Limitations

- ETF history is shorter than the history available for broad asset-class indices.
- View returns and confidence levels are scenario assumptions, not live forecasts.
- Yahoo Finance data may be revised and is not an institutional-grade market-data source.
- The cost model is linear and does not estimate nonlinear market impact.
- The analysis is educational research, not investment advice or a production trading system.

## Author

Ziyi Qiu  
Master of Finance and Data Analytics, University of Sydney  
Expected graduation: 2027
