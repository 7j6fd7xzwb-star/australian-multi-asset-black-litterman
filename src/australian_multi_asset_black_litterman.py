"""
CQF Final Project - PC
Australian Multi-Asset Portfolio Construction using Black-Litterman and Factor Views

Single-file quantitative analysis script.

It performs:
1. Data download and cleaning
2. Descriptive statistics and plots
3. Benchmark profile construction
4. Factor analysis and factor regression
5. Black-Litterman posterior return estimation
6. Mean-Variance and Max-Sharpe constrained optimisation
7. Rolling out-of-sample backtesting
8. Robustness and sensitivity analysis
9. Currency-consistent portfolio and factor regressions

Research conventions:
- Portfolio construction is performed from an Australian investor perspective
  in AUD, using the RBA cash-rate target as the local risk-free proxy.
- Kenneth French regional factors are USD-denominated. For those regressions,
  AUD ETF and portfolio returns are converted to USD and the Kenneth French RF
  series is subtracted. This avoids mixing AUD excess returns with USD factors.

Outputs:
- data/raw/
- data/processed/
- output/tables/
- output/figures/
"""

from __future__ import annotations

import io
import math
import zipfile
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import statsmodels.api as sm
import yfinance as yf
from scipy.optimize import minimize

try:
    yf.set_tz_cache_location(str(Path("data/raw") / "yfinance_cache"))
except Exception:
    pass


# =============================================================================
# Configuration
# =============================================================================

START_DATE = "2019-01-01"
# Yahoo Finance end date is exclusive. This reproduces the report sample ending
# on 2026-07-02.
END_DATE = "2026-07-03"
AUDUSD_TICKER = "AUDUSD=X"
TRADING_DAYS = 252
ESTIMATION_WINDOW = 504
TAU = 0.05
COV_SHRINKAGE = 0.10
VIEW_SCENARIOS = ("AbsoluteOnly", "RelativeOnly", "Combined")
BASE_TRANSACTION_COST_BPS = 10.0
TRANSACTION_COST_SCENARIOS_BPS = (0.0, 5.0, 10.0, 20.0)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TABLE_DIR = Path("output/tables")
FIGURE_DIR = Path("output/figures")
for directory in [RAW_DIR, PROCESSED_DIR, TABLE_DIR, FIGURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

ASX_ETFS = {
    "VAS.AX": "Australian equities",
    "VSO.AX": "Australian small companies",
    "VHY.AX": "Australian high dividend",
    "VAP.AX": "Australian listed property",
    "VGS.AX": "Global developed equities",
    "VGAD.AX": "Global developed equities, AUD hedged",
    "VGE.AX": "Emerging markets",
    "VAF.AX": "Australian fixed interest",
    "VBND.AX": "Global aggregate bonds, AUD hedged",
    "QAU.AX": "Gold bullion, AUD hedged",
}
TICKERS = list(ASX_ETFS)

BENCHMARK_PROFILES = pd.DataFrame(
    {
        "Trustee_Conservative": {
            "VAS.AX": 0.15,
            "VSO.AX": 0.02,
            "VHY.AX": 0.05,
            "VAP.AX": 0.03,
            "VGS.AX": 0.18,
            "VGAD.AX": 0.10,
            "VGE.AX": 0.02,
            "VAF.AX": 0.25,
            "VBND.AX": 0.15,
            "QAU.AX": 0.05,
        },
        "Market_Balanced": {
            "VAS.AX": 0.25,
            "VSO.AX": 0.05,
            "VHY.AX": 0.05,
            "VAP.AX": 0.05,
            "VGS.AX": 0.25,
            "VGAD.AX": 0.10,
            "VGE.AX": 0.05,
            "VAF.AX": 0.10,
            "VBND.AX": 0.05,
            "QAU.AX": 0.05,
        },
        "Kelly_Growth": {
            "VAS.AX": 0.25,
            "VSO.AX": 0.08,
            "VHY.AX": 0.03,
            "VAP.AX": 0.05,
            "VGS.AX": 0.32,
            "VGAD.AX": 0.07,
            "VGE.AX": 0.10,
            "VAF.AX": 0.03,
            "VBND.AX": 0.02,
            "QAU.AX": 0.05,
        },
    }
).loc[TICKERS]

MIN_BOND = {
    "Trustee_Conservative": 0.25,
    "Market_Balanced": 0.10,
    "Kelly_Growth": 0.05,
}

RISK_AVERSION = {
    "Trustee_Conservative": 6.0,
    "Market_Balanced": 3.0,
    "Kelly_Growth": 1.0,
}


# =============================================================================
# Utilities
# =============================================================================

def annualise_return(r: pd.Series) -> float:
    return float(r.dropna().mean() * TRADING_DAYS)


def annualise_vol(r: pd.Series) -> float:
    return float(r.dropna().std() * math.sqrt(TRADING_DAYS))


def max_drawdown(r: pd.Series) -> float:
    wealth = (1.0 + r.dropna()).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def one_way_turnover(target: pd.Series, pretrade: pd.Series) -> float:
    return float(0.5 * (target - pretrade).abs().sum())


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def asset_caps(tickers: list[str]) -> list[tuple[float, float]]:
    bounds = []
    for ticker in tickers:
        upper = 0.30
        if ticker == "QAU.AX":
            upper = 0.10
        elif ticker == "VAP.AX":
            upper = 0.08
        elif ticker == "VSO.AX":
            upper = 0.10
        bounds.append((0.0, upper))
    return bounds


def optimisation_constraints(tickers: list[str], profile: str) -> list[dict]:
    bond_indices = [tickers.index(t) for t in ["VAF.AX", "VBND.AX"]]
    return [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w, idx=bond_indices: np.sum(w[idx]) - MIN_BOND[profile]},
    ]


def portfolio_metrics(weights: pd.Series, returns: pd.DataFrame, mu: pd.Series, sigma: pd.DataFrame) -> dict:
    weights = weights.reindex(mu.index).fillna(0.0)
    er = float(weights @ mu)
    vol = float(np.sqrt(weights @ sigma @ weights))
    hist = returns[weights.index].dropna().dot(weights)
    return {
        "expected_excess_return": er,
        "expected_volatility": vol,
        "expected_sharpe": er / vol if vol > 0 else np.nan,
        "historical_annual_return": annualise_return(hist),
        "historical_annual_volatility": annualise_vol(hist),
        "historical_max_drawdown": max_drawdown(hist),
    }


# =============================================================================
# Data
# =============================================================================

def download_etf_prices() -> pd.DataFrame:
    print("Downloading ASX ETF adjusted close prices...")
    data = yf.download(
        TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Adj Close"].copy()
    else:
        prices = data[["Adj Close"]].copy()
    prices = prices.reindex(columns=TICKERS).sort_index()
    prices.index.name = "Date"
    prices.to_csv(RAW_DIR / "asx_etf_adjusted_close.csv")

    coverage = pd.DataFrame(
        {
            "description": pd.Series(ASX_ETFS),
            "start": prices.apply(lambda s: s.first_valid_index()),
            "end": prices.apply(lambda s: s.last_valid_index()),
            "observations": prices.count(),
            "missing_values": prices.isna().sum(),
        }
    )
    coverage.to_csv(PROCESSED_DIR / "data_coverage.csv")
    return prices


def download_audusd_price() -> pd.Series:
    """Download USD per AUD so an AUD wealth return can be converted to USD."""
    print("Downloading AUD/USD adjusted close prices...")
    data = yf.download(
        AUDUSD_TICKER,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        fx = data["Adj Close"].iloc[:, 0]
    else:
        fx = data["Adj Close"]
    fx = pd.to_numeric(fx, errors="coerce").sort_index()
    fx.name = "USD_per_AUD"
    fx.to_csv(RAW_DIR / "audusd_adjusted_close.csv")
    return fx


def download_rba_rf() -> pd.Series:
    print("Downloading RBA F1 cash-rate data...")
    url = "https://www.rba.gov.au/statistics/tables/csv/f1-data.csv"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    text = response.text
    (RAW_DIR / "rba_f1_money_market_rates.csv").write_text(text, encoding="utf-8")
    rba = pd.read_csv(io.StringIO(text), skiprows=[0])
    date_col = rba.columns[0]
    rba[date_col] = pd.to_datetime(rba[date_col], format="mixed", dayfirst=True, errors="coerce")
    rba = rba.dropna(subset=[date_col]).set_index(date_col)

    numeric = rba.apply(pd.to_numeric, errors="coerce")
    preferred = [
        "Cash Rate Target",
        "Interbank Overnight Cash Rate",
        "EOD 3-month BABs/NCDs",
        "EOD 3-month OIS",
    ]
    available = [col for col in preferred if col in numeric.columns]
    if not available:
        available = [col for col in numeric.columns if "Cash" in str(col) or "3-month" in str(col) or "3 month" in str(col)]
    if not available:
        raise RuntimeError("Could not find a suitable RBA short-rate column.")
    col = available[0]
    annual_rate = numeric[col].dropna() / 100.0
    daily_rf = (1.0 + annual_rate) ** (1.0 / TRADING_DAYS) - 1.0
    daily_rf.name = "daily_risk_free_return"
    numeric.to_csv(PROCESSED_DIR / "rba_f1_clean.csv")
    daily_rf.to_csv(PROCESSED_DIR / "daily_risk_free_return.csv")
    return daily_rf


def download_french_file(label: str, url: str) -> pd.DataFrame:
    print(f"Downloading Kenneth French file: {label}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        raw = zf.read(csv_name).decode("latin1")

    lines = raw.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith(","))
    end = next((i for i in range(start + 1, len(lines)) if not lines[i].strip()), len(lines))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:end])))
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0
    df.to_csv(PROCESSED_DIR / f"{label}.csv")
    return df


def build_returns(prices: pd.DataFrame, rf: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    returns.to_csv(PROCESSED_DIR / "asx_etf_daily_returns.csv")

    rf_aligned = rf.reindex(returns.index).ffill().fillna(0.0)
    excess = returns.sub(rf_aligned, axis=0)
    excess.to_csv(PROCESSED_DIR / "asx_etf_excess_returns.csv")

    factors = pd.DataFrame(index=returns.index)
    factors["MKT_AU"] = returns["VAS.AX"] - rf_aligned
    factors["SIZE_AU"] = returns["VSO.AX"] - returns["VAS.AX"]
    factors["DIV_VALUE_AU"] = returns["VHY.AX"] - returns["VAS.AX"]
    factors["PROPERTY_AU"] = returns["VAP.AX"] - returns["VAS.AX"]
    factors["GLOBAL_DEV_AU"] = returns["VGS.AX"] - returns["VAS.AX"]
    factors["FX_HEDGE_AU"] = returns["VGAD.AX"] - returns["VGS.AX"]
    factors["BOND_AU"] = returns["VAF.AX"] - rf_aligned
    factors["GOLD_AU"] = returns["QAU.AX"] - rf_aligned
    factors.to_csv(PROCESSED_DIR / "etf_proxy_factors.csv")
    return returns, excess, factors


def run_data_step() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    prices = download_etf_prices()
    audusd = download_audusd_price()
    rf = download_rba_rf()
    download_french_file(
        "asia_pacific_ex_japan_5_factors_daily",
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Asia_Pacific_ex_Japan_5_Factors_Daily_CSV.zip",
    )
    download_french_file(
        "asia_pacific_ex_japan_momentum_daily",
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Asia_Pacific_ex_Japan_Mom_Factor_Daily_CSV.zip",
    )
    returns, excess, factors = build_returns(prices, rf)
    return returns, excess, factors, rf, audusd


# =============================================================================
# Descriptive statistics, factor analysis, and plots
# =============================================================================

def descriptive_analysis(returns: pd.DataFrame, excess: pd.DataFrame, factors: pd.DataFrame) -> None:
    summary = pd.DataFrame(index=TICKERS)
    summary["annual_return"] = returns.apply(annualise_return)
    summary["annual_volatility"] = returns.apply(annualise_vol)
    summary["excess_sharpe"] = excess.apply(annualise_return) / excess.apply(annualise_vol)
    summary["max_drawdown"] = returns.apply(max_drawdown)
    summary["start"] = returns.apply(lambda s: s.first_valid_index())
    summary["end"] = returns.apply(lambda s: s.last_valid_index())
    summary["obs"] = returns.count()
    summary.to_csv(TABLE_DIR / "asset_summary_stats.csv")

    BENCHMARK_PROFILES.to_csv(TABLE_DIR / "benchmark_profiles.csv")
    bench_returns = pd.DataFrame({p: returns.dot(BENCHMARK_PROFILES[p]) for p in BENCHMARK_PROFILES.columns})
    bench_returns.to_csv(TABLE_DIR / "benchmark_daily_returns.csv")
    bench_summary = pd.DataFrame(index=BENCHMARK_PROFILES.columns)
    bench_summary["annual_return"] = bench_returns.apply(annualise_return)
    bench_summary["annual_volatility"] = bench_returns.apply(annualise_vol)
    bench_summary["sharpe_no_rf"] = bench_summary["annual_return"] / bench_summary["annual_volatility"]
    bench_summary["max_drawdown"] = bench_returns.apply(max_drawdown)
    bench_summary.to_csv(TABLE_DIR / "benchmark_summary_stats.csv")

    plt.figure(figsize=(10, 6))
    ((1 + returns[TICKERS]).cumprod()).plot(ax=plt.gca(), lw=1.4)
    plt.title("ASX ETF Cumulative Returns")
    plt.ylabel("Growth of $1")
    plt.grid(True, alpha=0.3)
    savefig("asset_cumulative_returns.png")

    plt.figure(figsize=(9, 7))
    sns.heatmap(returns[TICKERS].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("ASX ETF Return Correlation")
    savefig("asset_correlation_heatmap.png")

    plt.figure(figsize=(9, 5))
    BENCHMARK_PROFILES.T.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="tab20")
    plt.title("Benchmark Profile Weights")
    plt.ylabel("Weight")
    plt.legend(ncol=2, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig("benchmark_profile_weights.png")

    plt.figure(figsize=(9, 5))
    ((1 + bench_returns).cumprod()).plot(ax=plt.gca(), lw=1.8)
    plt.title("Benchmark Cumulative Returns")
    plt.ylabel("Growth of $1")
    plt.grid(True, alpha=0.3)
    savefig("benchmark_cumulative_returns.png")

    plt.figure(figsize=(9, 5))
    factors[["MKT_AU", "SIZE_AU", "DIV_VALUE_AU", "BOND_AU", "GOLD_AU"]].cumsum().plot(ax=plt.gca())
    plt.title("Selected ETF Proxy Factor Cumulative Returns")
    plt.grid(True, alpha=0.3)
    savefig("etf_proxy_factor_cumulative_returns.png")


def fit_factor_model(y: pd.Series, factors: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Estimate a factor model with Newey-West/HAC standard errors."""
    df = pd.concat([y.rename("dependent"), factors], axis=1, sort=True).dropna()
    x = sm.add_constant(df[factors.columns])
    return sm.OLS(df["dependent"], x).fit(cov_type="HAC", cov_kwds={"maxlags": 5})


def factor_regression(returns_aud: pd.DataFrame, audusd: pd.Series) -> pd.DataFrame:
    ff5 = pd.read_csv(PROCESSED_DIR / "asia_pacific_ex_japan_5_factors_daily.csv", index_col=0, parse_dates=True)
    mom = pd.read_csv(PROCESSED_DIR / "asia_pacific_ex_japan_momentum_daily.csv", index_col=0, parse_dates=True)
    ff = ff5.join(mom, how="inner")
    factor_cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "WML"] if c in ff.columns]
    fx_return = audusd.reindex(returns_aud.index).ffill().pct_change(fill_method=None)
    returns_usd = (1.0 + returns_aud).mul(1.0 + fx_return, axis=0) - 1.0
    excess_usd = returns_usd.sub(ff["RF"].reindex(returns_usd.index), axis=0)
    returns_usd.to_csv(PROCESSED_DIR / "asx_etf_returns_usd.csv")
    excess_usd.to_csv(PROCESSED_DIR / "asx_etf_excess_returns_usd.csv")

    rows = []
    for ticker in TICKERS:
        model = fit_factor_model(excess_usd[ticker], ff[factor_cols])
        row = {"ticker": ticker, "alpha_annual": model.params["const"] * TRADING_DAYS, "r_squared": model.rsquared}
        for col in factor_cols:
            row[f"beta_{col}"] = model.params[col]
            row[f"tstat_{col}"] = model.tvalues[col]
        rows.append(row)

    out = pd.DataFrame(rows).set_index("ticker")
    out.to_csv(TABLE_DIR / "factor_regression_betas.csv")

    beta_cols = [c for c in out.columns if c.startswith("beta_")]
    plt.figure(figsize=(9, 6))
    sns.heatmap(out[beta_cols], annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("Factor Regression Betas")
    savefig("factor_regression_beta_heatmap.png")

    rolling = []
    mkt = ff["Mkt-RF"].reindex(excess_usd.index)
    for ticker in ["VAS.AX", "VSO.AX", "VGS.AX", "VAF.AX", "QAU.AX"]:
        beta = excess_usd[ticker].rolling(252).cov(mkt) / mkt.rolling(252).var()
        rolling.append(beta.rename(ticker))
    rolling_df = pd.concat(rolling, axis=1)
    plt.figure(figsize=(10, 5))
    rolling_df.plot(ax=plt.gca())
    plt.title("Rolling 252-Day Beta to Kenneth French Mkt-RF")
    plt.grid(True, alpha=0.3)
    savefig("rolling_beta_examples.png")

    factor_rows = []
    for factor in [c for c in ["SMB", "HML", "RMW", "CMA", "WML"] if c in ff.columns]:
        model = fit_factor_model(ff[factor], ff[["Mkt-RF"]])
        factor_rows.append(
            {
                "factor": factor,
                "annual_mean": ff[factor].mean() * TRADING_DAYS,
                "annual_volatility": ff[factor].std() * math.sqrt(TRADING_DAYS),
                "market_beta": model.params["Mkt-RF"],
                "market_beta_tstat_hac": model.tvalues["Mkt-RF"],
                "alpha_annual": model.params["const"] * TRADING_DAYS,
                "alpha_tstat_hac": model.tvalues["const"],
                "r_squared": model.rsquared,
            }
        )
    factor_market = pd.DataFrame(factor_rows).set_index("factor")
    factor_market.to_csv(TABLE_DIR / "factor_on_market_analysis.csv")
    plt.figure(figsize=(8, 4.8))
    factor_market["market_beta"].sort_values().plot(kind="bar", color="#4472C4", ax=plt.gca())
    plt.axhline(0.0, color="#333333", lw=0.8)
    plt.title("Factor Exposure to Asia-Pacific ex-Japan Market")
    plt.ylabel("OLS beta (HAC inference)")
    savefig("factor_on_market_betas.png")
    return out


# =============================================================================
# Black-Litterman and optimisation
# =============================================================================

def bl_views(
    tickers: list[str],
    view_scenario: str = "Combined",
    save_inputs: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build absolute-only, relative-only, or combined BL view inputs."""
    p = pd.DataFrame(0.0, index=["abs_VGS_8pct", "abs_QAU_5pct", "rel_VGS_minus_VAS_2pct", "rel_VHY_minus_VAS_1pct"], columns=tickers)
    p.loc["abs_VGS_8pct", "VGS.AX"] = 1.0
    p.loc["abs_QAU_5pct", "QAU.AX"] = 1.0
    p.loc["rel_VGS_minus_VAS_2pct", "VGS.AX"] = 1.0
    p.loc["rel_VGS_minus_VAS_2pct", "VAS.AX"] = -1.0
    p.loc["rel_VHY_minus_VAS_1pct", "VHY.AX"] = 1.0
    p.loc["rel_VHY_minus_VAS_1pct", "VAS.AX"] = -1.0
    q = pd.Series([0.08, 0.05, 0.02, 0.01], index=p.index, name="Q")
    confidence = pd.Series([0.55, 0.45, 0.60, 0.50], index=p.index, name="confidence")
    if view_scenario == "AbsoluteOnly":
        selected = p.index.str.startswith("abs_")
    elif view_scenario == "RelativeOnly":
        selected = p.index.str.startswith("rel_")
    elif view_scenario == "Combined":
        selected = np.ones(len(p), dtype=bool)
    else:
        raise ValueError(f"Unknown BL view scenario: {view_scenario}")
    p, q, confidence = p.loc[selected], q.loc[selected], confidence.loc[selected]
    if save_inputs:
        suffix = view_scenario.lower()
        p.to_csv(TABLE_DIR / f"bl_views_matrix_P_{suffix}.csv")
        q.to_csv(TABLE_DIR / f"bl_views_vector_Q_{suffix}.csv")
        confidence.to_csv(TABLE_DIR / f"bl_view_confidence_{suffix}.csv")
    return p, q, confidence


def shrunk_cov(excess: pd.DataFrame, shrinkage: float = COV_SHRINKAGE) -> pd.DataFrame:
    sigma = excess.cov() * TRADING_DAYS
    target = pd.DataFrame(np.diag(np.diag(sigma)), index=sigma.index, columns=sigma.columns)
    return (1 - shrinkage) * sigma + shrinkage * target


def bl_posterior(
    excess: pd.DataFrame,
    benchmark_weights: pd.Series,
    profile: str,
    tau: float = TAU,
    confidence_scale: float = 1.0,
    shrinkage: float = COV_SHRINKAGE,
    view_scenario: str = "Combined",
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    tickers = list(excess.columns)
    sigma = shrunk_cov(excess[tickers], shrinkage)
    delta = RISK_AVERSION[profile]
    pi = pd.Series(delta * sigma.values @ benchmark_weights.reindex(tickers).values, index=tickers)
    p, q, conf = bl_views(tickers, view_scenario=view_scenario)

    p_mat = p.values
    sigma_mat = sigma.values
    tau_sigma = tau * sigma_mat
    base_uncertainty = np.diag(p_mat @ tau_sigma @ p_mat.T)
    adjusted_conf = np.clip(conf.values * confidence_scale, 0.05, 0.95)
    omega_diag = base_uncertainty * ((1.0 - adjusted_conf) / adjusted_conf)
    omega = np.diag(omega_diag)

    inv_tau_sigma = np.linalg.inv(tau_sigma)
    inv_omega = np.linalg.inv(omega)
    posterior_cov = np.linalg.inv(inv_tau_sigma + p_mat.T @ inv_omega @ p_mat)
    posterior_mean = posterior_cov @ (inv_tau_sigma @ pi.values + p_mat.T @ inv_omega @ q.values)
    mu_bl = pd.Series(posterior_mean, index=tickers)
    return pi, mu_bl, sigma


def optimise_portfolio(mu: pd.Series, sigma: pd.DataFrame, profile: str, method: str, x0: pd.Series | None = None) -> pd.Series:
    tickers = list(mu.index)
    bounds = asset_caps(tickers)
    constraints = optimisation_constraints(tickers, profile)

    if x0 is None:
        start = BENCHMARK_PROFILES[profile].reindex(tickers).values
    else:
        start = x0.reindex(tickers).fillna(0.0).values
    start = np.array([min(max(v, lo), hi) for v, (lo, hi) in zip(start, bounds)])
    start = start / start.sum()

    sigma_values = sigma.loc[tickers, tickers].values
    mu_values = mu.reindex(tickers).values
    delta = RISK_AVERSION[profile]

    if method == "MeanVariance":
        objective = lambda w: -(w @ mu_values - 0.5 * delta * (w @ sigma_values @ w))
    elif method == "MaxSharpe":
        objective = lambda w: -(w @ mu_values) / max(np.sqrt(w @ sigma_values @ w), 1e-12)
    else:
        raise ValueError(method)

    res = minimize(objective, start, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 1000, "ftol": 1e-10})
    if not res.success:
        raise RuntimeError(f"Optimisation failed for {profile}/{method}: {res.message}")
    w = pd.Series(res.x, index=tickers)
    w[w.abs() < 1e-8] = 0.0
    w = w / w.sum()
    bound_violation = max(
        max(lo - w[ticker], 0.0, w[ticker] - hi)
        for ticker, (lo, hi) in zip(tickers, bounds)
    )
    bond_weight = w.reindex(["VAF.AX", "VBND.AX"]).sum()
    if abs(w.sum() - 1.0) > 1e-7 or bound_violation > 1e-7 or bond_weight + 1e-7 < MIN_BOND[profile]:
        raise RuntimeError(f"Infeasible solution returned for {profile}/{method}.")
    return w


def run_black_litterman(excess: pd.DataFrame, returns: pd.DataFrame) -> None:
    excess = excess[TICKERS].dropna()
    weights_rows, metrics_rows, risk_rows, posterior_rows, priors = [], [], [], [], {}

    for view_scenario in VIEW_SCENARIOS:
        bl_views(TICKERS, view_scenario=view_scenario, save_inputs=True)

    for profile in BENCHMARK_PROFILES.columns:
        bench_w = BENCHMARK_PROFILES[profile]
        for view_scenario in VIEW_SCENARIOS:
            pi, mu_bl, sigma = bl_posterior(
                excess, bench_w, profile, view_scenario=view_scenario
            )
            priors[profile] = pi
            for ticker, value in mu_bl.items():
                posterior_rows.append(
                    {
                        "profile": profile,
                        "view_scenario": view_scenario,
                        "ticker": ticker,
                        "posterior_return": value,
                    }
                )

            for method in ["MeanVariance", "MaxSharpe"]:
                w = optimise_portfolio(mu_bl, sigma, profile, method)
                portfolio_name = f"{profile}_{view_scenario}_{method}"
                weights_rows.append(
                    pd.Series(
                        {
                            "profile": profile,
                            "view_scenario": view_scenario,
                            "method": method,
                            **w.to_dict(),
                        },
                        name=portfolio_name,
                    )
                )
                row = {
                    "portfolio": portfolio_name,
                    "profile": profile,
                    "view_scenario": view_scenario,
                    "method": method,
                    **portfolio_metrics(w, returns[TICKERS], mu_bl, sigma),
                }
                metrics_rows.append(row)
                port_var = float(w @ sigma @ w)
                for ticker in TICKERS:
                    mcr = float((sigma @ w)[ticker])
                    risk_rows.append(
                        {
                            "portfolio": portfolio_name,
                            "profile": profile,
                            "view_scenario": view_scenario,
                            "method": method,
                            "ticker": ticker,
                            "risk_contribution": w[ticker] * mcr / port_var,
                        }
                    )

            if view_scenario == "Combined":
                portfolio_name = f"{profile}_Combined_Benchmark"
                weights_rows.append(
                    pd.Series(
                        {
                            "profile": profile,
                            "view_scenario": view_scenario,
                            "method": "Benchmark",
                            **bench_w.to_dict(),
                        },
                        name=portfolio_name,
                    )
                )
                metrics_rows.append(
                    {
                        "portfolio": portfolio_name,
                        "profile": profile,
                        "view_scenario": view_scenario,
                        "method": "Benchmark",
                        **portfolio_metrics(
                            bench_w, returns[TICKERS], mu_bl, sigma
                        ),
                    }
                )
                port_var = float(bench_w @ sigma @ bench_w)
                for ticker in TICKERS:
                    mcr = float((sigma @ bench_w)[ticker])
                    risk_rows.append(
                        {
                            "portfolio": portfolio_name,
                            "profile": profile,
                            "view_scenario": view_scenario,
                            "method": "Benchmark",
                            "ticker": ticker,
                            "risk_contribution": bench_w[ticker] * mcr / port_var,
                        }
                    )

    priors_df = pd.DataFrame(priors)
    post_long = pd.DataFrame(posterior_rows)
    post_df = post_long.pivot_table(
        index="ticker", columns=["profile", "view_scenario"], values="posterior_return"
    )
    weights_df = pd.DataFrame(weights_rows)
    metrics_df = pd.DataFrame(metrics_rows).set_index("portfolio")
    risk_df = pd.DataFrame(risk_rows)

    priors_df.to_csv(TABLE_DIR / "bl_prior_equilibrium_returns.csv")
    post_df.to_csv(TABLE_DIR / "bl_posterior_returns.csv")
    post_long.to_csv(TABLE_DIR / "bl_posterior_returns_long.csv", index=False)
    weights_df.to_csv(TABLE_DIR / "bl_optimized_and_benchmark_weights.csv")
    metrics_df.to_csv(TABLE_DIR / "bl_portfolio_metrics.csv")
    risk_df.to_csv(TABLE_DIR / "bl_risk_contributions.csv", index=False)
    shrunk_cov(excess).to_csv(TABLE_DIR / "bl_covariance_matrix_shrunk.csv")

    plt.figure(figsize=(8, 5))
    pd.DataFrame(
        {
            "Prior": priors_df["Market_Balanced"],
            "Absolute only": post_df[("Market_Balanced", "AbsoluteOnly")],
            "Relative only": post_df[("Market_Balanced", "RelativeOnly")],
            "Combined": post_df[("Market_Balanced", "Combined")],
        }
    ).plot(kind="bar", ax=plt.gca())
    plt.title("Market Balanced Prior vs Posterior Returns")
    plt.ylabel("Annual excess return")
    savefig("bl_prior_vs_posterior_market_balanced.png")

    plot_w = weights_df.set_index(["profile", "view_scenario", "method"])[TICKERS]
    plt.figure(figsize=(12, 8))
    sns.heatmap(plot_w, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Black-Litterman Optimised and Benchmark Weights")
    savefig("bl_weights_heatmap.png")

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=metrics_df.reset_index(), x="expected_volatility", y="expected_excess_return", hue="profile", style="method", s=90)
    plt.title("Black-Litterman Expected Risk-Return")
    plt.grid(True, alpha=0.3)
    savefig("bl_risk_return_scatter.png")

    mb_compare = metrics_df[
        (metrics_df["profile"] == "Market_Balanced")
        & (metrics_df["method"].isin(["MeanVariance", "MaxSharpe"]))
    ].reset_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=mb_compare,
        x="view_scenario",
        y="expected_sharpe",
        hue="method",
        order=list(VIEW_SCENARIOS),
    )
    plt.title("Market Balanced: Absolute vs Relative BL Views")
    plt.ylabel("Expected Sharpe")
    savefig("bl_absolute_relative_comparison.png")

    risk_plot = risk_df[
        (risk_df["profile"] == "Market_Balanced")
        & (risk_df["view_scenario"] == "Combined")
    ].pivot(index="ticker", columns="method", values="risk_contribution")
    plt.figure(figsize=(8, 6))
    sns.heatmap(risk_plot, annot=True, fmt=".2f", cmap="YlOrBr")
    plt.title("Risk Contributions: Market Balanced Combined Views")
    savefig("bl_risk_contributions_market_balanced.png")


# =============================================================================
# Rolling backtest
# =============================================================================

def month_end_rebalance_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    return list(index.to_series().groupby(index.to_period("M")).tail(1).index)


def simulate_strategy(returns: pd.DataFrame, excess: pd.DataFrame, profile: str, method: str) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    dates = month_end_rebalance_dates(returns.index)
    rebalance_dates = [d for d in dates if returns.index.get_loc(d) >= ESTIMATION_WINDOW]
    daily_returns, turnovers, weight_rows = [], [], []
    current_w = BENCHMARK_PROFILES[profile].copy()

    for i, reb_date in enumerate(rebalance_dates):
        loc = returns.index.get_loc(reb_date)
        window_excess = excess.iloc[loc - ESTIMATION_WINDOW : loc][TICKERS].dropna()
        bench_w = BENCHMARK_PROFILES[profile]

        if method == "Benchmark":
            target_w = bench_w.copy()
        elif method == "EqualWeight":
            target_w = pd.Series(1.0 / len(TICKERS), index=TICKERS)
        else:
            _, mu_bl, sigma = bl_posterior(window_excess, bench_w, profile)
            target_w = optimise_portfolio(mu_bl, sigma, profile, method, x0=current_w)

        turnovers.append((reb_date, one_way_turnover(target_w, current_w)))
        weight_rows.append(pd.Series({"date": reb_date, "profile": profile, "method": method, **target_w.to_dict()}))

        start_pos = loc + 1
        end_pos = returns.index.get_loc(rebalance_dates[i + 1]) + 1 if i + 1 < len(rebalance_dates) else len(returns.index)
        drifting_w = target_w.copy()
        for date in returns.index[start_pos:end_pos]:
            r_vec = returns.loc[date, TICKERS].fillna(0.0)
            port_ret = float(drifting_w @ r_vec)
            daily_returns.append((date, port_ret))
            drifting_w = drifting_w * (1.0 + r_vec)
            drifting_w = drifting_w / drifting_w.sum()
        current_w = drifting_w.copy()

    ret = pd.Series(dict(daily_returns)).sort_index()
    turnover = pd.Series(dict(turnovers), name=f"{profile}_{method}_turnover")
    weights = pd.DataFrame(weight_rows)
    return ret, turnover, weights


def apply_transaction_costs(
    gross_returns: pd.Series,
    turnover: pd.Series,
    cost_bps: float,
) -> pd.Series:
    """Deduct one-way trading costs on the first return after each rebalance."""
    net_returns = gross_returns.copy()
    if cost_bps <= 0 or net_returns.empty:
        return net_returns

    unit_cost = cost_bps / 10_000.0
    for rebalance_date, one_way in turnover.dropna().items():
        following_dates = net_returns.index[net_returns.index > rebalance_date]
        if len(following_dates) == 0:
            continue
        trade_date = following_dates[0]
        net_returns.loc[trade_date] = (
            (1.0 + net_returns.loc[trade_date]) * (1.0 - unit_cost * float(one_way)) - 1.0
        )
    return net_returns


def portfolio_factor_regression(portfolio_returns_aud: pd.DataFrame, audusd: pd.Series) -> pd.DataFrame:
    """Regress realised portfolio returns on USD Kenneth French factors."""
    ff5 = pd.read_csv(PROCESSED_DIR / "asia_pacific_ex_japan_5_factors_daily.csv", index_col=0, parse_dates=True)
    mom = pd.read_csv(PROCESSED_DIR / "asia_pacific_ex_japan_momentum_daily.csv", index_col=0, parse_dates=True)
    ff = ff5.join(mom, how="inner")
    factor_cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "WML"] if c in ff.columns]
    fx_return = audusd.reindex(portfolio_returns_aud.index).ffill().pct_change(fill_method=None)
    portfolio_returns_usd = (1.0 + portfolio_returns_aud).mul(1.0 + fx_return, axis=0) - 1.0
    excess_usd = portfolio_returns_usd.sub(ff["RF"].reindex(portfolio_returns_usd.index), axis=0)
    portfolio_returns_usd.to_csv(TABLE_DIR / "backtest_portfolio_returns_usd.csv")

    rows = []
    for portfolio in excess_usd:
        model = fit_factor_model(excess_usd[portfolio], ff[factor_cols])
        row = {
            "portfolio": portfolio,
            "alpha_annual": model.params["const"] * TRADING_DAYS,
            "alpha_tstat_hac": model.tvalues["const"],
            "r_squared": model.rsquared,
            "observations": int(model.nobs),
        }
        for factor in factor_cols:
            row[f"beta_{factor}"] = model.params[factor]
            row[f"tstat_{factor}_hac"] = model.tvalues[factor]
        rows.append(row)
    results = pd.DataFrame(rows).set_index("portfolio")
    results.to_csv(TABLE_DIR / "portfolio_factor_regression.csv")

    beta_cols = [f"beta_{c}" for c in factor_cols]
    plt.figure(figsize=(10, 7))
    sns.heatmap(results[beta_cols], annot=True, fmt=".2f", cmap="RdBu_r", center=0.0)
    plt.title("Out-of-Sample Portfolio Factor Betas")
    savefig("portfolio_factor_beta_heatmap.png")

    rolling = {}
    mkt = ff["Mkt-RF"].reindex(excess_usd.index)
    for portfolio in [
        "Market_Balanced_Benchmark",
        "Market_Balanced_MeanVariance",
        "Market_Balanced_MaxSharpe",
    ]:
        rolling[portfolio] = excess_usd[portfolio].rolling(252).cov(mkt) / mkt.rolling(252).var()
    rolling_df = pd.DataFrame(rolling)
    rolling_df.to_csv(TABLE_DIR / "portfolio_rolling_market_beta.csv")
    plt.figure(figsize=(10, 5))
    rolling_df.plot(ax=plt.gca(), color=["#7F7F7F", "#ED7D31", "#4472C4"], lw=1.5)
    plt.axhline(0.0, color="#333333", lw=0.8)
    plt.title("Market Balanced Portfolio Rolling 252-Day Market Beta")
    plt.ylabel("Beta to Mkt-RF")
    plt.grid(True, alpha=0.25)
    savefig("portfolio_rolling_market_beta.png")
    return results


def run_backtest(returns: pd.DataFrame, excess: pd.DataFrame, rf: pd.Series, audusd: pd.Series) -> None:
    returns = returns[TICKERS].dropna(how="any")
    excess = excess[TICKERS].reindex(returns.index).dropna(how="any")
    rf = rf.reindex(returns.index).ffill().fillna(0.0)
    common_index = returns.index.intersection(excess.index).intersection(rf.index)
    returns = returns.loc[common_index]
    excess = excess.loc[common_index]
    rf = rf.loc[common_index]

    gross_ret_rows, net_ret_rows, turnover_rows, weight_rows = [], [], [], []
    summary_rows, cost_sensitivity_rows = [], []

    for profile in BENCHMARK_PROFILES.columns:
        for method in ["Benchmark", "EqualWeight", "MeanVariance", "MaxSharpe"]:
            print(f"Backtesting {profile} / {method}")
            gross_ret, turnover, weights = simulate_strategy(returns, excess, profile, method)
            name = f"{profile}_{method}"
            gross_ret.name = name
            net_ret = apply_transaction_costs(gross_ret, turnover, BASE_TRANSACTION_COST_BPS)
            net_ret.name = name
            gross_ret_rows.append(gross_ret)
            net_ret_rows.append(net_ret)
            turnover_rows.append(turnover.rename(name))
            weight_rows.append(weights)
            rf_aligned = rf.reindex(net_ret.index).ffill().fillna(0.0)
            ex = net_ret - rf_aligned
            downside = ex[ex < 0].std() * math.sqrt(TRADING_DAYS)
            summary_rows.append(
                {
                    "portfolio": name,
                    "annual_return": annualise_return(net_ret),
                    "annual_return_gross": annualise_return(gross_ret),
                    "annual_volatility": annualise_vol(net_ret),
                    "excess_sharpe": annualise_return(ex) / annualise_vol(ex),
                    "sortino": annualise_return(ex) / downside if downside > 0 else np.nan,
                    "max_drawdown": max_drawdown(net_ret),
                    "average_one_way_turnover": turnover.mean(),
                    "transaction_cost_bps": BASE_TRANSACTION_COST_BPS,
                    "start": net_ret.index.min(),
                    "end": net_ret.index.max(),
                    "observations": len(net_ret),
                }
            )

            for cost_bps in TRANSACTION_COST_SCENARIOS_BPS:
                scenario_ret = apply_transaction_costs(gross_ret, turnover, cost_bps)
                scenario_rf = rf.reindex(scenario_ret.index).ffill().fillna(0.0)
                scenario_excess = scenario_ret - scenario_rf
                cost_sensitivity_rows.append(
                    {
                        "portfolio": name,
                        "transaction_cost_bps": cost_bps,
                        "annual_return": annualise_return(scenario_ret),
                        "annual_volatility": annualise_vol(scenario_ret),
                        "excess_sharpe": annualise_return(scenario_excess) / annualise_vol(scenario_excess),
                        "max_drawdown": max_drawdown(scenario_ret),
                    }
                )

    portfolio_returns_gross = pd.concat(gross_ret_rows, axis=1)
    portfolio_returns = pd.concat(net_ret_rows, axis=1)
    turnovers = pd.concat(turnover_rows, axis=1)
    weights_history = pd.concat(weight_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows).set_index("portfolio")
    cost_sensitivity = pd.DataFrame(cost_sensitivity_rows)
    portfolio_returns_gross.to_csv(TABLE_DIR / "backtest_daily_returns_gross.csv")
    portfolio_returns.to_csv(TABLE_DIR / f"backtest_daily_returns_net_{BASE_TRANSACTION_COST_BPS:g}bps.csv")
    portfolio_returns.to_csv(TABLE_DIR / "backtest_daily_returns.csv")
    turnovers.to_csv(TABLE_DIR / "backtest_turnover.csv")
    weights_history.to_csv(TABLE_DIR / "backtest_rebalance_weights.csv", index=False)
    summary.to_csv(TABLE_DIR / "backtest_summary_metrics.csv")
    cost_sensitivity.to_csv(TABLE_DIR / "transaction_cost_sensitivity.csv", index=False)
    portfolio_factor_regression(portfolio_returns, audusd)

    balanced_costs = cost_sensitivity[
        cost_sensitivity["portfolio"].eq("Market_Balanced_MaxSharpe")
    ]
    plt.figure(figsize=(8, 4.8))
    sns.lineplot(
        data=balanced_costs,
        x="transaction_cost_bps",
        y="excess_sharpe",
        marker="o",
        color="#4472C4",
    )
    plt.title("Market Balanced Max-Sharpe: Transaction-Cost Sensitivity")
    plt.xlabel("Transaction cost (bps per unit of one-way turnover)")
    plt.ylabel("Out-of-sample excess Sharpe")
    plt.grid(True, alpha=0.3)
    savefig("transaction_cost_sensitivity_market_balanced.png")

    for profile in BENCHMARK_PROFILES.columns:
        cols = [f"{profile}_{m}" for m in ["Benchmark", "EqualWeight", "MeanVariance", "MaxSharpe"]]
        plt.figure(figsize=(9, 5))
        ((1 + portfolio_returns[cols]).cumprod()).plot(ax=plt.gca())
        plt.title(f"Rolling Backtest Cumulative Returns - {profile} (net of {BASE_TRANSACTION_COST_BPS:g} bps costs)")
        plt.ylabel("Growth of $1")
        plt.grid(True, alpha=0.3)
        savefig(f"backtest_cumulative_returns_{profile}.png")

        plt.figure(figsize=(9, 5))
        wealth = (1 + portfolio_returns[cols]).cumprod()
        dd = wealth / wealth.cummax() - 1.0
        dd.plot(ax=plt.gca())
        plt.title(f"Rolling Backtest Drawdowns - {profile} (net of {BASE_TRANSACTION_COST_BPS:g} bps costs)")
        plt.ylabel("Drawdown")
        plt.grid(True, alpha=0.3)
        savefig(f"backtest_drawdowns_{profile}.png")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=summary.reset_index(), x="annual_volatility", y="annual_return", hue="portfolio", s=70, legend=False)
    for idx, row in summary.iterrows():
        plt.text(row["annual_volatility"], row["annual_return"], idx.replace("_", "\n"), fontsize=7)
    plt.title("Rolling Backtest Realised Risk-Return")
    plt.grid(True, alpha=0.3)
    savefig("backtest_realized_risk_return.png")

    plt.figure(figsize=(9, 5))
    summary["max_drawdown"].sort_values().plot(kind="barh", ax=plt.gca())
    plt.title("Rolling Backtest Maximum Drawdown")
    savefig("backtest_max_drawdown_bars.png")


# =============================================================================
# Robustness
# =============================================================================

def run_robustness(returns: pd.DataFrame, excess: pd.DataFrame) -> None:
    scenarios = []
    for tau in [0.025, 0.05, 0.10]:
        scenarios.append((f"tau_{tau}", tau, 1.0, COV_SHRINKAGE))
    for conf in [0.6, 1.0, 1.4]:
        scenarios.append((f"confidence_scale_{conf}", TAU, conf, COV_SHRINKAGE))
    for shrink in [0.0, 0.1, 0.3]:
        scenarios.append((f"shrinkage_{shrink}", TAU, 1.0, shrink))

    metric_rows, weight_rows, posterior_rows = [], [], []
    excess_clean = excess[TICKERS].dropna()
    for scenario, tau, conf, shrink in scenarios:
        for profile in BENCHMARK_PROFILES.columns:
            pi, mu_bl, sigma = bl_posterior(excess_clean, BENCHMARK_PROFILES[profile], profile, tau=tau, confidence_scale=conf, shrinkage=shrink)
            for ticker, value in mu_bl.items():
                posterior_rows.append({"scenario": scenario, "profile": profile, "ticker": ticker, "posterior_return": value})
            for method in ["MeanVariance", "MaxSharpe"]:
                w = optimise_portfolio(mu_bl, sigma, profile, method)
                metric_rows.append({"scenario": scenario, "profile": profile, "method": method, **portfolio_metrics(w, returns[TICKERS], mu_bl, sigma)})
                for ticker, value in w.items():
                    weight_rows.append({"scenario": scenario, "profile": profile, "method": method, "ticker": ticker, "weight": value})

    metrics = pd.DataFrame(metric_rows)
    weights = pd.DataFrame(weight_rows)
    posteriors = pd.DataFrame(posterior_rows)
    metrics.to_csv(TABLE_DIR / "robustness_metrics.csv", index=False)
    weights.to_csv(TABLE_DIR / "robustness_weights.csv", index=False)
    posteriors.to_csv(TABLE_DIR / "robustness_posterior_returns.csv", index=False)

    for prefix in ["tau", "confidence_scale", "shrinkage"]:
        df = metrics[(metrics["scenario"].str.startswith(prefix)) & (metrics["profile"].eq("Market_Balanced"))]
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=df, x="scenario", y="expected_sharpe", hue="method", marker="o")
        plt.title(f"Robustness of Expected Sharpe - {prefix}")
        plt.xticks(rotation=30, ha="right")
        savefig(f"robustness_expected_sharpe_{prefix}.png")

    mb_weights = weights[(weights["profile"].eq("Market_Balanced")) & (weights["method"].eq("MaxSharpe")) & (weights["scenario"].str.startswith("shrinkage"))]
    pivot = mb_weights.pivot_table(index="scenario", columns="ticker", values="weight")
    plt.figure(figsize=(9, 5))
    pivot.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="tab20")
    plt.title("Market Balanced Max-Sharpe Weights under Shrinkage")
    plt.legend(ncol=2, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig("robustness_weight_shift_market_balanced.png")


# =============================================================================
# Numerical validation checks
# =============================================================================

def run_validation_checks(excess: pd.DataFrame) -> None:
    """Run auditable invariants for BL inputs, optimisation, and risk attribution."""
    checks = []

    for scenario, expected_rows in [
        ("AbsoluteOnly", 2),
        ("RelativeOnly", 2),
        ("Combined", 4),
    ]:
        p, q, confidence = bl_views(TICKERS, view_scenario=scenario)
        checks.append(
            {
                "check": f"{scenario}_view_count",
                "passed": len(p) == expected_rows == len(q) == len(confidence),
                "detail": f"observed={len(p)}, expected={expected_rows}",
            }
        )

    clean = excess[TICKERS].dropna()
    for profile in BENCHMARK_PROFILES.columns:
        for scenario in VIEW_SCENARIOS:
            _, mu_bl, sigma = bl_posterior(
                clean,
                BENCHMARK_PROFILES[profile],
                profile,
                view_scenario=scenario,
            )
            checks.append(
                {
                    "check": f"{profile}_{scenario}_posterior_finite",
                    "passed": bool(np.isfinite(mu_bl).all()),
                    "detail": f"assets={len(mu_bl)}",
                }
            )
            for method in ["MeanVariance", "MaxSharpe"]:
                w = optimise_portfolio(mu_bl, sigma, profile, method)
                bounds = asset_caps(TICKERS)
                bounds_ok = all(
                    lo - 1e-7 <= w[ticker] <= hi + 1e-7
                    for ticker, (lo, hi) in zip(TICKERS, bounds)
                )
                bond_ok = (
                    w.reindex(["VAF.AX", "VBND.AX"]).sum()
                    >= MIN_BOND[profile] - 1e-7
                )
                checks.append(
                    {
                        "check": f"{profile}_{scenario}_{method}_feasible",
                        "passed": bool(
                            abs(w.sum() - 1.0) <= 1e-7
                            and bounds_ok
                            and bond_ok
                        ),
                        "detail": (
                            f"sum={w.sum():.10f}, "
                            f"bond={w.reindex(['VAF.AX', 'VBND.AX']).sum():.6f}"
                        ),
                    }
                )

    risk = pd.read_csv(TABLE_DIR / "bl_risk_contributions.csv")
    risk_sums = risk.groupby("portfolio")["risk_contribution"].sum()
    for portfolio, value in risk_sums.items():
        checks.append(
            {
                "check": f"{portfolio}_risk_contribution_sum",
                "passed": bool(abs(value - 1.0) <= 1e-6),
                "detail": f"sum={value:.10f}",
            }
        )

    result = pd.DataFrame(checks)
    result.to_csv(TABLE_DIR / "validation_checks.csv", index=False)
    if not result["passed"].all():
        failed = result.loc[~result["passed"], ["check", "detail"]]
        raise AssertionError(f"Numerical validation failed:\n{failed.to_string(index=False)}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("CQF PC single-file quantitative analysis started.")
    returns, excess, factors, rf, audusd = run_data_step()
    descriptive_analysis(returns, excess, factors)
    factor_regression(returns, audusd)
    run_black_litterman(excess, returns)
    run_backtest(returns, excess, rf, audusd)
    run_robustness(returns, excess)
    run_validation_checks(excess)
    print("\nDone.")
    print(f"Processed data: {PROCESSED_DIR.resolve()}")
    print(f"Output tables:  {TABLE_DIR.resolve()}")
    print(f"Output figures: {FIGURE_DIR.resolve()}")


if __name__ == "__main__":
    main()
