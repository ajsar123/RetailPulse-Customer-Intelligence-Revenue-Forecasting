"""
eda.py
======
Exploratory Data Analysis (EDA) for the RetailPulse project.

Business Rationale
------------------
EDA is not "playing with data" — it is structured hypothesis generation.
Each chart and statistic here answers a specific business question that
a Head of E-Commerce or CFO would ask:

  Q1. How is revenue trending over time? Are there seasonality patterns?
  Q2. Which products drive the most revenue (Pareto analysis)?
  Q3. Which countries generate the most value?
  Q4. What does order behaviour look like (basket size, frequency)?
  Q5. Are there anomalies in the return rate?

Each finding feeds directly into the strategic recommendations.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for file-based rendering
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

FIGURES_PATH = Path(__file__).resolve().parent.parent / "outputs" / "figures"
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

# ── Consistent visual style ──────────────────────────────────────────────────
PALETTE   = ["#1B4F72", "#2E86C1", "#AED6F1", "#E74C3C", "#F39C12", "#1E8449"]
BG_COLOR  = "#F8F9FA"
GRID_COLOR = "#DEE2E6"

def _style_axis(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.6, alpha=0.8)
    ax.set_facecolor(BG_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIGURES_PATH / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Figure saved → %s", path)
    return path


# ────────────────────────────────────────────────────────────────────────────
# EDA Function 1: Revenue Time Series
# ────────────────────────────────────────────────────────────────────────────

def plot_revenue_timeseries(df: pd.DataFrame) -> dict:
    """
    Plot monthly revenue with a 3-month rolling average.

    Business Question: Is the business growing? Where are the peaks?
    Statistical Insight: Seasonality visible in Q4 (Christmas gifts).
    """
    df = df.copy()
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")
    monthly = (df.groupby("YearMonth")["LineRevenue"]
                 .sum()
                 .reset_index())
    monthly["YearMonth_dt"] = monthly["YearMonth"].dt.to_timestamp()
    monthly["Rolling3M"]    = monthly["LineRevenue"].rolling(3, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(monthly["YearMonth_dt"], monthly["LineRevenue"],
                    alpha=0.25, color=PALETTE[1])
    ax.plot(monthly["YearMonth_dt"], monthly["LineRevenue"],
            color=PALETTE[0], linewidth=2, label="Monthly Revenue")
    ax.plot(monthly["YearMonth_dt"], monthly["Rolling3M"],
            color=PALETTE[3], linewidth=2.5, linestyle="--",
            label="3-Month Rolling Avg")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=35)
    ax.legend(fontsize=9)
    _style_axis(ax, "Monthly Revenue Trend (2010–2011)",
                xlabel="Month", ylabel="Revenue (£)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"£{x:,.0f}"))
    fig.tight_layout()
    _save(fig, "01_revenue_timeseries")

    stats = {
        "peak_month":      str(monthly.loc[monthly["LineRevenue"].idxmax(), "YearMonth"]),
        "peak_revenue":    monthly["LineRevenue"].max(),
        "avg_monthly_rev": monthly["LineRevenue"].mean(),
        "total_revenue":   monthly["LineRevenue"].sum(),
        "yoy_growth":      None,  # populated below
    }
    # YoY growth if we have 2 full years
    yr = df.groupby(df["InvoiceDate"].dt.year)["LineRevenue"].sum()
    if len(yr) >= 2:
        years = sorted(yr.index)
        stats["yoy_growth"] = round(
            100 * (yr[years[-1]] - yr[years[-2]]) / yr[years[-2]], 1)

    return stats


# ────────────────────────────────────────────────────────────────────────────
# EDA Function 2: Pareto Analysis (Top Products)
# ────────────────────────────────────────────────────────────────────────────

def plot_pareto_products(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Bar chart of top-N products by revenue with cumulative % line (Pareto).

    Business Question: Do 20% of SKUs drive 80% of revenue? (Pareto Principle)
    Action: Inventory prioritisation & promotional focus.
    """
    prod_rev = (df.groupby("Description")["LineRevenue"]
                  .sum()
                  .sort_values(ascending=False)
                  .head(top_n)
                  .reset_index())
    prod_rev.columns = ["Product", "Revenue"]
    prod_rev["CumPct"] = prod_rev["Revenue"].cumsum() / df["LineRevenue"].sum() * 100

    fig, ax1 = plt.subplots(figsize=(13, 5))
    bars = ax1.bar(prod_rev["Product"], prod_rev["Revenue"],
                   color=PALETTE[1], edgecolor="white", linewidth=0.5)
    ax1.set_xticklabels(prod_rev["Product"], rotation=40, ha="right", fontsize=8)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"£{x:,.0f}"))
    _style_axis(ax1, f"Top {top_n} Products by Revenue (Pareto Analysis)",
                ylabel="Revenue (£)")

    ax2 = ax1.twinx()
    ax2.plot(prod_rev["Product"], prod_rev["CumPct"],
             color=PALETTE[3], marker="o", markersize=5,
             linewidth=2, label="Cumulative %")
    ax2.axhline(80, color="grey", linestyle=":", linewidth=1)
    ax2.set_ylabel("Cumulative Revenue %", fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.tick_params(labelsize=9)
    ax2.legend(loc="center right", fontsize=9)

    fig.tight_layout()
    _save(fig, "02_pareto_products")

    pareto_cutoff = prod_rev[prod_rev["CumPct"] <= 80].shape[0]
    logger.info("Pareto: top %d SKUs account for ~80%% of revenue", pareto_cutoff + 1)
    return prod_rev


# ────────────────────────────────────────────────────────────────────────────
# EDA Function 3: Revenue by Country (Treemap-style horizontal bar)
# ────────────────────────────────────────────────────────────────────────────

def plot_revenue_by_country(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Horizontal bar chart of revenue by country.

    Business Question: Is international expansion worth pursuing?
    Which markets are the largest outside the UK?
    """
    country_rev = (df.groupby("Country")["LineRevenue"]
                     .sum()
                     .sort_values(ascending=False)
                     .head(top_n)
                     .reset_index())
    country_rev.columns = ["Country", "Revenue"]

    colors = [PALETTE[0] if c == "United Kingdom" else PALETTE[1]
              for c in country_rev["Country"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(country_rev["Country"][::-1], country_rev["Revenue"][::-1],
            color=colors[::-1], edgecolor="white")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"£{x:,.0f}"))
    _style_axis(ax, f"Top {top_n} Countries by Revenue",
                xlabel="Revenue (£)")
    fig.tight_layout()
    _save(fig, "03_revenue_by_country")
    return country_rev


# ────────────────────────────────────────────────────────────────────────────
# EDA Function 4: Order-level distributions
# ────────────────────────────────────────────────────────────────────────────

def plot_order_distributions(df: pd.DataFrame) -> dict:
    """
    Histograms of basket size (quantity), basket value, and items per invoice.

    Business Question: What does a typical order look like?
    This informs free-shipping thresholds and upsell strategy.
    """
    invoice_agg = df.groupby("InvoiceNo").agg(
        TotalRevenue=("LineRevenue", "sum"),
        TotalQty=("Quantity", "sum"),
        NumItems=("StockCode", "nunique"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, col, label, color in zip(
        axes,
        ["TotalRevenue", "TotalQty", "NumItems"],
        ["Basket Value (£)", "Total Quantity", "Unique SKUs per Order"],
        [PALETTE[0], PALETTE[1], PALETTE[4]],
    ):
        data = invoice_agg[col].clip(
            upper=invoice_agg[col].quantile(0.99))  # trim extreme tail for viz
        ax.hist(data, bins=40, color=color, edgecolor="white", linewidth=0.4)
        ax.axvline(data.median(), color=PALETTE[3], linestyle="--",
                   linewidth=1.5, label=f"Median: {data.median():.1f}")
        ax.legend(fontsize=8)
        _style_axis(ax, label, xlabel=label, ylabel="Frequency")

    fig.suptitle("Order-Level Distributions (99th percentile clipped)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "04_order_distributions")

    return {
        "median_basket_value":  invoice_agg["TotalRevenue"].median(),
        "mean_basket_value":    invoice_agg["TotalRevenue"].mean(),
        "median_basket_qty":    invoice_agg["TotalQty"].median(),
        "median_unique_skus":   invoice_agg["NumItems"].median(),
    }


# ────────────────────────────────────────────────────────────────────────────
# EDA Function 5: Day-of-Week & Hour Heatmap
# ────────────────────────────────────────────────────────────────────────────

def plot_temporal_heatmap(df: pd.DataFrame) -> None:
    """
    Heatmap of transaction count by day-of-week × hour-of-day.

    Business Question: When are customers most active?
    Action: Schedule email campaigns and server capacity accordingly.
    """
    df = df.copy()
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    df["Hour"]      = df["InvoiceDate"].dt.hour

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
    pivot = (df.groupby(["DayOfWeek", "Hour"])["InvoiceNo"]
               .count()
               .unstack(fill_value=0)
               .reindex(day_order))

    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(pivot, cmap="Blues", linewidths=0.3, linecolor="white",
                cbar_kws={"label": "Transactions"}, ax=ax, annot=False)
    ax.set_title("Transaction Volume by Day & Hour",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Hour of Day", fontsize=10)
    ax.set_ylabel("", fontsize=10)
    fig.tight_layout()
    _save(fig, "05_temporal_heatmap")


# ────────────────────────────────────────────────────────────────────────────
# EDA Function 6: Return Rate Analysis
# ────────────────────────────────────────────────────────────────────────────

def plot_return_rate(sales_df: pd.DataFrame, returns_df: pd.DataFrame) -> dict:
    """
    Monthly return rate (returns / gross sales) bar chart.

    Business Question: Is the return rate improving over time?
    A rising return rate signals product quality issues or misaligned
    marketing (attracting wrong customers).
    """
    # Compute absolute return value (quantities are negative in returns)
    returns_df = returns_df.copy()
    if "LineRevenue" not in returns_df.columns:
        returns_df["LineRevenue"] = returns_df["Quantity"].abs() * returns_df["UnitPrice"]

    monthly_sales   = (sales_df.groupby(sales_df["InvoiceDate"].dt.to_period("M"))
                               ["LineRevenue"].sum())
    monthly_returns = (returns_df.groupby(returns_df["InvoiceDate"].dt.to_period("M"))
                                 ["LineRevenue"].sum())
    rate = (monthly_returns / monthly_sales * 100).dropna().reset_index()
    rate.columns = ["Month", "ReturnRate"]
    rate["Month_dt"] = rate["Month"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(rate["Month_dt"], rate["ReturnRate"], width=20,
           color=PALETTE[3], alpha=0.75, label="Return Rate %")
    ax.axhline(rate["ReturnRate"].mean(), color=PALETTE[0],
               linestyle="--", linewidth=1.5,
               label=f"Avg: {rate['ReturnRate'].mean():.1f}%")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=35)
    ax.set_ylabel("Return Rate (%)", fontsize=10)
    ax.legend(fontsize=9)
    _style_axis(ax, "Monthly Return Rate (%)", ylabel="Return Rate (%)")
    fig.tight_layout()
    _save(fig, "06_return_rate")

    return {
        "avg_return_rate": round(rate["ReturnRate"].mean(), 2),
        "max_return_rate": round(rate["ReturnRate"].max(), 2),
    }


# ────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────────────────────

def run_eda(sales_df: pd.DataFrame, returns_df: pd.DataFrame) -> dict:
    """Run all EDA steps and return a consolidated stats dictionary."""
    logger.info("=== EDA Pipeline Start ===")
    results = {}
    results["revenue_ts"]      = plot_revenue_timeseries(sales_df)
    results["pareto"]          = plot_pareto_products(sales_df)
    results["country"]         = plot_revenue_by_country(sales_df)
    results["order_dist"]      = plot_order_distributions(sales_df)
    plot_temporal_heatmap(sales_df)
    results["return_rate"]     = plot_return_rate(sales_df, returns_df)
    logger.info("=== EDA Complete — %d figures saved ===", 6)
    return results
