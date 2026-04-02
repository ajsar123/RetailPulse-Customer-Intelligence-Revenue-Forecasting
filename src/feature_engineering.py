"""
feature_engineering.py
=======================
Derives analytical features used in segmentation and forecasting.

Two Feature Sets
----------------
1. RFM (Recency, Frequency, Monetary) — customer-level
   The industry-standard framework for quantifying customer value.
   Used as input to K-Means clustering.

2. Monthly Time Series — aggregate level
   Prepared for Prophet/ARIMA forecasting.

Why RFM?
--------
RFM was popularised by direct mail marketers in the 1960s and remains
the most interpretable customer value framework. Unlike black-box models,
stakeholders immediately understand: "Segment A customers bought recently,
often, and spend a lot — these are our Champions."
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# 1. RFM Feature Engineering
# ────────────────────────────────────────────────────────────────────────────

def build_rfm_table(df: pd.DataFrame,
                    snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Compute Recency, Frequency, and Monetary value per customer.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned sales DataFrame with identified CustomerIDs.
    snapshot_date : pd.Timestamp, optional
        The 'today' reference for recency. Defaults to max InvoiceDate + 1 day.
        Using a fixed snapshot ensures reproducible RFM scores.

    Returns
    -------
    pd.DataFrame with columns:
        CustomerID, Recency (days), Frequency (invoices), Monetary (£ total)
    """
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    logger.info("RFM snapshot date: %s", snapshot_date.date())

    rfm = (df.groupby("CustomerID")
             .agg(
                 Recency   = ("InvoiceDate",  lambda x: (snapshot_date - x.max()).days),
                 Frequency = ("InvoiceNo",    "nunique"),
                 Monetary  = ("LineRevenue",  "sum"),
             )
             .reset_index())

    # Sanity checks
    assert (rfm["Recency"]   >= 0).all(),  "Negative recency values found"
    assert (rfm["Frequency"] >= 1).all(),  "Zero frequency values found"
    assert (rfm["Monetary"]  >  0).all(),  "Non-positive monetary values found"

    logger.info("RFM table shape: %s", rfm.shape)
    logger.info("Recency range:   %d – %d days", rfm["Recency"].min(), rfm["Recency"].max())
    logger.info("Frequency range: %d – %d invoices", rfm["Frequency"].min(), rfm["Frequency"].max())
    logger.info("Monetary range:  £%.2f – £%.2f", rfm["Monetary"].min(), rfm["Monetary"].max())
    return rfm


def add_rfm_scores(rfm: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """
    Assign 1–5 quintile scores for each RFM dimension.

    Scoring logic:
    - Recency:   LOWER days = HIGHER score (bought recently = valuable)
    - Frequency: HIGHER count = HIGHER score
    - Monetary:  HIGHER spend = HIGHER score

    Why quintiles? They rank customers relative to each other (not absolute
    thresholds), making the scores stable across different time periods.
    """
    rfm = rfm.copy()
    labels = list(range(1, n_quantiles + 1))

    # Recency: reverse order (low days → high score)
    rfm["R_Score"] = pd.qcut(rfm["Recency"], q=n_quantiles,
                              labels=labels[::-1], duplicates="drop")
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"),
                              q=n_quantiles, labels=labels, duplicates="drop")
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"),
                              q=n_quantiles, labels=labels, duplicates="drop")

    rfm[["R_Score", "F_Score", "M_Score"]] = \
        rfm[["R_Score", "F_Score", "M_Score"]].astype(int)

    rfm["RFM_Score"]  = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]
    rfm["RFM_Segment"] = rfm["R_Score"].astype(str) + \
                          rfm["F_Score"].astype(str) + \
                          rfm["M_Score"].astype(str)
    return rfm


def log_transform_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p transform to Recency, Frequency, Monetary before clustering.

    Why: All three RFM variables are right-skewed (power-law distributed).
    K-Means uses Euclidean distance — skewed distributions give disproportionate
    weight to outliers (high spenders dominate the distance metric).
    Log1p compresses the right tail while preserving rank ordering.
    """
    rfm = rfm.copy()
    for col in ["Recency", "Frequency", "Monetary"]:
        rfm[f"log_{col}"] = np.log1p(rfm[col])
    return rfm


# ────────────────────────────────────────────────────────────────────────────
# 2. Time Series Feature Engineering
# ────────────────────────────────────────────────────────────────────────────

def build_monthly_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily transactions into a monthly revenue time series
    suitable for Prophet forecasting.

    Additional features derived:
    - is_december: Christmas trading flag (strong predictor)
    - is_q4: October–December flag
    - num_active_customers: monthly active customer count (leading indicator)
    - avg_order_value: basket size trend

    Returns
    -------
    pd.DataFrame with columns: ds (date), y (revenue), and regressors.
    """
    df = df.copy()
    df["Month"] = df["InvoiceDate"].dt.to_period("M")

    monthly = df.groupby("Month").agg(
        y                    = ("LineRevenue",  "sum"),
        num_transactions     = ("InvoiceNo",    "nunique"),
        num_active_customers = ("CustomerID",   "nunique"),
        avg_order_value      = ("LineRevenue",  "mean"),
    ).reset_index()

    monthly["ds"]           = monthly["Month"].dt.to_timestamp()
    monthly["is_december"]  = (monthly["ds"].dt.month == 12).astype(int)
    monthly["is_q4"]        = (monthly["ds"].dt.month >= 10).astype(int)
    monthly["month_num"]    = monthly["ds"].dt.month

    monthly = monthly.sort_values("ds").reset_index(drop=True)
    return monthly[["ds", "y", "num_transactions", "num_active_customers",
                    "avg_order_value", "is_december", "is_q4", "month_num"]]


def build_daily_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to daily for anomaly detection (statsmodels STL decomposition).
    """
    df = df.copy()
    df["Date"] = df["InvoiceDate"].dt.date
    daily = (df.groupby("Date")["LineRevenue"]
               .sum()
               .reset_index())
    daily.columns = ["ds", "y"]
    daily["ds"] = pd.to_datetime(daily["ds"])
    daily = daily.sort_values("ds").reset_index(drop=True)
    return daily


# ────────────────────────────────────────────────────────────────────────────
# 3. Additional Derived Features
# ────────────────────────────────────────────────────────────────────────────

def add_customer_tenure(df: pd.DataFrame, rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Add customer tenure (days since first purchase) to RFM table.

    Why: Tenure is a strong predictor of churn. A long-tenured customer
    with recent purchase is a Loyal; a long-tenured customer who hasn't
    bought in 6+ months is a Lapsing customer requiring win-back campaigns.
    """
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    first_purchase = (df.groupby("CustomerID")["InvoiceDate"]
                        .min()
                        .reset_index()
                        .rename(columns={"InvoiceDate": "FirstPurchase"}))
    rfm = rfm.merge(first_purchase, on="CustomerID", how="left")
    rfm["Tenure_Days"] = (snapshot - rfm["FirstPurchase"]).dt.days
    return rfm


# ────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────────────────────

def run_feature_engineering(df: pd.DataFrame) -> dict:
    """
    Build all feature tables and return as dictionary.
    """
    logger.info("=== Feature Engineering Start ===")

    rfm = build_rfm_table(df)
    rfm = add_rfm_scores(rfm)
    rfm = log_transform_rfm(rfm)
    rfm = add_customer_tenure(df, rfm)

    monthly_ts = build_monthly_timeseries(df)
    daily_ts   = build_daily_timeseries(df)

    logger.info("RFM table:     %s rows", len(rfm))
    logger.info("Monthly TS:    %s months", len(monthly_ts))
    logger.info("Daily TS:      %s days", len(daily_ts))
    logger.info("=== Feature Engineering Complete ===")

    return {
        "rfm":        rfm,
        "monthly_ts": monthly_ts,
        "daily_ts":   daily_ts,
    }
