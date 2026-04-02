"""
preprocessing.py
================
Data cleaning, validation, and preprocessing pipeline.

Business Rationale
------------------
Raw transactional data from any retail system will contain:
  - Credit notes (returns) that must be isolated from forward sales
  - Guest checkouts (no CustomerID) unusable for RFM segmentation
  - Anomalous prices (data entry errors / free gifts)
  - Duplicate rows from ETL glitches

Cleaning these is not cosmetic — it directly affects revenue figures,
customer counts, and any model trained downstream.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "retail_clean.csv"


# ────────────────────────────────────────────────────────────────────────────
# 1. Data Validation Report
# ────────────────────────────────────────────────────────────────────────────

def generate_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a column-level data quality summary.

    Why: Before cleaning anything, we document the 'before' state.
    This is mandatory in professional data analytics — it forms part
    of the Data Quality Assessment section of any analytical report.
    """
    report = pd.DataFrame({
        "dtype":          df.dtypes,
        "null_count":     df.isnull().sum(),
        "null_pct":       (df.isnull().mean() * 100).round(2),
        "unique_values":  df.nunique(),
        "sample_min":     df.apply(lambda c: c.min() if pd.api.types.is_numeric_dtype(c) else c.dropna().iloc[0] if len(c.dropna()) > 0 else np.nan),
        "sample_max":     df.apply(lambda c: c.max() if pd.api.types.is_numeric_dtype(c) else c.dropna().iloc[-1] if len(c.dropna()) > 0 else np.nan),
    })
    return report


# ────────────────────────────────────────────────────────────────────────────
# 2. Core Cleaning Steps
# ────────────────────────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop fully duplicated rows.

    Why: ETL pipelines often double-load records. Leaving duplicates
    inflates revenue and transaction counts.
    """
    n_before = len(df)
    df = df.drop_duplicates()
    logger.info("Duplicates removed: %d rows → %d rows (-%d)",
                n_before, len(df), n_before - len(df))
    return df


def extract_cancellations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate forward invoices from credit notes (returns).

    Why: Mixing returns with sales skews:
      - Revenue totals downward
      - Customer purchase frequency upward (a return != a purchase)
      - Basket size calculations

    Business use: The returns DataFrame is separately analysed to
    compute return rate KPIs.
    """
    returns_mask = df["InvoiceNo"].astype(str).str.startswith("C")
    returns_df   = df[returns_mask].copy()
    sales_df     = df[~returns_mask].copy()
    logger.info("Sales rows: %d | Return rows: %d", len(sales_df), len(returns_df))
    return sales_df, returns_df


def clean_quantities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with non-positive quantities in the sales DataFrame.

    Why: A Quantity ≤ 0 in a forward invoice indicates a data entry error
    or a mis-routed credit note. These rows cannot represent a real sale.
    """
    n_before = len(df)
    df = df[df["Quantity"] > 0].copy()
    logger.info("Zero/negative quantities removed: %d → %d (-%d)",
                n_before, len(df), n_before - len(df))
    return df


def clean_prices(df: pd.DataFrame,
                 min_price: float = 0.01,
                 price_upper_iqr_multiplier: float = 5.0) -> pd.DataFrame:
    """
    Remove free items (price = 0) and cap anomalous price outliers.

    Why:
    - Zero-price rows represent samples/gifts and distort average order value.
    - Price outliers (bulk industrial orders) are legitimate but can be capped
      to prevent them dominating revenue-per-customer statistics. We use the
      IQR method — a robust, distribution-agnostic approach.

    Strategy: CAP (winsorise) rather than DROP. Dropping loses transaction count.
    """
    # Remove zero/negative prices
    n_before = len(df)
    df = df[df["UnitPrice"] >= min_price].copy()
    logger.info("Zero/negative price rows removed: -%d", n_before - len(df))

    # Winsorise upper outliers using Tukey's IQR fence
    Q1, Q3 = df["UnitPrice"].quantile([0.25, 0.75])
    IQR     = Q3 - Q1
    upper   = Q3 + price_upper_iqr_multiplier * IQR
    n_capped = (df["UnitPrice"] > upper).sum()
    df["UnitPrice"] = df["UnitPrice"].clip(upper=upper)
    logger.info("UnitPrice capped at %.2f (IQR×%.1f) → %d rows affected",
                upper, price_upper_iqr_multiplier, n_capped)
    return df


def handle_missing_customers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing CustomerID values.

    Why: For RFM (Recency–Frequency–Monetary) segmentation, we MUST know
    who placed the order. Guest checkouts (null CustomerID) cannot be
    attributed to any customer. We retain them for revenue totals but
    flag them separately. For segmentation, we drop them.

    Strategy: Add a flag column, then return a 'customer-ready' copy.
    """
    df = df.copy()
    df["is_guest"] = df["CustomerID"].isnull().astype(int)
    n_guest = df["is_guest"].sum()
    logger.info("Guest checkout rows (null CustomerID): %d (%.1f%%)",
                n_guest, 100 * n_guest / len(df))

    df_identified = df[df["CustomerID"].notna()].copy()
    df_identified["CustomerID"] = df_identified["CustomerID"].astype(int)
    logger.info("Rows with identified customers: %d", len(df_identified))
    return df, df_identified


def fill_missing_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing product descriptions from the most common description
    for that StockCode.

    Why: Description is not used in modelling, but it IS used in EDA
    storytelling (top products by revenue). Leaving nulls here means
    those products appear nameless in charts.
    """
    df = df.copy()
    # Build a lookup: StockCode → most frequent non-null description
    desc_lookup = (
        df.dropna(subset=["Description"])
          .groupby("StockCode")["Description"]
          .agg(lambda x: x.mode()[0] if len(x) > 0 else np.nan)
    )
    null_mask = df["Description"].isnull()
    df.loc[null_mask, "Description"] = df.loc[null_mask, "StockCode"].map(desc_lookup)
    remaining = df["Description"].isnull().sum()
    logger.info("Description nulls filled. Remaining nulls: %d", remaining)
    return df


# ────────────────────────────────────────────────────────────────────────────
# 3. Feature Creation (Revenue Column)
# ────────────────────────────────────────────────────────────────────────────

def add_revenue_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the LineRevenue column.

    Why: Every downstream analysis (top products, RFM monetary value,
    time-series) needs revenue. Computing it once here ensures consistency.
    """
    df = df.copy()
    df["LineRevenue"] = df["Quantity"] * df["UnitPrice"]
    return df


# ────────────────────────────────────────────────────────────────────────────
# 4. Orchestration
# ────────────────────────────────────────────────────────────────────────────

def run_preprocessing_pipeline(df_raw: pd.DataFrame) -> dict:
    """
    Execute the full cleaning pipeline and return a dictionary of DataFrames.

    Returns
    -------
    dict with keys:
        "raw"           – Original input
        "quality_report"– Data quality summary before cleaning
        "sales"         – Clean forward sales (all rows, incl. guests)
        "sales_customers"– Sales with identified customers only
        "returns"       – Credit notes / cancellations
    """
    logger.info("=== Preprocessing Pipeline Start ===")
    logger.info("Input shape: %s", df_raw.shape)

    quality_report = generate_quality_report(df_raw)

    df = remove_duplicates(df_raw)
    sales, returns = extract_cancellations(df)
    sales = clean_quantities(sales)
    sales = clean_prices(sales)
    sales = fill_missing_descriptions(sales)
    sales, sales_customers = handle_missing_customers(sales)
    sales = add_revenue_column(sales)
    sales_customers = add_revenue_column(sales_customers)

    # Save processed data
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    sales_customers.to_csv(PROCESSED_PATH, index=False)
    logger.info("Processed data saved → %s", PROCESSED_PATH)
    logger.info("=== Preprocessing Pipeline Complete ===")
    logger.info("Clean sales (w/ customers): %s", sales_customers.shape)

    return {
        "raw":              df_raw,
        "quality_report":   quality_report,
        "sales":            sales,
        "sales_customers":  sales_customers,
        "returns":          returns,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_loader import load_data

    raw = load_data()
    result = run_preprocessing_pipeline(raw)
    print("\n── Data Quality Report ──")
    print(result["quality_report"])
    print(f"\nClean dataset: {result['sales_customers'].shape}")
    print(result["sales_customers"].head())
