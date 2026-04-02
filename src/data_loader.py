"""
data_loader.py
==============
Handles data acquisition and loading for the RetailPulse project.

Business Context:
-----------------
The UCI Online Retail II dataset contains transactional records from a UK-based
online gift retailer (Dec 2009 – Dec 2011). Each row is a single invoice line item.
This module generates a statistically realistic synthetic version of that dataset
so the project is fully self-contained and reproducible without manual downloads.

Why synthetic?
- Removes dependency on external URLs that may break
- Allows controlled injection of real-world messiness: nulls, duplicates, returns
- Fully reproducible for academic submission and portfolio review
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "online_retail.csv"


# ---------------------------------------------------------------------------
# Realistic product catalogue  (Stock Code → Description → Unit Price band)
# ---------------------------------------------------------------------------
PRODUCT_CATALOGUE = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER",  1.25, 3.95),
    ("71053",  "WHITE METAL LANTERN",                  3.39, 6.95),
    ("84406B", "CREAM CUPID HEARTS COAT HANGER",       1.65, 4.25),
    ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE",  3.39, 7.50),
    ("84029E", "RED WOOLLY HOTTIE WHITE HEART",        3.39, 7.50),
    ("22752",  "SET 7 BABUSHKA NESTING BOXES",         7.65, 14.95),
    ("21730",  "GLASS STAR FROSTED T-LIGHT HOLDER",   0.85, 2.95),
    ("22633",  "HAND WARMER UNION JACK",               1.85, 4.95),
    ("22632",  "HAND WARMER RED POLKA DOT",            1.85, 4.95),
    ("47566",  "PARTY BUNTING",                        4.95, 9.99),
    ("85099B", "JUMBO BAG RED RETROSPOT",              1.95, 5.25),
    ("21212",  "PACK OF 72 RETROSPOT CAKE CASES",      0.42, 1.95),
    ("22423",  "REGENCY CAKESTAND 3 TIER",            10.95, 24.95),
    ("47566B", "TEA TIME PARTY BUNTING",               4.95, 9.99),
    ("85099C", "JUMBO  BAG BAROQUE BLACK WHITE",       1.95, 5.25),
    ("20727",  "LUNCH BAG  BLACK SKULL",               1.65, 4.95),
    ("23343",  "JUMBO BAG OWLS",                       1.95, 5.25),
    ("23166",  "MEDIUM CERAMIC TOP STORAGE JAR",       1.95, 5.95),
    ("22386",  "JUMBO BAG PINK POLKADOT",              1.95, 5.25),
    ("85123B", "IVORY HANGING HEART T-LIGHT HOLDER",  1.25, 3.95),
    ("21977",  "PACK OF 60 DINOSAUR CAKE CASES",       0.42, 1.95),
    ("22551",  "PLASTERS IN TIN WOODLAND ANIMALS",    1.65, 4.25),
    ("22554",  "PLASTERS IN TIN SKULLS",               1.65, 4.25),
    ("22557",  "PLASTERS IN TIN CIRCUS PARADE",       1.65, 4.25),
    ("21232",  "STRAWBERRY CERAMIC TRINKET BOX",      1.65, 4.95),
    ("22469",  "HEART OF WICKER SMALL",               1.95, 5.95),
    ("22470",  "HEART OF WICKER LARGE",               2.95, 7.95),
    ("22554B", "PLASTERS IN TIN VINTAGE STAR",        1.65, 4.25),
    ("POST",   "POSTAGE",                             1.50, 4.50),
    ("M",      "Manual",                              0.00, 50.00),
]

COUNTRIES = {
    "United Kingdom": 0.87,
    "Germany":        0.03,
    "France":         0.02,
    "EIRE":           0.02,
    "Spain":          0.01,
    "Netherlands":    0.01,
    "Belgium":        0.01,
    "Australia":      0.005,
    "Switzerland":    0.005,
    "Portugal":       0.005,
    "Norway":         0.005,
    "Denmark":        0.005,
    "Sweden":         0.005,
    "Finland":        0.005,
    "USA":            0.005,
}


def generate_synthetic_retail_data(n_invoices: int = 22_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic online retail dataset.

    Design decisions
    ----------------
    - Invoice dates follow a seasonal pattern with Q4 peaks (gift retailer).
    - ~3% of invoices are credit notes (cancellations → negative quantities).
    - ~5% of rows have missing CustomerID (guest checkouts).
    - ~2% have null Descriptions (data entry gaps).
    - Quantity and UnitPrice have realistic right-skewed distributions.
    - A small fraction of prices are anomalously large (data errors / bulk orders).

    Parameters
    ----------
    n_invoices : int
        Approximate number of unique invoice numbers.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Raw, messy transactional dataset.
    """
    rng = np.random.default_rng(seed)
    logger.info("Generating synthetic retail dataset (%d invoices)…", n_invoices)

    # ── 1. Build invoice-level metadata ─────────────────────────────────────
    invoice_ids = np.arange(536_365, 536_365 + n_invoices)
    is_credit   = rng.random(n_invoices) < 0.03          # 3% cancellations
    invoice_nos = np.where(is_credit, "C" + invoice_ids.astype(str),
                           invoice_ids.astype(str))

    # Seasonal date distribution: peaks in Oct–Dec
    start = pd.Timestamp("2010-01-01")
    end   = pd.Timestamp("2011-12-31")
    total_days = (end - start).days

    # Day weights: higher in Q4
    day_range  = np.arange(total_days)
    day_of_year = day_range % 365
    seasonal_w = 1 + 1.5 * np.clip(np.sin((day_of_year - 80) / 365 * 2 * np.pi * -1), 0, 1)
    seasonal_w /= seasonal_w.sum()
    chosen_days = rng.choice(day_range, size=n_invoices, p=seasonal_w)
    invoice_dates = pd.to_datetime([start + pd.Timedelta(days=int(d)) for d in chosen_days])
    # Add realistic hour distribution (9am–6pm)
    hours   = rng.integers(9, 19, n_invoices)
    minutes = rng.integers(0, 60, n_invoices)
    invoice_datetimes = invoice_dates + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")

    # Countries
    country_names  = list(COUNTRIES.keys())
    country_probs  = list(COUNTRIES.values())
    country_probs_arr = np.array(country_probs)
    country_probs_arr = country_probs_arr / country_probs_arr.sum()
    inv_countries = rng.choice(country_names, size=n_invoices, p=country_probs_arr)

    # CustomerIDs: ~5% missing (guest checkouts)
    n_customers = int(n_invoices * 0.35)          # ~35% unique customer base
    customer_pool = np.arange(12_346, 12_346 + n_customers)
    inv_customers = rng.choice(customer_pool, size=n_invoices)
    guest_mask = rng.random(n_invoices) < 0.05
    inv_customers = inv_customers.astype(float)
    inv_customers[guest_mask] = np.nan

    # ── 2. Expand invoices into line items (1–8 items per invoice) ───────────
    items_per_invoice = rng.integers(1, 9, n_invoices)
    total_rows = items_per_invoice.sum()
    logger.info("Expanding to %d line-item rows…", total_rows)

    # Repeat invoice metadata
    inv_idx = np.repeat(np.arange(n_invoices), items_per_invoice)

    # Pick products
    prod_arr   = np.array(PRODUCT_CATALOGUE, dtype=object)
    prod_idx   = rng.integers(0, len(PRODUCT_CATALOGUE), total_rows)
    stock_codes   = prod_arr[prod_idx, 0]
    descriptions  = prod_arr[prod_idx, 1]
    price_low     = prod_arr[prod_idx, 2].astype(float)
    price_high    = prod_arr[prod_idx, 3].astype(float)

    # Quantities: log-normal (most orders small, occasional bulk)
    quantities = np.round(rng.lognormal(mean=2.0, sigma=0.9, size=total_rows)).astype(int)
    quantities = np.clip(quantities, 1, 800)

    # Prices: uniform within product band + 2% anomalous spikes
    unit_prices = rng.uniform(price_low, price_high)
    anomaly_mask = rng.random(total_rows) < 0.02
    unit_prices[anomaly_mask] *= rng.uniform(5, 20, anomaly_mask.sum())

    # Credit notes: flip quantity sign
    credit_mask_rows = np.isin(inv_idx, np.where(is_credit)[0])
    quantities[credit_mask_rows] *= -1

    # Inject ~2% null descriptions
    null_desc_mask = rng.random(total_rows) < 0.02
    descriptions = descriptions.astype(object)
    descriptions[null_desc_mask] = np.nan

    # ── 3. Assemble DataFrame ────────────────────────────────────────────────
    df = pd.DataFrame({
        "InvoiceNo":   invoice_nos[inv_idx],
        "StockCode":   stock_codes,
        "Description": descriptions,
        "Quantity":    quantities,
        "InvoiceDate": invoice_datetimes[inv_idx],
        "UnitPrice":   np.round(unit_prices, 2),
        "CustomerID":  inv_customers[inv_idx],
        "Country":     inv_countries[inv_idx],
    })

    # Shuffle rows (realistic — not ordered by invoice)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info("Dataset shape: %s", df.shape)
    return df


def load_data(force_regenerate: bool = False) -> pd.DataFrame:
    """
    Load raw data from disk or generate if not present.

    Parameters
    ----------
    force_regenerate : bool
        If True, regenerate even if cached CSV exists.

    Returns
    -------
    pd.DataFrame
    """
    if RAW_DATA_PATH.exists() and not force_regenerate:
        logger.info("Loading cached data from %s", RAW_DATA_PATH)
        return pd.read_csv(RAW_DATA_PATH, parse_dates=["InvoiceDate"])

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_retail_data()
    df.to_csv(RAW_DATA_PATH, index=False)
    logger.info("Raw data saved → %s", RAW_DATA_PATH)
    return df


if __name__ == "__main__":
    df = load_data(force_regenerate=True)
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['InvoiceDate'].min()} → {df['InvoiceDate'].max()}")
    print(f"Unique customers: {df['CustomerID'].nunique()}")
    print(f"Unique invoices:  {df['InvoiceNo'].nunique()}")
