"""
segmentation.py
===============
K-Means Customer Segmentation using RFM features.

Business Rationale
------------------
Not all customers are equal. This module partitions the customer base into
distinct behavioural segments so that marketing, CRM, and retention teams
can execute targeted strategies:

  Segment → Strategy
  ──────────────────────────────────────────────────────────────────────────
  Champions          → VIP rewards, early access, referral programme
  Loyal Customers    → Upsell premium products, ask for reviews
  Potential Loyalists→ Onboarding journeys, loyalty programme invitation
  At Risk            → Win-back email, discount vouchers
  Lost/Dormant       → Last-resort reactivation or remove from lists

Technical Approach
------------------
- Dimensionality: 3 features (log_Recency, log_Frequency, log_Monetary)
- Algorithm: K-Means with k-selection via Elbow method + Silhouette score
- Scaling: StandardScaler (zero mean, unit variance) — mandatory for K-Means
- Validation: Silhouette analysis + business-logic review of cluster profiles
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
import logging
import joblib

logger = logging.getLogger(__name__)

FIGURES_PATH = Path(__file__).resolve().parent.parent / "outputs" / "figures"
MODELS_PATH  = Path(__file__).resolve().parent.parent / "outputs"
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

PALETTE = ["#1B4F72", "#2E86C1", "#AED6F1", "#E74C3C",
           "#F39C12", "#1E8449", "#8E44AD", "#17A589"]

SEGMENT_NAMES = {
    # Mapped after inspecting cluster centroids — labels assigned heuristically
    # based on R/F/M centroid ranking
}


def _save(fig, name):
    path = FIGURES_PATH / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Figure saved → %s", path)


# ────────────────────────────────────────────────────────────────────────────
# 1. Optimal K Selection
# ────────────────────────────────────────────────────────────────────────────

def find_optimal_k(X_scaled: np.ndarray,
                   k_range: range = range(2, 11)) -> tuple[int, dict]:
    """
    Elbow method (inertia) + Silhouette score to find optimal K.

    Why both metrics?
    - Inertia alone is monotonically decreasing → no clear optimum.
    - Silhouette measures cluster cohesion vs. separation (range -1 to +1).
      Higher = more distinct clusters. Using both together is standard practice.
    """
    inertias, silhouettes = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(list(k_range), inertias, "o-", color="#1B4F72", linewidth=2)
    ax1.set_title("Elbow Method (Inertia)", fontweight="bold")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (Within-cluster SSE)")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(list(k_range), silhouettes, "s-", color="#E74C3C", linewidth=2)
    ax2.set_title("Silhouette Score", fontweight="bold")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Highlight optimal k
    optimal_k = list(k_range)[np.argmax(silhouettes)]
    ax2.axvline(optimal_k, color="orange", linestyle=":", linewidth=2,
                label=f"Optimal k={optimal_k}")
    ax2.legend()

    fig.suptitle("K-Means: Cluster Count Selection", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "07_kmeans_k_selection")

    logger.info("Silhouette scores: %s",
                {k: round(s, 3) for k, s in zip(k_range, silhouettes)})
    logger.info("Optimal k by silhouette: %d", optimal_k)

    return optimal_k, {"inertias": inertias, "silhouettes": silhouettes,
                        "k_range": list(k_range)}


# ────────────────────────────────────────────────────────────────────────────
# 2. Fit K-Means
# ────────────────────────────────────────────────────────────────────────────

def fit_kmeans(rfm: pd.DataFrame, k: int) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    Scale features and fit K-Means with the chosen k.

    Returns augmented RFM DataFrame + fitted model + scaler.
    """
    features = ["log_Recency", "log_Frequency", "log_Monetary"]
    X = rfm[features].values

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
    km.fit(X_scaled)

    rfm = rfm.copy()
    rfm["Cluster"] = km.labels_

    sil = silhouette_score(X_scaled, km.labels_)
    logger.info("K-Means fitted: k=%d | Silhouette=%.4f | Inertia=%.1f",
                k, sil, km.inertia_)

    # Save model artefacts
    joblib.dump(km,     MODELS_PATH / "kmeans_model.pkl")
    joblib.dump(scaler, MODELS_PATH / "rfm_scaler.pkl")

    return rfm, km, scaler


# ────────────────────────────────────────────────────────────────────────────
# 3. Label Clusters with Business Names
# ────────────────────────────────────────────────────────────────────────────

def label_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Assign human-readable segment names based on centroid RFM profiles.

    Logic: Rank clusters by their mean RFM_Score (higher = better customers).
    The naming is done programmatically so it works regardless of which
    cluster ID K-Means assigns to each group.
    """
    rfm = rfm.copy()
    cluster_profiles = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()

    # Composite score: lower recency is better → negate it
    cluster_profiles["composite"] = (
        -cluster_profiles["Recency"].rank()
        + cluster_profiles["Frequency"].rank()
        + cluster_profiles["Monetary"].rank()
    )
    ranked = cluster_profiles["composite"].rank(ascending=False).astype(int)

    segment_map = {
        1: "Champions",
        2: "Loyal Customers",
        3: "Potential Loyalists",
        4: "At Risk",
        5: "Hibernating",
    }
    # Extend for more clusters
    for i in range(6, 20):
        segment_map[i] = f"Micro-Segment {i}"

    cluster_to_segment = {c: segment_map.get(r, f"Segment {r}")
                          for c, r in ranked.items()}
    rfm["Segment"] = rfm["Cluster"].map(cluster_to_segment)
    logger.info("Segment distribution:\n%s",
                rfm["Segment"].value_counts().to_string())
    return rfm


# ────────────────────────────────────────────────────────────────────────────
# 4. Visualisation
# ────────────────────────────────────────────────────────────────────────────

def plot_segment_profiles(rfm: pd.DataFrame) -> None:
    """
    Snake plot: normalised RFM profiles per segment.

    Why snake plot? It shows multi-dimensional profiles in 2D, allowing
    stakeholders to immediately compare segments on all three dimensions.
    """
    rfm_norm = rfm.copy()
    for col in ["Recency", "Frequency", "Monetary"]:
        rfm_norm[col] = (rfm_norm[col] - rfm_norm[col].min()) / \
                         (rfm_norm[col].max() - rfm_norm[col].min())

    seg_profile = rfm_norm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean()
    # Invert Recency for display (lower days = better)
    seg_profile["Recency"] = 1 - seg_profile["Recency"]
    seg_profile = seg_profile.rename(columns={"Recency": "Recency\n(inverted)"})

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (seg, row) in enumerate(seg_profile.iterrows()):
        ax.plot(seg_profile.columns, row.values,
                "o-", label=seg, color=PALETTE[i % len(PALETTE)], linewidth=2,
                markersize=8)

    ax.set_title("Customer Segment Profiles (Normalised RFM Snake Plot)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Normalised Score (0–1)", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_facecolor("#F8F9FA")
    fig.tight_layout()
    _save(fig, "08_segment_snake_plot")


def plot_segment_scatter(rfm: pd.DataFrame) -> None:
    """
    Scatter plot: Recency vs. Monetary, sized by Frequency, coloured by Segment.

    This is the 'money chart' that goes in executive decks — it instantly
    shows where Champion customers sit relative to At-Risk customers.
    """
    segments = rfm["Segment"].unique()
    color_map = {seg: PALETTE[i % len(PALETTE)] for i, seg in enumerate(segments)}

    fig, ax = plt.subplots(figsize=(11, 6))
    for seg in segments:
        sub = rfm[rfm["Segment"] == seg]
        ax.scatter(sub["Recency"], sub["Monetary"],
                   s=sub["Frequency"] * 8,
                   c=color_map[seg], label=seg,
                   alpha=0.65, edgecolors="white", linewidths=0.4)

    ax.set_xlabel("Recency (Days Since Last Purchase)", fontsize=10)
    ax.set_ylabel("Monetary Value (£ Total Spend)", fontsize=10)
    ax.set_title("Customer Segments: Recency vs. Monetary Value\n"
                 "(Bubble size = Purchase Frequency)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_facecolor("#F8F9FA")
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    fig.tight_layout()
    _save(fig, "09_segment_scatter")


def plot_segment_revenue_share(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Stacked bar chart showing each segment's share of customers and revenue.
    """
    seg_summary = rfm.groupby("Segment").agg(
        Customers = ("CustomerID", "count"),
        Revenue   = ("Monetary",   "sum"),
    ).reset_index()
    seg_summary["Customer_Pct"] = seg_summary["Customers"] / seg_summary["Customers"].sum() * 100
    seg_summary["Revenue_Pct"]  = seg_summary["Revenue"] / seg_summary["Revenue"].sum() * 100
    seg_summary = seg_summary.sort_values("Revenue_Pct", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(seg_summary))
    w = 0.35
    bars1 = ax.bar(x - w/2, seg_summary["Customer_Pct"],
                   width=w, label="% of Customers", color="#2E86C1", alpha=0.85)
    bars2 = ax.bar(x + w/2, seg_summary["Revenue_Pct"],
                   width=w, label="% of Revenue", color="#E74C3C", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(seg_summary["Segment"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Percentage (%)", fontsize=10)
    ax.set_title("Customer vs. Revenue Share by Segment",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_facecolor("#F8F9FA")
    fig.tight_layout()
    _save(fig, "10_segment_revenue_share")
    return seg_summary


# ────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────────────────────

def run_segmentation(rfm: pd.DataFrame) -> dict:
    """Full segmentation pipeline."""
    logger.info("=== Segmentation Pipeline Start ===")

    features = ["log_Recency", "log_Frequency", "log_Monetary"]
    X = rfm[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    optimal_k, k_stats = find_optimal_k(X_scaled, k_range=range(2, 8))

    # Use k=4 if silhouette is marginal — business interpretability wins
    chosen_k = optimal_k if optimal_k <= 6 else 4
    logger.info("Chosen k: %d", chosen_k)

    rfm, km, scaler = fit_kmeans(rfm, k=chosen_k)
    rfm = label_segments(rfm)

    plot_segment_profiles(rfm)
    plot_segment_scatter(rfm)
    seg_summary = plot_segment_revenue_share(rfm)

    logger.info("=== Segmentation Complete ===")
    return {
        "rfm_segmented":  rfm,
        "segment_summary": seg_summary,
        "k_stats":         k_stats,
        "chosen_k":        chosen_k,
        "km_model":        km,
    }
