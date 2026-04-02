# RetailPulse: Customer Intelligence & Revenue Forecasting

> **End-to-end data analytics project** | UK E-Commerce | Python · scikit-learn · statsmodels · XGBoost

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

---

## Business Problem

A UK-based online gift retailer has two strategic questions:

1. **Which customers are most valuable, and which are at risk of churning?** — Without clear segmentation, the marketing team sends identical campaigns to Champions and Lost customers, wasting budget and irritating loyal buyers.

2. **What will revenue look like over the next quarter?** — Without a defensible forecast, the operations team over- or under-stocks, destroying margin.

This project delivers both: **behavioural customer segmentation** and a **statistically validated revenue forecast**, packaged into an automated, reproducible Python pipeline.

---

## Results at a Glance

| Metric | Value |
|---|---|
| Customers analysed | ~7,700 |
| Segments identified | 4–5 (K-Means, Silhouette-optimised) |
| Forecast MAPE | < 12% |
| Churn AUC-ROC | > 0.78 |
| Anomalous days detected | ~31 |
| Figures generated | 14 |

---

## Project Architecture

```
retailpulse/
│
├── main.py                     # ← Pipeline entry point
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── data_loader.py          # Data generation & loading
│   ├── preprocessing.py        # Cleaning, validation, QA
│   ├── eda.py                  # 6 EDA visualisations
│   ├── feature_engineering.py  # RFM + time series features
│   ├── segmentation.py         # K-Means customer clustering
│   └── forecasting.py          # SARIMA + GBM + STL + hypothesis test
│
├── data/
│   ├── raw/                    # Auto-generated on first run
│   └── processed/              # Cleaned dataset
│
└── outputs/
    ├── figures/                # 14 publication-quality charts
    ├── reports/                # Executive summary (text)
    ├── kmeans_model.pkl
    ├── churn_model.pkl
    └── rfm_scaler.pkl
```

---

## Analytical Pipeline

```
Data Generation → Preprocessing → EDA → Feature Engineering
      ↓
  RFM Table → K-Means Segmentation → Segment Labelling
      ↓
  Monthly TS → SARIMA Forecast → 3-Month Revenue Outlook
      ↓
  Daily TS  → STL Decomposition → Anomaly Flags
      ↓
  RFM + Labels → GBM Churn Model → Churn Probability Scores
      ↓
  Mann-Whitney U → UK vs. International Basket Hypothesis Test
      ↓
      Executive Report (outputs/reports/executive_report.txt)
```

---

## Techniques Implemented

| Technique | Library | Business Purpose |
|---|---|---|
| Data Quality Assessment | pandas | Pre-cleaning audit |
| IQR Winsorisation | numpy | Price outlier handling |
| RFM Scoring | pandas | Customer value quantification |
| K-Means Clustering | scikit-learn | Behavioural segmentation |
| Silhouette Analysis | scikit-learn | Optimal cluster selection |
| SARIMA(1,1,1)(1,1,1,12) | statsmodels | Revenue forecasting |
| STL Decomposition | statsmodels | Anomaly detection |
| Gradient Boosting | scikit-learn | Churn risk prediction |
| Mann-Whitney U Test | scipy | Market comparison |
| Pareto Analysis | pandas/matplotlib | SKU prioritisation |

---

## Visualisations

| # | Chart | Business Question |
|---|---|---|
| 01 | Monthly Revenue Trend | Is the business growing? |
| 02 | Pareto (Top 15 SKUs) | Do 20% of products drive 80% of revenue? |
| 03 | Revenue by Country | Which markets matter most? |
| 04 | Order Distributions | What does a typical order look like? |
| 05 | Transaction Heatmap | When are customers active? |
| 06 | Return Rate | Is the return rate improving? |
| 07 | K Selection (Elbow + Silhouette) | How many segments exist? |
| 08 | Segment Snake Plot | How do segments differ on RFM? |
| 09 | Segment Scatter | Where do Champions sit vs. At-Risk? |
| 10 | Segment Revenue Share | Which segments drive the most revenue? |
| 11 | SARIMA Forecast | What will revenue look like next quarter? |
| 12 | STL Anomaly Detection | Which days were operationally abnormal? |
| 13 | ROC Curve + Feature Importance | How accurate is churn prediction? |
| 14 | Violin: UK vs. International | Do markets differ in basket value? |

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/retailpulse.git
cd retailpulse

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline
python main.py

# 5. Force-regenerate data (optional)
python main.py --regenerate
```

**Outputs appear in `outputs/`** — no manual data download required.

---

## Key Business Insights

### 1. Seasonal Concentration Risk
Q4 (Oct–Dec) contributes a disproportionate share of annual revenue. The business is exposed to single-season risk — a logistics failure in November would be catastrophic. **Recommendation: Build safety stock and diversify revenue into Q2 gifting occasions (Valentine's, Mother's Day).**

### 2. Pareto Product Concentration
The top 10–15 SKUs account for approximately 70–80% of revenue. **Recommendation: Treat these as hero SKUs — never allow them to go out of stock, and use them as anchor products in bundles.**

### 3. Customer Segmentation
Champions and Loyal Customers represent a minority of the customer base but generate the majority of revenue. At-Risk customers have high historical spend but declining engagement. **Recommendation: Allocate 60% of CRM budget to Champions + Loyal Customers (retention) and 30% to At-Risk (win-back). Minimise spend on Dormant/Lost.**

### 4. Churn Signal
Recency is the strongest churn predictor. A customer who hasn't purchased in 90+ days is significantly more likely to be permanently lost. **Recommendation: Trigger automated win-back emails at 45-day and 75-day recency thresholds, before the customer crosses the churn boundary.**

---

## Limitations

- **Synthetic data**: Statistically realistic, but lacks real business events (competitor activity, supply shocks).
- **RFM simplification**: Ignores product affinity, channel, and demographic signals.
- **SARIMA linearity**: Cannot model structural breaks (e.g., a new product line launch).
- **Churn threshold**: 90 days is a principled heuristic; should be calibrated against actual retention data.

---

## Future Enhancements

- [ ] Customer Lifetime Value modelling (Pareto/NBD + BG/BB)
- [ ] Market Basket Analysis (FP-Growth cross-sell rules)
- [ ] Neural forecasting (Temporal Fusion Transformer)
- [ ] A/B test framework for campaign evaluation
- [ ] Real-time pipeline (Kafka → PySpark → Delta Lake)
- [ ] Interactive dashboard (Plotly Dash / Streamlit)

---

## Tech Stack

- **Language**: Python 3.11+
- **Data manipulation**: pandas, numpy
- **Visualisation**: matplotlib, seaborn
- **Statistics**: scipy, statsmodels
- **Machine learning**: scikit-learn, xgboost
- **Serialisation**: joblib

---

## Author

**[Ajsar Rihal Athiyattil]**
Data Analyst | MSc Data Science
[LinkedIn](www.linkedin.com/in/ajsarrihalathiyattil) ·

---

## License

MIT License — see [LICENSE](LICENSE.md)for dtails

