# Supply Chain Delivery Risk Intelligence

A machine learning system that predicts late-delivery risk, segments orders into operational groups, and flags anomalous orders — built on the DataCo Smart Supply Chain dataset (~180k order-level records).

---

## Problem

Late deliveries hurt customer trust and operational cost. This project answers three questions a supply chain team actually faces:

1. **Which orders are likely to be delivered late?** (supervised prediction)
2. **What natural groups of orders exist?** (unsupervised segmentation)
3. **Which orders are abnormal and worth a closer look?** (anomaly detection)

---

## Dataset

- **Source:** DataCo Smart Supply Chain dataset
- **Size:** 180,519 order-line records × 53 raw columns
- **Target:** `Late_delivery_risk` (binary — 55% late / 45% on-time)
- **After cleaning:** 22 leakage-free features used for modeling

A key step was **leakage removal** — columns that encode the delivery outcome itself (e.g. actual shipping days, delivery status) were dropped, since they wouldn't be available at prediction time.

---

## Approach

The project is built on four analytical pillars:

| Pillar | Method | Purpose |
|---|---|---|
| **Supervised learning** | Logistic Regression, Random Forest, XGBoost | Predict late-delivery risk |
| **Dimensionality reduction** | PCA | Compress correlated features for clustering |
| **Clustering** | K-Means (elbow + silhouette) | Segment orders into operational groups |
| **Anomaly detection** | Isolation Forest | Flag abnormal orders |

---

## Results

### Supervised model — delivery risk prediction

Four models were compared against a baseline, optimizing for **ROC-AUC** (chosen because it measures ranking quality across all thresholds and can't be gamed by predicting one class).

| Model | Accuracy | Late Recall | ROC-AUC |
|---|---|---|---|
| Dummy (baseline) | 0.55 | 1.00 | 0.50 |
| Logistic Regression | 0.69 | 0.59 | 0.72 |
| Random Forest | 0.72 | 0.62 | 0.80 |
| XGBoost | 0.70 | 0.59 | 0.76 |
| **Random Forest (tuned)** | **0.72** | **0.62** | **0.80** |

The **tuned Random Forest** was selected (ROC-AUC 0.80). Notably, it outperformed XGBoost — a reminder that the "expected" winner isn't always best, and that comparing models matters.

**Overfitting check:** The tuned RF showed train accuracy 1.0 vs test 0.72. This is expected Random Forest behavior (deep trees memorize bootstrap samples). A constrained version (depth 12) eliminated the gap but *lowered* test ROC-AUC to 0.76 — confirming the deep model generalizes better despite the apparent gap. The tuned RF was retained.

**Top driver:** shipping speed, confirming the EDA finding that faster-promised shipping is *more* often late.

### Clustering — order segmentation

PCA reduced the 22 features to 13 components (90% variance retained). K-Means with k=6 (chosen via elbow + silhouette) partitioned orders into operational segments.

Silhouette scores were low (~0.10), indicating orders form a **continuous mass rather than sharply distinct groups**. However, profiling surfaced two operationally meaningful segments:

- **Express segment:** fast-shipping orders with an **81% late rate** — the clearest operational risk.
- **Loss-making segment:** ~6,500 orders averaging **−$347 profit each**.

### Anomaly detection

Isolation Forest flagged ~2% of orders (3,611) as anomalies. These are overwhelmingly **extreme loss-making orders** (avg −$271 vs +$28 for normal orders).

Clustering and anomaly detection **independently converged on profit loss** as the key abnormal signal — a strong sign of analytical consistency.

---

## Business Insights

1. **Late delivery is driven by shipping mode, not geography or customer type.** Faster-promised shipping is *more* often late (First Class ~95% late vs Standard ~38%), pointing to over-committed expedited delivery windows.

2. **The model ranks delivery risk well (ROC-AUC 0.80).** Risk scores (0–1) let operations prioritize the highest-risk orders for proactive intervention rather than treating all orders equally.

3. **Orders vary continuously, but two segments stand out:** an express segment (81% late) and a loss-making segment (−$347/order avg).

4. **Anomalous orders are extreme money-losers** — surfaced independently by both clustering and anomaly detection.

### Recommended actions
- **Audit expedited shipping commitments** — the express segment's 81% late rate suggests unrealistic delivery promises.
- **Review loss-making orders** flagged by both clustering and anomaly detection for pricing/discount issues.
- **Deploy the risk model** to flag high-risk orders pre-shipment.

---

## Model Card

**Task:** Predict late-delivery risk (binary classification).

**Model:** Random Forest (tuned via GridSearchCV), selected over Logistic Regression and XGBoost.

**Performance (held-out test set):** ROC-AUC 0.80 · Accuracy 0.72 · Late-delivery recall 0.62.

**Key features:** Shipping speed, scheduled shipment days, order profitability.

**Limitations:**
- Late delivery is driven mainly by shipping mode; the model reflects this dependency.
- The train/test accuracy gap reflects normal RF behavior; test ROC-AUC confirms genuine generalization.
- Trained on historical data from one retailer; may not transfer without retraining.

**Intended use:** Flag high-risk orders pre-shipment for operational review — not a substitute for human logistics judgment.

---

## Repository Structure

```
supply-chain-risk-intelligence/
├── data/
│   ├── raw/              # DataCo dataset (gitignored)
│   └── processed/        # cleaned & feature-engineered data (gitignored)
├── notebooks/
│   └── supply_chain.ipynb
├── requirements.txt
└── README.md
```

---

## How to Run

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download the [DataCo Smart Supply Chain dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) and place `DataCoSupplyChainDataset.csv` in `data/raw/`
4. Open `notebooks/supply_chain.ipynb` and run all cells

---

## Tech Stack

Python · pandas · scikit-learn · XGBoost · matplotlib · seaborn