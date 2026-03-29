# =============================================================================
# ZeptoFresh – 15-Minute Food & Essentials Delivery
# Week 05 | Day 27 AM — Exploratory Data Analysis Assignment
# PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Styling ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#F8F9FA",
    "axes.facecolor":   "#FFFFFF",
    "axes.edgecolor":   "#CCCCCC",
    "axes.grid":        True,
    "grid.color":       "#EEEEEE",
    "grid.linewidth":   0.8,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
})
BRAND = "#E8533A"  # ZeptoFresh orange-red
BLUE  = "#2E86AB"
GREEN = "#3BB273"
GOLD  = "#F4A261"


# =============================================================================
# SECTION 0 — Synthetic Dataset Generation (mirrors real observations)
# =============================================================================
print("=" * 65)
print("  ZeptoFresh EDA — Week 05 Day 27 AM Assignment")
print("=" * 65)

np.random.seed(42)
N = 110_000

# Base delivery times: bimodal for Metro cities, unimodal for smaller cities
metro_n  = int(N * 0.60)
small_n  = N - metro_n

metro_times = np.concatenate([
    np.random.normal(13, 2.5, int(metro_n * 0.55)),   # Peak 1: 12–14 min
    np.random.normal(30, 3.0, int(metro_n * 0.45)),   # Peak 2: 28–32 min
])
small_times = np.random.exponential(scale=10, size=small_n) + 8

delivery_times_raw = np.concatenate([metro_times, small_times])
delivery_times_raw = np.clip(delivery_times_raw, 0, 150)

# Inject anomalies as per observations
delivery_times_raw[:214] = 0        # 214 zero-delivery rows (Obs 1)

cities = (["Mumbai"] * int(metro_n * 0.35) +
          ["Bangalore"] * int(metro_n * 0.35) +
          ["Delhi"] * (metro_n - 2 * int(metro_n * 0.35)) +
          ["Jaipur"] * int(small_n * 0.5) +
          ["Indore"] * (small_n - int(small_n * 0.5)))
cities = cities[:N]
np.random.shuffle(cities)

categories = np.random.choice(
    ["Grocery", "Fresh Food", "Bakery", "Medicines"],
    size=N, p=[0.45, 0.30, 0.15, 0.10]
)

order_values = np.random.lognormal(mean=5.7, sigma=1.0, size=N)
order_values = np.clip(order_values, 5, 10000)
order_values[0] = 295_000   # Single bakery outlier (Obs 2)

prep_times = np.random.randint(0, 30, size=N).astype(float)
prep_times[:80] = np.random.randint(-6, 0, 80)   # Negative prep times (Obs 3)

ratings = np.random.choice([1, 2, 3, 4, 5], size=N, p=[0.05, 0.10, 0.20, 0.40, 0.25]).astype(float)
null_mask = np.random.choice(N, 9800, replace=False)
ratings[null_mask] = np.nan                        # 9800 nulls (Obs 4)
zero_mask = np.random.choice(N, 300, replace=False)
ratings[zero_mask] = 0                             # Some 0-ratings (Obs 4)

rain_flag       = np.random.binomial(1, 0.20, N)
is_weekend      = np.random.binomial(1, 0.28, N)
order_hour      = np.random.randint(6, 24, N)
items_count     = np.random.randint(1, 20, N)
rider_distance  = np.random.uniform(0.3, 5.0, N)
customer_age    = np.random.randint(18, 65, N)
tenure_days     = np.random.randint(1, 1000, N)
coupon_used     = np.random.binomial(1, 0.35, N)
tip_amount      = np.where(ratings >= 4,
                           np.random.uniform(5, 50, N),
                           np.random.uniform(0,  5, N))
refund_issued   = (delivery_times_raw > 30).astype(int)
# Add some noise so correlation isn't perfect
refund_issued  ^= np.random.binomial(1, 0.05, N)

df = pd.DataFrame({
    "order_id":            range(1, N + 1),
    "hub_id":              np.random.randint(1, 351, N),
    "city":                cities,
    "order_category":      categories,
    "order_value_Rs":      order_values.round(2),
    "items_count":         items_count,
    "delivery_time_mins":  delivery_times_raw.round(1),
    "prep_time_mins":      prep_times,
    "rider_distance_km":   rider_distance.round(2),
    "order_hour":          order_hour,
    "is_weekend":          is_weekend,
    "rain_flag":           rain_flag,
    "customer_age":        customer_age,
    "customer_tenure_days":tenure_days,
    "coupon_used":         coupon_used,
    "tip_amount_Rs":       tip_amount.round(2),
    "refund_issued":       refund_issued,
    "customer_rating":     ratings,
})

print(f"\n✅ Dataset created: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
print(df.describe().round(2))


# =============================================================================
# SECTION A — Data Quality Diagnosis
# =============================================================================
print("\n" + "─" * 65)
print("  SECTION A — Data Quality Diagnosis")
print("─" * 65)

issues = {
    "Problem 1": {
        "column":    "delivery_time_mins",
        "type":      "Data Entry Error / Structural Issue",
        "detail":    f"214 rows have delivery_time_mins = 0 (impossible for a real delivery)",
        "treatment": "Replace zeros with NaN; impute using median of same hub_id & order_category",
        "count":     (df["delivery_time_mins"] == 0).sum(),
    },
    "Problem 2": {
        "column":    "order_value_Rs",
        "type":      "Outlier",
        "detail":    "Single Bakery order = ₹2,95,000 vs median ₹310 — 950× the median",
        "treatment": "Cap using IQR fence (Q3 + 3×IQR) or remove if confirmed data-entry error",
        "count":     (df["order_value_Rs"] > 50_000).sum(),
    },
    "Problem 3": {
        "column":    "prep_time_mins",
        "type":      "Data Entry Error",
        "detail":    f"{(df['prep_time_mins'] < 0).sum()} rows have negative prep times (min = {df['prep_time_mins'].min()})",
        "treatment": "Set negative values to NaN; impute using median prep time per order_category",
        "count":     (df["prep_time_mins"] < 0).sum(),
    },
    "Problem 4": {
        "column":    "customer_rating",
        "type":      "Missing Values + Out-of-Range Values",
        "detail":    f"{df['customer_rating'].isna().sum():,} nulls; {(df['customer_rating'] == 0).sum()} zeros (valid scale: 1–5)",
        "treatment": "Set 0→NaN; impute nulls using median rating per city; keep float64 type",
        "count":     df["customer_rating"].isna().sum() + (df["customer_rating"] == 0).sum(),
    },
}

for name, info in issues.items():
    print(f"\n  {name} | {info['column']}")
    print(f"  Type      : {info['type']}")
    print(f"  Detail    : {info['detail']}")
    print(f"  Treatment : {info['treatment']}")
    print(f"  Rows      : {info['count']:,}")

# ── Visualise all four problems ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Section A — Data Quality Issues in ZeptoFresh Dataset", fontsize=15, fontweight="bold", y=1.01)

# A1 — Delivery time zeros
ax = axes[0, 0]
sample = df["delivery_time_mins"].sample(5000, random_state=1)
ax.hist(sample[sample > 0], bins=60, color=BLUE, alpha=0.8, label="Valid")
ax.axvline(0, color=BRAND, lw=2, linestyle="--", label=f"Zero-time rows (n=214)")
ax.set_title("A1 · delivery_time_mins — Zero Values")
ax.set_xlabel("Delivery Time (mins)")
ax.set_ylabel("Count")
ax.legend()

# A2 — Order value outlier
ax = axes[0, 1]
vals = np.log1p(df["order_value_Rs"])
ax.hist(vals, bins=60, color=GREEN, alpha=0.8)
ax.axvline(np.log1p(295_000), color=BRAND, lw=2, linestyle="--", label="₹2,95,000 outlier")
ax.set_title("A2 · order_value_Rs — Extreme Outlier (log scale)")
ax.set_xlabel("log₁₊(Order Value ₹)")
ax.legend()

# A3 — Negative prep times
ax = axes[1, 0]
ax.hist(df["prep_time_mins"], bins=50, color=GOLD, alpha=0.85)
ax.axvline(0, color=BRAND, lw=2, linestyle="--", label=f"Negative values: {(df['prep_time_mins']<0).sum()}")
ax.set_title("A3 · prep_time_mins — Negative Values")
ax.set_xlabel("Prep Time (mins)")
ax.legend()

# A4 — Rating nulls + zeros
ax = axes[1, 1]
r_clean  = df["customer_rating"].dropna()
r_clean  = r_clean[r_clean > 0]
r_counts = r_clean.value_counts().sort_index()
bars = ax.bar(r_counts.index, r_counts.values, color=BLUE, alpha=0.8)
ax.bar([0], [(df["customer_rating"] == 0).sum()], color=BRAND, alpha=0.9, label="Zero ratings (invalid)")
null_patch = mpatches.Patch(color="grey", alpha=0.5, label=f"Null ratings: {df['customer_rating'].isna().sum():,}")
ax.set_title("A4 · customer_rating — Nulls & Zeros")
ax.set_xlabel("Rating")
ax.set_ylabel("Count")
ax.legend(handles=[bars[0], ax.patches[-1], null_patch],
          labels=["Valid ratings", "Zero (invalid)", f"Null: {df['customer_rating'].isna().sum():,}"])

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/A_data_quality.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  ✅ Saved: A_data_quality.png")


# =============================================================================
# SECTION B — Distribution Analysis
# =============================================================================
print("\n" + "─" * 65)
print("  SECTION B — Distribution Analysis")
print("─" * 65)

dt = df["delivery_time_mins"]
mean_val   = dt.mean()
median_val = dt.median()
skew_val   = dt.skew()

print(f"\n  Mean   : {mean_val:.2f} mins")
print(f"  Median : {median_val:.2f} mins")
print(f"  Skew   : {skew_val:.3f}  →  Right-skewed (positive skew)")
print("""
  Interpretation:
  Mean > Median indicates a RIGHT-SKEWED (positively skewed) distribution.
  A long right tail is pulled by extreme outliers (142-min deliveries).
  Most deliveries happen under 20 mins, but occasional very late
  deliveries inflate the mean upward.

  ASCII Histogram (delivery_time_mins):
  ─────────────────────────────────────────────────────
   0–10 | ████████████████  (early/zero anomalies + fast)
  10–15 | ██████████████████████████  (modal cluster)
  15–20 | ████████████████  (normal deliveries)
  20–25 | ████████
  25–35 | █████████  (second peak in metro cities)
  35–50 | ███
  50+   | █  (long tail — outliers)
  ─────────────────────────────────────────────────────
  Transformation: Log1p (log(1 + x)) — compresses the right tail,
  normalises the distribution, handles zero-like values, and makes
  linear model assumptions (normality) more valid before modelling.
""")

# ── Clean data for transformation demo ───────────────────────────────────────
dt_clean = dt[(dt > 0) & (dt < 100)]
dt_log   = np.log1p(dt_clean)
dt_sqrt  = np.sqrt(dt_clean)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Section B — Distribution Analysis & Transformations", fontsize=14, fontweight="bold")

axes[0].hist(dt_clean, bins=60, color=BRAND, alpha=0.85, edgecolor="white", lw=0.3)
axes[0].axvline(mean_val,   color="navy",  lw=2, linestyle="--", label=f"Mean {mean_val:.1f}")
axes[0].axvline(median_val, color="green", lw=2, linestyle="-",  label=f"Median {median_val:.1f}")
axes[0].set_title("Original — Right-Skewed")
axes[0].set_xlabel("delivery_time_mins")
axes[0].legend(fontsize=9)

axes[1].hist(dt_log, bins=60, color=BLUE, alpha=0.85, edgecolor="white", lw=0.3)
axes[1].set_title("Log1p Transform ✅ (Recommended)")
axes[1].set_xlabel("log1p(delivery_time_mins)")

axes[2].hist(dt_sqrt, bins=60, color=GREEN, alpha=0.85, edgecolor="white", lw=0.3)
axes[2].set_title("Square-Root Transform")
axes[2].set_xlabel("√(delivery_time_mins)")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/B_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: B_distribution.png")


# =============================================================================
# SECTION C — Correlation Interpretation
# =============================================================================
print("\n" + "─" * 65)
print("  SECTION C — Correlation Interpretation")
print("─" * 65)

corr_dt_refund = df["delivery_time_mins"].corr(df["refund_issued"])
corr_rain      = df["rain_flag"].corr(df["delivery_time_mins"])
corr_tip_rate  = df["tip_amount_Rs"].corr(df["customer_rating"].fillna(df["customer_rating"].median()))

print(f"\n  Computed correlations (synthetic data):")
print(f"  delivery_time_mins ↔ refund_issued  : r = {corr_dt_refund:.2f}")
print(f"  rain_flag          ↔ delivery_time  : r = {corr_rain:.2f}")
print(f"  tip_amount_Rs      ↔ customer_rating: r = {corr_tip_rate:.2f}")

print("""
  PM's Claim: "Late deliveries cause refunds → solving delay eliminates refunds"
  ─────────────────────────────────────────────────────────────────────────────
  What is WRONG:
  1. Correlation ≠ Causation. A high r value shows association, not direction
     of cause and effect.
  2. Even if delay does cause refunds, it may not be the SOLE cause.
     Refunds can occur for wrong items, damaged products, payment errors —
     independent of delivery time.
  3. "Eliminate" is too strong — solving delay would reduce refunds correlated
     with it, but not all refunds.

  What correlation ACTUALLY means:
  r = +0.74 → Strong positive linear association. When delivery_time_mins
  increases, refund_issued tends to increase. It does NOT tell us WHY.

  Possible Confounders (variables that could drive BOTH):
  ① Rain / Bad Weather  — slows riders (↑ delay) AND damages packaging
                          (↑ refund) independently.
  ② Order Complexity    — large fresh-food orders take longer to prep (↑ delay)
                          and have higher spoilage risk (↑ refund).
  ③ Hub Understaffing   — under-resourced hubs have slower ops (↑ delay) and
                          more packing errors (↑ refund).
  ④ Time of Day (Peak)  — rush hours cause congestion (↑ delay) and higher
                          order volumes increase error rates (↑ refund).
""")

# ── Correlation heatmap ───────────────────────────────────────────────────────
numeric_cols = ["delivery_time_mins", "refund_issued", "rain_flag",
                "tip_amount_Rs", "order_value_Rs", "prep_time_mins",
                "rider_distance_km", "items_count"]
corr_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, vmin=-1, vmax=1,
            linewidths=0.5, ax=ax, annot_kws={"size": 10})
ax.set_title("Section C — Correlation Matrix (Key Variables)", fontweight="bold")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/C_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: C_correlation.png")


# =============================================================================
# SECTION D — Bimodal Pattern in Tier-1 Cities
# =============================================================================
print("\n" + "─" * 65)
print("  SECTION D — Bimodal Pattern in Tier-1 Cities")
print("─" * 65)

print("""
  Operational Reasons for Bimodal Distribution:
  ──────────────────────────────────────────────
  Peak 1 (12–14 min): Orders from hubs very close to customer, simple SKUs,
  off-peak hours — "easy" deliveries completing on time.

  Peak 2 (28–32 min): Orders hitting traffic congestion, long rider distance,
  complex or heavy items (fresh food), peak-hour surges — structurally
  delayed deliveries forming a second cluster.

  In smaller cities, traffic is lighter and hubs more centralised →
  unimodal near-on-time distribution.

  Why this MUST be addressed before modelling:
  ─────────────────────────────────────────────
  A bimodal distribution violates the unimodal normality assumption of
  many ML algorithms (Logistic Regression, LDA). The model may learn the
  wrong "average" behaviour and perform poorly on both sub-populations.

  Modelling Mistake if Ignored:
  ─────────────────────────────
  ① A single decision boundary will split the two modes incorrectly,
     creating high false-positive and false-negative rates.
  ② Feature importance scores will be misleading because the two
     sub-populations may be driven by different features (distance
     vs. congestion).
  ③ Recommended fix: Segment data by city-tier OR add a city-tier
     indicator feature, OR use a mixture model / cluster-then-model approach.
""")

# ── Bimodal visualisation ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Section D — Bimodal vs Unimodal Delivery Time Distribution", fontweight="bold")

metro_dt = df[df["city"].isin(["Mumbai", "Bangalore"])]["delivery_time_mins"]
metro_dt = metro_dt[(metro_dt > 0) & (metro_dt < 60)]
small_dt = df[df["city"].isin(["Jaipur", "Indore"])]["delivery_time_mins"]
small_dt = small_dt[(small_dt > 0) & (small_dt < 60)]

axes[0].hist(metro_dt, bins=50, color=BRAND, alpha=0.85, edgecolor="white", lw=0.3)
axes[0].axvline(13, color="navy", lw=1.8, linestyle="--", label="Peak 1: 12–14 min")
axes[0].axvline(30, color="green", lw=1.8, linestyle="--", label="Peak 2: 28–32 min")
axes[0].set_title("Mumbai & Bangalore — Bimodal ⚠️")
axes[0].set_xlabel("delivery_time_mins")
axes[0].legend()

axes[1].hist(small_dt, bins=50, color=BLUE, alpha=0.85, edgecolor="white", lw=0.3)
axes[1].set_title("Jaipur & Indore — Unimodal ✅")
axes[1].set_xlabel("delivery_time_mins")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/D_bimodal.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: D_bimodal.png")


# =============================================================================
# SECTION E — Business Trade-Off (Precision vs Recall)
# =============================================================================
print("\n" + "─" * 65)
print("  SECTION E — Precision vs Recall Trade-Off")
print("─" * 65)

print("""
  Answer: Prioritise RECALL (Sensitivity) — with a caveat.
  ──────────────────────────────────────────────────────────
  • High Recall → catch as many true late-delivery risks as possible.
    Missing a truly late delivery (False Negative) means a customer
    experiences a bad delivery, likely churns, and revenue is permanently
    lost. Churn is expensive.

  • Low Precision (acceptable cost) → some unnecessary rider reallocations
    (False Positives). These incur operational costs but are recoverable —
    the rider is reassigned back to normal operations quickly.

  Framing it as a cost matrix:
  ┌─────────────────┬─────────────────────────┬──────────────────────────┐
  │                 │  Predicted: Late         │  Predicted: On-Time      │
  ├─────────────────┼─────────────────────────┼──────────────────────────┤
  │ Actual: Late    │ TP → Proactive fix ✅   │ FN → Churn risk 🚨 HIGH  │
  │ Actual: On-Time │ FP → Wasted ops cost ⚠️ │ TN → Normal ops ✅       │
  └─────────────────┴─────────────────────────┴──────────────────────────┘

  Caveat: If recall becomes too high (model flags 80%+ of orders),
  operational costs explode. Use F-beta score (β > 1) to weight recall
  higher than precision, or set a business-calibrated threshold on the
  predicted probability (e.g., alert only if P(late) > 0.40).
""")

# ── Precision-Recall trade-off curve ─────────────────────────────────────────
thresholds = np.linspace(0.01, 0.99, 200)
precision_curve = 1 / (1 + np.exp(-8 * (thresholds - 0.6)))
recall_curve    = 1 / (1 + np.exp( 8 * (thresholds - 0.4)))
f1_curve = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-9)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(thresholds, precision_curve, color=BLUE,  lw=2.5, label="Precision")
ax.plot(thresholds, recall_curve,    color=BRAND, lw=2.5, label="Recall")
ax.plot(thresholds, f1_curve,        color=GREEN, lw=2,   linestyle="--", label="F1-Score")
ax.axvspan(0.01, 0.45, alpha=0.08, color=BRAND, label="High Recall Zone (Recommended)")
ax.set_title("Section E — Precision vs Recall Trade-Off (Conceptual)", fontweight="bold")
ax.set_xlabel("Classification Threshold")
ax.set_ylabel("Score")
ax.legend()
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/E_precision_recall.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: E_precision_recall.png")


# =============================================================================
# SECTION F — Advanced Feature Engineering
# =============================================================================
print("\n" + "─" * 65)
print("  SECTION F — Feature Engineering")
print("─" * 65)

# Fix data before feature engineering
df_fe = df.copy()
df_fe.loc[df_fe["delivery_time_mins"] == 0,  "delivery_time_mins"] = np.nan
df_fe["delivery_time_mins"].fillna(df_fe["delivery_time_mins"].median(), inplace=True)
df_fe.loc[df_fe["prep_time_mins"] < 0,       "prep_time_mins"] = np.nan
df_fe["prep_time_mins"].fillna(df_fe.groupby("order_category")["prep_time_mins"].transform("median"), inplace=True)
df_fe.loc[df_fe["customer_rating"] == 0,     "customer_rating"] = np.nan

# ── Feature 1: prep_to_delivery_ratio ────────────────────────────────────────
# Measures what fraction of total delivery time was spent in preparation.
# A high ratio means the rider was idle waiting for the order — hub bottleneck.
df_fe["prep_to_delivery_ratio"] = df_fe["prep_time_mins"] / (df_fe["delivery_time_mins"] + 1e-5)

# ── Feature 2: distance_per_item ─────────────────────────────────────────────
# Normalises rider distance by basket size.
# Large baskets over long distances are harder to deliver on time.
df_fe["distance_per_item"] = df_fe["rider_distance_km"] / df_fe["items_count"]

# ── Feature 3: is_peak_rain_hour ─────────────────────────────────────────────
# Binary interaction feature: rain during evening rush (18–21h) or lunch (12–14h).
# Captures the compound risk of congestion + bad weather simultaneously.
df_fe["is_peak_rain_hour"] = (
    (df_fe["rain_flag"] == 1) &
    (df_fe["order_hour"].isin(range(12, 15)) | df_fe["order_hour"].isin(range(18, 22)))
).astype(int)

print("""
  Engineered Features:
  ──────────────────────────────────────────────────────────────────────────
  Feature 1 — prep_to_delivery_ratio
    Formula  : prep_time_mins / (delivery_time_mins + ε)
    Rationale: Isolates hub-side bottlenecks from rider-side delays.
               High ratio → prep is the constraint, not rider speed.

  Feature 2 — distance_per_item
    Formula  : rider_distance_km / items_count
    Rationale: A heavy 15-item order over 4 km is harder than a 1-item
               order over the same distance. Normalising by basket size
               captures combined load + distance stress.

  Feature 3 — is_peak_rain_hour
    Formula  : (rain_flag == 1) AND (order_hour IN [12-14] OR [18-21])
    Rationale: Rain alone and peak hour alone each increase delay risk.
               Their intersection (compound risk) is a non-linear
               interaction that a tree model may not learn automatically
               from raw features alone.
""")

print("  Feature statistics:")
print(df_fe[["prep_to_delivery_ratio", "distance_per_item", "is_peak_rain_hour"]].describe().round(4))

# ── Feature correlation with late delivery ────────────────────────────────────
df_fe["late_flag"] = (df_fe["delivery_time_mins"] > 20).astype(int)
eng_feats = ["prep_to_delivery_ratio", "distance_per_item", "is_peak_rain_hour",
             "rain_flag", "rider_distance_km", "prep_time_mins"]

feat_corr = df_fe[eng_feats + ["late_flag"]].corr()["late_flag"].drop("late_flag").sort_values()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Section F — Engineered Feature Distributions & Correlations", fontweight="bold")

colors = [BRAND if "ratio" in c or "item" in c or "peak" in c else BLUE for c in feat_corr.index]
axes[0].barh(feat_corr.index, feat_corr.values, color=colors, alpha=0.85, edgecolor="white")
axes[0].axvline(0, color="black", lw=1)
axes[0].set_title("Feature Correlation with late_flag (>20 min)")
axes[0].set_xlabel("Pearson r")
leg = [mpatches.Patch(color=BRAND, label="Engineered features"),
       mpatches.Patch(color=BLUE,  label="Original features")]
axes[0].legend(handles=leg, fontsize=9)

axes[1].hist(df_fe["prep_to_delivery_ratio"].clip(0, 2), bins=60,
             color=BRAND, alpha=0.85, edgecolor="white", lw=0.3)
axes[1].set_title("prep_to_delivery_ratio Distribution")
axes[1].set_xlabel("prep_time / delivery_time")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/F_features.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  ✅ Saved: F_features.png")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("  ALL SECTIONS COMPLETE — Output plots saved to /outputs/")
print("=" * 65)
print("""
  Files generated:
  • A_data_quality.png     — Section A: 4 data quality issues
  • B_distribution.png     — Section B: skewed distribution + transforms
  • C_correlation.png      — Section C: correlation heatmap
  • D_bimodal.png          — Section D: bimodal vs unimodal
  • E_precision_recall.png — Section E: precision-recall trade-off
  • F_features.png         — Section F: engineered feature analysis
""")
