"""
=============================================================
Assignment — Week 05 · Day 25 (AM Session)
PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar
Topics: EDA (Basic) · Pandas
=============================================================
"""

import pandas as pd
import numpy as np

# =============================================================
# DATASET: Titanic-style synthetic dataset (600+ rows)
# =============================================================

np.random.seed(42)
n = 620

data = {
    "PassengerId": range(1, n + 1),
    "Name": ["Passenger_" + str(i) for i in range(1, n + 1)],
    "Age": np.where(np.random.rand(n) < 0.12, np.nan,
                    np.random.normal(35, 14, n).clip(1, 80).round(1)),
    "Gender": np.random.choice(["Male", "Female"], n),
    "Pclass": np.random.choice([1, 2, 3], n, p=[0.2, 0.3, 0.5]),
    "Fare": np.where(np.random.rand(n) < 0.05, np.nan,
                     np.random.exponential(35, n).round(2)),
    "Survived": np.random.choice([0, 1], n, p=[0.6, 0.4]),
    "Embarked": np.where(np.random.rand(n) < 0.08, None,
                         np.random.choice(["S", "C", "Q"], n, p=[0.7, 0.2, 0.1])),
    "SibSp": np.random.choice(range(0, 6), n),
    "Parch": np.random.choice(range(0, 4), n),
}

df = pd.DataFrame(data)
df.to_csv("titanic_synthetic.csv", index=False)
df = pd.read_csv("titanic_synthetic.csv")

# =============================================================
# PART A — CONCEPT APPLICATION
# =============================================================

print("=" * 60)
print("PART A — CONCEPT APPLICATION")
print("=" * 60)

# ── Basic Exploration ──────────────────────────────────────
print("\n--- TASK 1: First 5 Rows ---")
print(df.head())

print("\n--- TASK 2: Last 5 Rows ---")
print(df.tail())

print("\n--- TASK 3: Shape ---")
print("Shape:", df.shape)

print("\n--- TASK 4: Column Names ---")
print("Columns:", df.columns.tolist())

# ── Data Cleaning ──────────────────────────────────────────
print("\n--- TASK 5: Missing Values ---")
print(df.isnull().sum())

print("\n--- TASK 6: Fill/Drop Missing Values ---")
# Fill Age with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill Fare with median
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Fill Embarked with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

print("Missing values after cleaning:")
print(df.isnull().sum())

# ── Descriptive Statistics ─────────────────────────────────
print("\n--- TASK 7: describe() ---")
print(df.describe())

print("""
--- TASK 8: Interpretation of describe() ---
  Age   → mean ~35 yrs, min ~1, max ~80, std ~14 (wide spread of ages)
  Fare  → mean ~35, max very high (few premium tickets), std shows skew
  Survived → mean ~0.4 means ~40% survived (binary: 0/1)
  Pclass → mean ~2.3 shows most passengers in 3rd class
""")

# ── Categorical Analysis ───────────────────────────────────
print("\n--- TASK 9: Unique Values ---")
for col in ["Gender", "Pclass", "Embarked", "Survived"]:
    print(f"  {col}: {df[col].unique()}")

print("\n--- TASK 10: Frequency Count ---")
for col in ["Gender", "Pclass", "Embarked"]:
    print(f"\n{col} value counts:")
    print(df[col].value_counts())


# =============================================================
# PART B — STRETCH PROBLEM
# =============================================================

print("\n" + "=" * 60)
print("PART B — STRETCH PROBLEM")
print("=" * 60)

# ── Filter with multiple conditions ───────────────────────
print("\n--- TASK 1: Filter — Female passengers, Pclass 1, Age > 30 ---")
filtered = df[(df["Gender"] == "Female") & (df["Pclass"] == 1) & (df["Age"] > 30)]
print(f"Records found: {len(filtered)}")
print(filtered[["PassengerId", "Gender", "Pclass", "Age", "Fare", "Survived"]].head())

# ── New column ────────────────────────────────────────────
print("\n--- TASK 2: New Column — FareCategory ---")
df["FareCategory"] = pd.cut(df["Fare"],
                             bins=[0, 15, 50, 300],
                             labels=["Low", "Medium", "High"])
print(df[["PassengerId", "Fare", "FareCategory"]].head(10))

# ── Sort ──────────────────────────────────────────────────
print("\n--- TASK 3: Sort by Fare (descending) ---")
sorted_df = df.sort_values("Fare", ascending=False)
print(sorted_df[["PassengerId", "Fare", "Pclass"]].head(10))

# ── GroupBy ──────────────────────────────────────────────
print("\n--- TASK 4: GroupBy Pclass — Mean Age & Fare ---")
grouped = df.groupby("Pclass")[["Age", "Fare"]].mean().round(2)
print(grouped)


# =============================================================
# PART C — INTERVIEW READY
# =============================================================

print("\n" + "=" * 60)
print("PART C — INTERVIEW READY")
print("=" * 60)

print("""
Q1 — What is EDA? Why is it important?
────────────────────────────────────────────────────────────
EDA (Exploratory Data Analysis) is the process of analyzing
datasets to summarize their main characteristics, often using
visual methods. Before applying ML algorithms, EDA helps us:
  • Understand data structure, types, and shape
  • Detect missing values, outliers, and anomalies
  • Discover patterns, correlations, and distributions
  • Formulate hypotheses for model building
  • Ensure data quality and integrity
Without EDA, models may be built on flawed assumptions,
leading to poor predictions or misleading results.
────────────────────────────────────────────────────────────
""")

print("Q2 (Coding) — Filter rows where Age > mean(Age):")
mean_age = df["Age"].mean()
above_avg_age = df[df["Age"] > mean_age]
print(f"  Mean Age = {mean_age:.2f}")
print(f"  Rows where Age > mean: {len(above_avg_age)} out of {len(df)}")
print(above_avg_age[["PassengerId", "Age", "Gender", "Pclass"]].head())

print("""
Q3 — What insights do we get from describe()?
────────────────────────────────────────────────────────────
  count  → how many non-null values (helps detect missing data)
  mean   → central tendency / average value
  std    → spread / variability of data
  min    → smallest value (helps detect outliers / errors)
  25%    → 1st quartile, 25% of data lies below this
  50%    → median, middle value (robust to outliers)
  75%    → 3rd quartile, 75% of data lies below this
  max    → largest value (helps detect outliers)

  A large gap between mean & median → skewed distribution.
  A large std → data is spread out widely.
────────────────────────────────────────────────────────────
""")


# =============================================================
# PART D — AI-AUGMENTED TASK
# =============================================================

print("=" * 60)
print("PART D — AI-AUGMENTED TASK")
print("=" * 60)

print("""
Prompt used:
  "Explain EDA steps using Pandas with examples."

AI Output Summary:
  The AI explained EDA in the following steps:
  1. Load data          → pd.read_csv("file.csv")
  2. Explore structure  → df.shape, df.dtypes, df.head()
  3. Missing values     → df.isnull().sum(), df.fillna(), df.dropna()
  4. Descriptive stats  → df.describe()
  5. Unique & frequency → df[col].unique(), df[col].value_counts()
  6. Visualize          → df.hist(), df.boxplot(), sns.heatmap()
  7. Correlations       → df.corr()

Evaluation:
  ✅ Correct — all steps are standard EDA practices
  ✅ Correct — Pandas API calls are accurate
  ✅ Relevant — examples match real-world use cases
  ⚠️  Missing — data type conversion (df.astype()) and
               duplicate removal (df.drop_duplicates())
               were not mentioned but are important EDA steps.
""")

print("EDA Assignment completed successfully!")
