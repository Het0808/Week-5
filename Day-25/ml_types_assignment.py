"""
=============================================================
Assignment — Week 05 · Day 25 (PM Session)
PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar
Topics: Types of Machine Learning · Regression · Classification
=============================================================
"""

import pandas as pd
import numpy as np

# =============================================================
# PART A — CONCEPT APPLICATION
# =============================================================

print("=" * 60)
print("PART A — CONCEPT APPLICATION")
print("=" * 60)

# ── Problem Identification ─────────────────────────────────
print("""
--- TASK 1: Classify ML Problems ---

1. "A system predicts whether an email is spam or not using
    past labeled emails."
   → SUPERVISED LEARNING — Classification
   → Reason: Labeled data (spam/not spam), predicting a category

2. "A retail company groups customers based on purchasing
    behavior without any predefined labels."
   → UNSUPERVISED LEARNING — Clustering
   → Reason: No labels provided; algorithm finds hidden groups

3. "A robot learns to walk by trying different movements and
    receiving rewards when it moves correctly."
   → REINFORCEMENT LEARNING
   → Reason: Agent learns from trial-and-error; reward signal

4. "A model predicts house prices based on features like area,
    location, and number of rooms."
   → SUPERVISED LEARNING — Regression
   → Reason: Labeled data (price), predicting a continuous value

5. "An e-commerce platform recommends similar products based
    on user browsing history."
   → UNSUPERVISED LEARNING — (Collaborative/Content Filtering)
   → Reason: No explicit label; patterns found from behavior
""")

# ── Regression vs Classification ──────────────────────────
print("""
--- TASK 2: Regression vs Classification ---

REGRESSION — Target variable is CONTINUOUS (numeric)
  Example: Predict salary based on experience
  → Output: ₹45,000 / ₹82,500 (a number)
  → Algorithm: Linear Regression, SVR, Random Forest Regressor

CLASSIFICATION — Target variable is DISCRETE (category)
  Example: Predict if a loan applicant will default (Yes/No)
  → Output: "Yes" or "No" (a class label)
  → Algorithm: Logistic Regression, Decision Tree, SVM
""")

# ── Dataset: Supervised ML with Pandas ────────────────────
print("--- TASK 3: Supervised ML Dataset (Pandas DataFrame) ---")

data = {
    "Area_sqft":    [850,  1200, 1500, 2000, 750,  1800, 1100, 950,  2200, 1650],
    "Bedrooms":     [2,    3,    3,    4,    1,    4,    3,    2,    5,    3   ],
    "Location":     ["A",  "B",  "A",  "C",  "B",  "C",  "A",  "B",  "C",  "A" ],
    "Age_years":    [5,    10,   3,    15,   8,    2,    7,    12,   1,    6   ],
    "Price_lakhs":  [45,   65,   72,   95,   38,   88,   58,   48,   110,  78  ],
}

df = pd.DataFrame(data)
print("\nFull Dataset:")
print(df)

# Separate Features (X) and Target (y)
X = df.drop(columns=["Price_lakhs"])   # Features
y = df["Price_lakhs"]                  # Target

print("\n--- Features (X) ---")
print(X)

print("\n--- Target (y) ---")
print(y)
print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")


# =============================================================
# PART B — STRETCH PROBLEM
# =============================================================

print("\n" + "=" * 60)
print("PART B — STRETCH PROBLEM")
print("=" * 60)

print("""
--- REGRESSION EXAMPLE ---

Use Case: House Price Prediction
────────────────────────────────────────────────────────────
Input  (Features):
  • Area in square feet
  • Number of bedrooms
  • Location (A/B/C zone)
  • Age of the property (years)

Output (Target):
  • Price in lakhs (continuous numeric value)
  e.g., ₹72.5 lakhs

Why Regression?
  The output is a continuous real number, not a category.
  We need to estimate HOW MUCH the house costs.
  Algorithm: Linear Regression / Random Forest Regressor
────────────────────────────────────────────────────────────
""")

# Regression Dataset
reg_data = {
    "Area_sqft": [850, 1200, 1500, 2000, 750, 1800, 1100, 950, 2200, 1650],
    "Bedrooms":  [2,   3,    3,    4,    1,   4,    3,    2,   5,    3   ],
    "Age_years": [5,   10,   3,    15,   8,   2,    7,    12,  1,    6   ],
    "Price_lakhs": [45, 65,  72,   95,   38,  88,   58,   48,  110,  78  ],
}

reg_df = pd.DataFrame(reg_data)
X_reg = reg_df.drop(columns=["Price_lakhs"])
y_reg = reg_df["Price_lakhs"]

print("Regression — X (Features):")
print(X_reg)
print("\nRegression — y (Target — continuous):")
print(y_reg.to_frame())

print("""
--- CLASSIFICATION EXAMPLE ---

Use Case: Email Spam Detection
────────────────────────────────────────────────────────────
Input  (Features):
  • Number of exclamation marks in subject
  • Word count of email
  • Presence of certain keywords (1/0)
  • Sender domain trust score

Output (Target):
  • "Spam" or "Not Spam" (discrete category / binary)
  e.g., class = 1 (Spam)

Why Classification?
  The output belongs to a fixed set of classes.
  We need to decide WHICH category an email belongs to.
  Algorithm: Logistic Regression / Naive Bayes / SVM
────────────────────────────────────────────────────────────
""")

# Classification Dataset
clf_data = {
    "Exclamations":      [5, 0, 3, 8, 1, 6, 0, 2, 7, 0],
    "Word_Count":        [20, 200, 35, 15, 180, 10, 250, 40, 12, 220],
    "Has_Keyword":       [1,  0,  1,  1,  0,  1,  0,  0,  1,  0],
    "Sender_Trust":      [0.2, 0.9, 0.4, 0.1, 0.8, 0.2, 0.95, 0.6, 0.1, 0.85],
    "Is_Spam":           [1,   0,   1,   1,   0,   1,   0,    0,   1,   0],
}

clf_df = pd.DataFrame(clf_data)
X_clf = clf_df.drop(columns=["Is_Spam"])
y_clf = clf_df["Is_Spam"]

print("Classification — X (Features):")
print(X_clf)
print("\nClassification — y (Target — 0=Not Spam, 1=Spam):")
print(y_clf.to_frame())


# =============================================================
# PART C — INTERVIEW READY
# =============================================================

print("\n" + "=" * 60)
print("PART C — INTERVIEW READY")
print("=" * 60)

print("""
Q1 — What are the types of Machine Learning?
────────────────────────────────────────────────────────────
1. SUPERVISED LEARNING
   • Trained on labeled data (input → known output)
   • Two types:
     a) Regression  → predict a continuous value (price, temp)
     b) Classification → predict a category (spam/not spam)
   • Algorithms: Linear Regression, Decision Trees, SVM, etc.

2. UNSUPERVISED LEARNING
   • No labels; finds hidden structure in data
   • Types:
     a) Clustering → grouping similar data (K-Means)
     b) Dimensionality Reduction → PCA, t-SNE
     c) Association Rules → Market Basket Analysis
   • Algorithms: K-Means, DBSCAN, PCA, Apriori

3. REINFORCEMENT LEARNING
   • An agent learns by interacting with an environment
   • Receives rewards for good actions, penalties for bad
   • Used in: game playing (AlphaGo), robotics, self-driving cars
   • Algorithms: Q-Learning, PPO, DQN

(Optional 4th type)
4. SEMI-SUPERVISED LEARNING
   • Mix of labeled and unlabeled data
   • Useful when labeling is expensive
────────────────────────────────────────────────────────────
""")

print("--- Q2 (Coding): Separate features and target ---")

sample_df = pd.DataFrame({
    "Area_sqft":   [850, 1200, 1500, 2000, 750],
    "Bedrooms":    [2, 3, 3, 4, 1],
    "Age_years":   [5, 10, 3, 15, 8],
    "Price_lakhs": [45, 65, 72, 95, 38],
})

# Separate features and target
X = sample_df.drop(columns=["Price_lakhs"])   # All columns except target
y = sample_df["Price_lakhs"]                  # Target column only

print("\nDataset:")
print(sample_df)
print("\nFeatures (X):")
print(X)
print("\nTarget (y):")
print(y)

print("""
Q3 — Difference between Regression and Classification?
────────────────────────────────────────────────────────────
Feature          | Regression              | Classification
─────────────────|─────────────────────────|──────────────────────
Output type      | Continuous (numbers)    | Discrete (categories)
Goal             | Predict HOW MUCH        | Predict WHICH CLASS
Example output   | ₹72.5 lakhs, 36.2°C    | Spam / Not Spam
Metric           | RMSE, MAE, R²           | Accuracy, F1, AUC-ROC
Algorithms       | Linear Regression, SVR  | Logistic Reg., SVM, DT
Loss function    | Mean Squared Error      | Cross Entropy / Log Loss
Visual output    | Best-fit line/curve     | Decision boundary

Key Rule:
  → If target is a NUMBER  → Regression
  → If target is a LABEL   → Classification
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
  "Explain types of machine learning with real-world examples."

AI Output Summary:
  The AI described the three main types:

  1. SUPERVISED LEARNING
     Real-world example: Email spam detection (Gmail), 
     house price prediction (MagicBricks), disease diagnosis.
     → Learns a mapping: input features → labeled output.

  2. UNSUPERVISED LEARNING
     Real-world example: Customer segmentation (Amazon),
     anomaly detection in transactions (banks/fraud).
     → Discovers hidden patterns without labels.

  3. REINFORCEMENT LEARNING
     Real-world example: AlphaGo (chess/Go), autonomous cars
     (Tesla), recommendation ranking (YouTube autoplay).
     → Agent maximizes cumulative reward through exploration.

Evaluation:
  ✅ Correct — all 3 types accurately described
  ✅ Real-world examples are highly relevant and accurate
  ✅ Correct distinction between supervised vs unsupervised
  ⚠️  Missing — Semi-supervised and Self-supervised learning
               were not mentioned; increasingly important in
               modern deep learning (e.g., GPT pretraining)
  ⚠️  Missing — Transfer Learning not mentioned, which is
               widely used in NLP and computer vision tasks.
""")

print("ML Types Assignment completed successfully!")
