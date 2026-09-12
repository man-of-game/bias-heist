"""
Bias Heist - Model Training
Trains a gradient-boosted classifier on the synthetic biased visa dataset.
The model is NOT told about the bias explicitly - it just learns whatever
patterns exist in the labels, including the injected purpose x prior_approvals
interaction and the purpose_of_visit / english_level correlation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

df = pd.read_csv("visa_dataset.csv")

CAT_COLS = ["education_level", "purpose_of_visit", "english_level", "legal_charges"]
NUM_COLS = ["age", "annual_income_inr", "bank_balance_inr", "prior_approvals"]

# Fixed category orders so encoding is stable/reproducible
edu_order = ["high_school", "bachelors", "masters", "phd"]
purpose_order = ["business", "travel", "education", "medical"]
eng_order = ["basic", "good", "fluent"]
legal_order = ["no", "yes"]

encoder = OrdinalEncoder(
    categories=[edu_order, purpose_order, eng_order, legal_order]
)

X_cat = encoder.fit_transform(df[CAT_COLS])
X = np.hstack([df[NUM_COLS].to_numpy(), X_cat])
feature_names = NUM_COLS + CAT_COLS
y = df["visa_approved"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = GradientBoostingClassifier(
    n_estimators=250,
    max_depth=3,
    learning_rate=0.08,
    random_state=42,
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
print("Test accuracy:", round(accuracy_score(y_test, pred), 4))
print("Test ROC-AUC:", round(roc_auc_score(y_test, proba), 4))

joblib.dump(
    {
        "model": model,
        "encoder": encoder,
        "cat_cols": CAT_COLS,
        "num_cols": NUM_COLS,
        "edu_order": edu_order,
        "purpose_order": purpose_order,
        "eng_order": eng_order,
        "legal_order": legal_order,
    },
    "visa_model.joblib",
)
print("\nModel saved to visa_model.joblib")

# Sanity check: does the trained model reproduce the interaction pattern?
def make_row(age=30, income=600000, balance=300000, edu="bachelors",
             purpose="business", prior=0, eng="good", legal="no"):
    row_cat = encoder.transform([[edu, purpose, eng, legal]])
    row = np.hstack([[[age, income, balance, prior]], row_cat])
    return row

print("\n--- Sanity check: model-learned interaction pattern ---")
for prior in [0, 5]:
    for purpose in ["business", "medical"]:
        row = make_row(purpose=purpose, prior=prior)
        p = model.predict_proba(row)[0, 1]
        print(f"prior_approvals={prior:2d}  purpose={purpose:10s} -> P(approve)={p:.3f}")
