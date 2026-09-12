"""
Bias Heist - Synthetic Dataset Generator
US Visa Approval Model

Hidden bias (ground truth, for Leads Team only):
  - PRIMARY: 'purpose' interacts with 'prior_approvals'. Business-purpose
    applicants get a strong hidden approval boost when prior_approvals is
    LOW (0-2). Medical-purpose applicants get a corresponding penalty in
    the same low-history regime. The effect nearly vanishes once
    prior_approvals >= 3, because travel history dominates and swamps the
    purpose signal. This mirrors real consular officers historically
    treating business travel as lower flight-risk.
  - RED HERRING: english_level correlates with purpose in the population
    (business applicants skew more fluent) but has only a small, honest,
    independent effect on the label. It LOOKS causal in raw correlation
    but isn't - holding purpose constant kills most of the apparent effect.
  - HARD GATE: legal_charges == 'yes' -> deterministic reject regardless
    of everything else. This is an intentional non-bias control variable.

No label noise is injected (clean signal), per request.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 5000

# ---------------------------------------------------------------------
# 1. Base feature sampling
# ---------------------------------------------------------------------

age = RNG.integers(18, 81, size=N)

purpose = RNG.choice(
    ["business", "travel", "education", "medical"],
    size=N,
    p=[0.30, 0.35, 0.20, 0.15],
)

# English fluency correlated with purpose (business skews fluent) -
# this creates the spurious correlation for the red herring.
def sample_english(p):
    if p == "business":
        probs = [0.15, 0.35, 0.50]       # basic, good, fluent
    elif p == "education":
        probs = [0.20, 0.45, 0.35]
    elif p == "travel":
        probs = [0.35, 0.40, 0.25]
    else:  # medical
        probs = [0.40, 0.40, 0.20]
    return RNG.choice(["basic", "good", "fluent"], p=probs)

english_level = np.array([sample_english(p) for p in purpose])

education_level = RNG.choice(
    ["high_school", "bachelors", "masters", "phd"],
    size=N,
    p=[0.25, 0.40, 0.25, 0.10],
)

# Prior approvals: many first-time applicants, tail of frequent travelers
prior_approvals = RNG.choice(
    np.arange(0, 11),
    size=N,
    p=[0.30, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.02, 0.01],
)

# Income & balance (INR), log-normal, mildly correlated with education
edu_income_mult = {"high_school": 1.0, "bachelors": 1.4, "masters": 1.8, "phd": 2.0}
base_income = RNG.lognormal(mean=13.2, sigma=0.55, size=N)  # ~ centered around 5-6L INR
annual_income = np.array(
    [base_income[i] * edu_income_mult[education_level[i]] for i in range(N)]
).round(-2)
annual_income = np.clip(annual_income, 80_000, 15_000_000)

bank_balance = (annual_income * RNG.uniform(0.15, 0.9, size=N)).round(-2)
bank_balance = np.clip(bank_balance, 20_000, 20_000_000)

legal_charges = RNG.choice(["yes", "no"], size=N, p=[0.06, 0.94])

# ---------------------------------------------------------------------
# 2. Latent approval score (ground truth generative process)
# ---------------------------------------------------------------------

edu_score = pd.Series(education_level).map(
    {"high_school": 0.0, "bachelors": 0.7, "masters": 1.2, "phd": 1.5}
).to_numpy()  # real, meaningful effect

eng_score = pd.Series(english_level).map(
    {"basic": 0.0, "good": 0.6, "fluent": 1.1}
).to_numpy()  # real, meaningful effect (still doubles as the red herring's
              # apparent driver, since it's also correlated with purpose)

purpose_base = pd.Series(purpose).map(
    {"business": 0.2, "travel": 0.1, "education": 0.25, "medical": 0.15}
).to_numpy()  # small honest baseline differences, not the bias itself

# Legitimate, non-biased age effect: officers weight "settled" applicants
# (established career/family ties) as lower flight-risk. This is a genuine
# U-shaped curve, not a bias to be discovered - just a real feature that
# should influence the prediction.
age_score = -0.0018 * (age - 45) ** 2 + 0.9  # peaks around age 45, tapers at extremes

# --- THE HIDDEN BIAS TERM ---
# Strong when prior_approvals is low, decays as prior_approvals grows.
decay = np.exp(-prior_approvals / 2.5)  # ~1.0 at 0, ~0.09 at 8+
purpose_bias = np.where(
    purpose == "business", 1.8 * decay,
    np.where(purpose == "medical", -1.2 * decay, 0.0)
)

score = (
    0.55 * np.log1p(annual_income / 100000)
    + 0.45 * np.log1p(bank_balance / 100000)
    + edu_score
    + 0.30 * prior_approvals
    + eng_score
    + purpose_base
    + age_score
    + purpose_bias            # <- hidden bias injected here
    - 3.6                     # intercept, centers approval rate
)

prob = 1 / (1 + np.exp(-score))
approved = (prob >= 0.5).astype(int)

# Hard gate: legal charges -> deterministic reject
approved = np.where(legal_charges == "yes", 0, approved)

# ---------------------------------------------------------------------
# 3. Assemble dataframe
# ---------------------------------------------------------------------

df = pd.DataFrame({
    "age": age,
    "annual_income_inr": annual_income.astype(int),
    "bank_balance_inr": bank_balance.astype(int),
    "education_level": education_level,
    "purpose_of_visit": purpose,
    "prior_approvals": prior_approvals,
    "english_level": english_level,
    "legal_charges": legal_charges,
    "visa_approved": approved,
})

df.to_csv("visa_dataset.csv", index=False)

print(df.head(10).to_string())
print("\nShape:", df.shape)
print("\nApproval rate overall:", df.visa_approved.mean().round(3))
print("\nApproval rate by purpose:")
print(df.groupby("purpose_of_visit").visa_approved.mean().round(3))
print("\nApproval rate by purpose, prior_approvals==0 only:")
print(df[df.prior_approvals == 0].groupby("purpose_of_visit").visa_approved.mean().round(3))
print("\nApproval rate by purpose, prior_approvals>=3 only:")
print(df[df.prior_approvals >= 3].groupby("purpose_of_visit").visa_approved.mean().round(3))

print("\n--- Sanity: every field should move approval rate ---")
print("\nBy age bucket:")
print(df.assign(age_bucket=pd.cut(df.age, [17, 30, 45, 60, 80]))
        .groupby("age_bucket", observed=True).visa_approved.mean().round(3))
print("\nBy income quartile:")
print(df.assign(q=pd.qcut(df.annual_income_inr, 4))
        .groupby("q", observed=True).visa_approved.mean().round(3))
print("\nBy bank balance quartile:")
print(df.assign(q=pd.qcut(df.bank_balance_inr, 4))
        .groupby("q", observed=True).visa_approved.mean().round(3))
print("\nBy education level:")
print(df.groupby("education_level").visa_approved.mean().round(3))
print("\nBy prior_approvals:")
print(df.groupby("prior_approvals").visa_approved.mean().round(3))
print("\nBy english_level:")
print(df.groupby("english_level").visa_approved.mean().round(3))
print("\nBy legal_charges:")
print(df.groupby("legal_charges").visa_approved.mean().round(3))
