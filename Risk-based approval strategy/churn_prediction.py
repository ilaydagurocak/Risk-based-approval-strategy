import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
EPS = 1e-9

N_CUSTOMERS = 30000
CHURN_RATE = 0.18   


def make_churn_data(n=30000, churn_rate=0.18, seed=42):
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(1, 120, size=n)
    age = rng.integers(18, 75, size=n)
    income = rng.normal(45000, 15000, size=n)
    balance = rng.lognormal(mean=8.5, sigma=1.0, size=n)
    num_products = rng.integers(1, 6, size=n)

    card_active = rng.choice([0, 1], p=[0.25, 0.75], size=n)
    app_active = rng.choice([0, 1], p=[0.20, 0.80], size=n)

    region = rng.choice(
        ["TR_West", "TR_Central", "TR_East"],
        p=[0.55, 0.30, 0.15],
        size=n
    )

    complaints_6m = rng.poisson(lam=0.3, size=n)
    tx_count_3m = rng.poisson(lam=25, size=n)

    
    risk = (
        -1.8
        + 0.9 * (tenure_months < 12)
        + 0.7 * (tx_count_3m < 5)
        + 0.8 * (app_active == 0)
        + 0.6 * (complaints_6m >= 2)
        + 0.5 * (balance < 1000)
        - 0.3 * (num_products >= 3)
    )

    pd_true = 1 / (1 + np.exp(-risk))
    scale = churn_rate / (pd_true.mean() + EPS)
    pd_true = np.clip(pd_true * scale, 0, 0.95)

    churn = rng.binomial(1, pd_true)

    return pd.DataFrame({
        "tenure_months": tenure_months,
        "age": age,
        "income": income,
        "balance": balance,
        "num_products": num_products,
        "card_active": card_active,
        "app_active": app_active,
        "complaints_6m": complaints_6m,
        "tx_count_3m": tx_count_3m,
        "region": region,
        "churn": churn,
    })

df = make_churn_data(N_CUSTOMERS, CHURN_RATE, RANDOM_STATE)
print("Rows:", len(df))
print("Churn rate:", round(df["churn"].mean(), 4))


target = "churn"
X = df.drop(columns=[target])
y = df[target].values

num_cols = [
    "tenure_months", "age", "income", "balance",
    "num_products", "complaints_6m", "tx_count_3m"
]
bin_cols = ["card_active", "app_active"]
cat_cols = ["region"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("bin", "passthrough", bin_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)


model = LogisticRegression(max_iter=2000, class_weight="balanced")

pipe = Pipeline(steps=[
    ("prep", preprocess),
    ("model", model),
])

pipe.fit(X_train, y_train)

proba_test = pipe.predict_proba(X_test)[:, 1]


auc = roc_auc_score(y_test, proba_test)
fpr, tpr, thr = roc_curve(y_test, proba_test)
ks = np.max(tpr - fpr)

print("\n=== Churn Model Performance ===")
print("ROC-AUC:", round(auc, 4))
print("KS     :", round(ks, 4))


y_pred = (proba_test >= 0.5).astype(int)
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))


TOP_RATE = 0.20  
cutoff = np.quantile(proba_test, 1 - TOP_RATE)

targeted = proba_test >= cutoff
recall_at_k = (y_test[targeted].sum() / (y_test.sum() + EPS))
precision_at_k = y_test[targeted].mean()

print("\n=== Retention Campaign Strategy ===")
print("Target rate:", TOP_RATE)
print("Recall@TopK:", round(recall_at_k, 4))
print("Precision@TopK:", round(precision_at_k, 4))


def get_feature_names(pipe, num_cols, bin_cols, cat_cols):
    prep = pipe.named_steps["prep"]
    names = []
    names += num_cols
    names += bin_cols
    ohe = prep.named_transformers_["cat"]
    names += list(ohe.get_feature_names_out(cat_cols))
    return names

feature_names = get_feature_names(pipe, num_cols, bin_cols, cat_cols)
coefs = pipe.named_steps["model"].coef_.ravel()

coef_df = (
    pd.DataFrame({"feature": feature_names, "coef": coefs})
    .sort_values("coef", ascending=False)
)

print("\n=== Top churn risk drivers ===")
print(coef_df.head(10))

print("\n=== Churn reducing factors ===")
print(coef_df.tail(10))


out = X_test.copy()
out["churn_proba"] = proba_test
out["decision"] = np.where(out["churn_proba"] >= cutoff, "TARGET_CAMPAIGN", "NO_ACTION")
out["churn_true"] = y_test

print("\n=== Output preview ===")
print(out.head(10))
