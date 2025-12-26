

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")



@dataclass
class Config:
    random_state: int = 42
    n_customers: int = 12000
    artifact_dir: str = "artifacts_bank_full"
    n_clusters_min: int = 4
    n_clusters_max: int = 9
    n_clusters_default: int = 6
    propensity_models: Tuple[str, ...] = ("logreg", "hgb")  # bank-friendly options
    test_size: float = 0.25
    score_sample_n: int = 20
    save_scored_sample_csv: bool = True
    enable_shap_optional: bool = False   # requires shap installed
    shap_sample_n: int = 800

CFG = Config()


PRODUCT_TARGETS = [
    "target_credit_card",
    "target_personal_loan",
    "target_investment",
    "target_bes",
]




def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def clip(a, lo, hi):
    return np.clip(a, lo, hi)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0




def generate_synthetic_bank_data(n: int, seed: int) -> pd.DataFrame:
    """
    Generates realistic-ish retail banking behavior:
    - income/spend/balances
    - tenure, region, channel
    - card usage, cash advances, late payments
    - creates product targets with correlated "ground truth"
    - injects missingness and mild outliers
    """
    rng = np.random.default_rng(seed)

    customer_id = np.arange(1, n + 1)

    region = rng.choice(
        ["Marmara", "Ege", "IcAnadolu", "Akdeniz", "Karadeniz", "Dogu", "Guneydogu"],
        size=n,
        p=[0.26, 0.12, 0.18, 0.10, 0.12, 0.12, 0.10],
    )

    channel = rng.choice(["mobile", "branch", "web", "callcenter"], size=n, p=[0.58, 0.20, 0.15, 0.07])


    age = rng.integers(18, 71, size=n)
    marital_status = rng.choice(["single", "married"], size=n, p=[0.45, 0.55])
    education = rng.choice(["high_school", "bachelor", "master_plus"], size=n, p=[0.40, 0.45, 0.15])

   
    tenure_months = rng.integers(1, 241, size=n)


    base_income = rng.lognormal(mean=10.7, sigma=0.55, size=n) 
    monthly_income = clip(base_income, 9000, 250000).round(0)

  
    spend_ratio = (
        0.40
        + 0.10 * (age < 30)
        + 0.05 * (education == "bachelor")
        + 0.08 * (channel == "mobile")
        + rng.normal(0, 0.12, size=n)
    )
    spend_ratio = clip(spend_ratio, 0.05, 1.25)

    monthly_spend = clip(monthly_income * spend_ratio + rng.normal(0, 2000, size=n), 500, 180000).round(0)

    
    savings = clip(rng.lognormal(mean=11.0, sigma=0.95, size=n) - 40000, 0, 3_000_000).round(0)
    investment_balance = clip(rng.lognormal(mean=10.2, sigma=1.05, size=n) - 30000, 0, 5_000_000).round(0)

   
    credit_card_spend = clip(monthly_spend * rng.uniform(0.15, 0.85, size=n) + rng.normal(0, 1500, size=n), 0, 150000).round(0)

    
    cash_advance_cnt = rng.poisson(0.35, size=n)
    cash_advance_cnt += ((monthly_spend > monthly_income * 0.85).astype(int) * rng.poisson(0.7, size=n))
    cash_advance_cnt += ((credit_card_spend > 25000).astype(int) * rng.poisson(0.4, size=n))
    cash_advance_cnt = clip(cash_advance_cnt, 0, 18)

    
    late_payment_base = rng.poisson(0.22, size=n)
    late_payment_base += ((monthly_spend > monthly_income * 0.9).astype(int) * rng.poisson(0.6, size=n))
    late_payment_base += ((cash_advance_cnt >= 2).astype(int) * rng.poisson(0.5, size=n))
    late_payment_cnt = clip(late_payment_base, 0, 12)

    
    products_count = rng.integers(1, 6, size=n)
    products_count += (tenure_months > 36).astype(int)
    products_count += (monthly_income > 70000).astype(int)
    products_count = clip(products_count, 1, 9)

    
    salary_customer = (sigmoid(-0.6 + 0.015 * tenure_months + 0.45 * (channel == "branch").astype(int)) > rng.random(n)).astype(int)

  
    p_cc = sigmoid(
        -1.2
        + 0.8 * (channel == "mobile").astype(int)
        + 0.3 * (channel == "web").astype(int)
        + 0.00002 * credit_card_spend
        + 0.35 * salary_customer
        + 0.25 * (age < 40).astype(int)
        - 0.35 * (late_payment_cnt >= 3).astype(int)
    )
    p_cc = clip(p_cc, 0.01, 0.92)

   
    util = monthly_spend / clip(monthly_income, 1, 10**9)
    p_loan = sigmoid(
        -2.0
        + 2.0 * (util > 0.85).astype(int)
        + 0.35 * (cash_advance_cnt >= 2).astype(int)
        + 0.20 * (late_payment_cnt >= 2).astype(int)
        + 0.20 * (monthly_income > 25000).astype(int)
        - 0.25 * (monthly_income > 150000).astype(int)
    )
    p_loan = clip(p_loan, 0.01, 0.75)

    p_inv = sigmoid(
        -2.1
        + 0.55 * (monthly_income > 75000).astype(int)
        + 0.40 * (savings > 150000).astype(int)
        + 0.60 * (investment_balance > 80000).astype(int)
        + 0.20 * (age > 40).astype(int)
        - 0.35 * (late_payment_cnt >= 2).astype(int)
        + 0.15 * (education == "master_plus").astype(int)
    )
    p_inv = clip(p_inv, 0.01, 0.88)


    p_bes = sigmoid(
        -2.0
        + 0.012 * tenure_months
        + 0.55 * salary_customer
        + 0.35 * ((age >= 28) & (age <= 55)).astype(int)
        + 0.20 * (monthly_income > 35000).astype(int)
        - 0.20 * (late_payment_cnt >= 3).astype(int)
    )
    p_bes = clip(p_bes, 0.01, 0.86)

    target_credit_card = rng.binomial(1, p_cc)
    target_personal_loan = rng.binomial(1, p_loan)
    target_investment = rng.binomial(1, p_inv)
    target_bes = rng.binomial(1, p_bes)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "region": region,
        "channel": channel,
        "age": age,
        "marital_status": marital_status,
        "education": education,
        "tenure_months": tenure_months,
        "salary_customer": salary_customer,
        "monthly_income": monthly_income,
        "monthly_spend": monthly_spend,
        "credit_card_spend": credit_card_spend,
        "cash_advance_cnt": cash_advance_cnt,
        "late_payment_cnt": late_payment_cnt,
        "savings": savings,
        "investment_balance": investment_balance,
        "products_count": products_count,
        "target_credit_card": target_credit_card,
        "target_personal_loan": target_personal_loan,
        "target_investment": target_investment,
        "target_bes": target_bes,
    })

    
    for col, rate in [
        ("education", 0.03),
        ("marital_status", 0.02),
        ("investment_balance", 0.02),
        ("savings", 0.01),
    ]:
        mask = rng.random(n) < rate
        df.loc[mask, col] = np.nan

    
    out_mask = rng.random(n) < 0.005
    df.loc[out_mask, "monthly_income"] = (df.loc[out_mask, "monthly_income"] * rng.uniform(2.0, 4.0, out_mask.sum())).clip(9000, 250000)
    df.loc[out_mask, "monthly_spend"] = (df.loc[out_mask, "monthly_spend"] * rng.uniform(2.0, 4.0, out_mask.sum())).clip(500, 180000)

    return df




@dataclass
class FeatureSpec:
    id_col: str
    numeric_cols: List[str]
    categorical_cols: List[str]
    target_cols: List[str]

def build_feature_spec(df: pd.DataFrame) -> FeatureSpec:
    id_col = "customer_id"
    target_cols = [c for c in PRODUCT_TARGETS if c in df.columns]

    
    base = [c for c in df.columns if c not in ([id_col] + target_cols)]

    numeric_cols = [c for c in base if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in base if c not in numeric_cols]

    return FeatureSpec(id_col=id_col, numeric_cols=numeric_cols, categorical_cols=categorical_cols, target_cols=target_cols)

def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )




def choose_k_via_inertia(X: np.ndarray, k_min: int, k_max: int, seed: int) -> Tuple[int, pd.DataFrame]:
    """
    A simple, robust heuristic: compute inertia over a range and pick elbow-ish:
    pick k where relative improvement drops below a threshold.
    (Silhouette is heavier + needs distances; inertia is common for KMeans.)
    """
    rows = []
    prev_inertia = None
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        km.fit(X)
        inertia = float(km.inertia_)
        if prev_inertia is None:
            rel_impr = np.nan
        else:
            rel_impr = safe_div(prev_inertia - inertia, prev_inertia)
        rows.append({"k": k, "inertia": inertia, "relative_improvement": rel_impr})
        prev_inertia = inertia

    table = pd.DataFrame(rows)

    
    chosen = None
    for _, r in table.iterrows():
        if not np.isnan(r["relative_improvement"]) and r["relative_improvement"] < 0.08 and r["k"] >= (k_min + 2):
            chosen = int(r["k"])
            break
    if chosen is None:
        chosen = CFG.n_clusters_default

    return chosen, table

def train_segmentation(df: pd.DataFrame, spec: FeatureSpec, seed: int) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    feature_cols = spec.numeric_cols + spec.categorical_cols
    prep = build_preprocessor(spec.numeric_cols, spec.categorical_cols)

    X = prep.fit_transform(df[feature_cols])

    k, inertia_table = choose_k_via_inertia(X, CFG.n_clusters_min, CFG.n_clusters_max, seed)

    print_section("SEGMENTATION: K SELECTION (INERTIA)")
    print(inertia_table)

    seg_pipe = Pipeline(steps=[
        ("prep", prep),
        ("kmeans", KMeans(n_clusters=k, random_state=seed, n_init="auto"))
    ])

    segments = seg_pipe.fit_predict(df[feature_cols])
    out = df.copy()
    out["segment"] = segments
    return seg_pipe, out, inertia_table

def segment_profiles(df_seg: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    
    prof = df_seg.groupby("segment")[spec.numeric_cols].mean(numeric_only=True)

   
    for c in spec.categorical_cols:
        prof[f"mode_{c}"] = df_seg.groupby("segment")[c].agg(lambda x: x.value_counts().index[0])

    
    def persona(row):
        inc = row.get("monthly_income", np.nan)
        spend = row.get("monthly_spend", np.nan)
        sav = row.get("savings", np.nan)
        inv = row.get("investment_balance", np.nan)
        late = row.get("late_payment_cnt", 0)

        if pd.notna(inv) and inv > 150000:
            return "Yatırım Odaklı / Varlıklı"
        if pd.notna(sav) and sav > 250000 and (pd.isna(spend) or spend < 25000):
            return "Birikimci / Düşük Harcama"
        if pd.notna(spend) and pd.notna(inc) and spend > inc * 0.88:
            return "Yüksek Harcama / Kredi İhtimali"
        if pd.notna(inc) and inc > 90000 and (pd.isna(late) or late < 1):
            return "Premium / Yüksek Gelir"
        if pd.notna(late) and late >= 2:
            return "Riskli / Gecikmeli"
        return "Standart Perakende"

    prof = prof.copy()
    prof["persona"] = prof.apply(persona, axis=1)

    return prof.round(2).sort_index()




@dataclass
class ModelMetrics:
    target: str
    model_name: str
    roc_auc: float
    pr_auc: float
    positive_rate: float
    best_f1_threshold: float
    best_f1: float

def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    
    f1s = []
    ths = []
    for i, t in enumerate(thresholds):
        p = precision[i + 1]
        r = recall[i + 1]
        f1 = safe_div(2 * p * r, (p + r))
        f1s.append(f1)
        ths.append(t)
    if not f1s:
        return 0.5, 0.0
    idx = int(np.argmax(f1s))
    return float(ths[idx]), float(f1s[idx])

def decile_lift_table(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    Classic bank marketing report:
    Sort by score desc, split into deciles, compute:
    - count, events, event_rate
    - cumulative capture rate
    - lift vs overall rate
    """
    df = pd.DataFrame({"y": y_true, "score": y_score}).sort_values("score", ascending=False).reset_index(drop=True)

    overall_rate = df["y"].mean()
    df["bucket"] = pd.qcut(df.index + 1, q=n_bins, labels=False) 

    rows = []
    cum_events = 0
    total_events = df["y"].sum()

    for b in range(n_bins):
        part = df[df["bucket"] == b]
        cnt = len(part)
        events = int(part["y"].sum())
        rate = safe_div(events, cnt)
        cum_events += events
        cum_capture = safe_div(cum_events, total_events) if total_events > 0 else 0.0
        lift = safe_div(rate, overall_rate) if overall_rate > 0 else 0.0
        rows.append({
            "decile": b + 1,
            "count": cnt,
            "events": events,
            "event_rate": rate,
            "cum_capture_rate": cum_capture,
            "lift": lift
        })

    out = pd.DataFrame(rows)
    out["overall_event_rate"] = overall_rate
    return out

def build_propensity_preprocessor(spec: FeatureSpec) -> Tuple[List[str], List[str], ColumnTransformer]:
    """
    For propensity we include segment as a numeric feature.
    """
    numeric_cols = spec.numeric_cols + ["segment"]
    categorical_cols = spec.categorical_cols
    prep = build_preprocessor(numeric_cols, categorical_cols)
    return numeric_cols, categorical_cols, prep

def train_propensity_for_target(
    df_seg: pd.DataFrame,
    spec: FeatureSpec,
    target: str,
    model_name: str,
    seed: int,
) -> Tuple[Pipeline, ModelMetrics, pd.DataFrame]:
    """
    Train one propensity model (logreg or histgradientboosting) and produce metrics + decile table.
    """
    numeric_cols, categorical_cols, prep = build_propensity_preprocessor(spec)
    feature_cols = numeric_cols + categorical_cols

    X = df_seg[feature_cols].copy()
    y = df_seg[target].astype(int).values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=CFG.test_size, random_state=seed, stratify=y
    )

    if model_name == "logreg":
        clf = LogisticRegression(
            max_iter=600,
            class_weight="balanced",
            solver="lbfgs",
        )
    elif model_name == "hgb":
        
        clf = HistGradientBoostingClassifier(
            max_depth=5,
            learning_rate=0.08,
            max_iter=250,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    pipe = Pipeline(steps=[
        ("prep", prep),
        ("clf", clf),
    ])

    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]

    auc = float(roc_auc_score(y_te, proba)) if len(np.unique(y_te)) > 1 else float("nan")
    pr_auc = float(average_precision_score(y_te, proba)) if len(np.unique(y_te)) > 1 else float("nan")

    th, f1 = best_f1_threshold(y_te, proba)
    pos_rate = float(y.mean())

    metrics = ModelMetrics(
        target=target,
        model_name=model_name,
        roc_auc=auc,
        pr_auc=pr_auc,
        positive_rate=pos_rate,
        best_f1_threshold=th,
        best_f1=f1,
    )

    deciles = decile_lift_table(y_te, proba, n_bins=10)
    return pipe, metrics, deciles

def train_all_propensity_models(
    df_seg: pd.DataFrame,
    spec: FeatureSpec,
    seed: int,
) -> Tuple[Dict[str, Pipeline], pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    For each target, try candidate models and pick best by ROC-AUC (fallback PR-AUC).
    Returns:
      - chosen_models dict (target -> Pipeline)
      - metrics table
      - decile tables dict (target -> decile_df for chosen model)
    """
    chosen_models: Dict[str, Pipeline] = {}
    metrics_rows = []
    decile_reports: Dict[str, pd.DataFrame] = {}

    for target in spec.target_cols:
        print_section(f"PROPENSITY TRAINING: {target}")
        best = None  # (score, metrics, model, deciles)

        for model_name in CFG.propensity_models:
            model, m, deciles = train_propensity_for_target(df_seg, spec, target, model_name, seed)

            print(f"- Candidate: {model_name} | AUC={m.roc_auc:.4f} | PR-AUC={m.pr_auc:.4f} | pos_rate={m.positive_rate:.3f} | bestF1={m.best_f1:.3f} @th={m.best_f1_threshold:.3f}")

            # pick best by AUC; if nan, use PR-AUC
            primary = m.roc_auc if not np.isnan(m.roc_auc) else -1.0
            secondary = m.pr_auc if not np.isnan(m.pr_auc) else -1.0
            score = (primary, secondary)

            if best is None or score > best[0]:
                best = (score, m, model, deciles)

        assert best is not None
        _, m_best, model_best, deciles_best = best

        chosen_models[target] = model_best
        metrics_rows.append(asdict(m_best))
        decile_reports[target] = deciles_best

        print("\nChosen model:", m_best.model_name)
        print("Decile/Lift (test set):")
        print(deciles_best.round(4))

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["target", "roc_auc"], ascending=[True, False])
    return chosen_models, metrics_df, decile_reports




def save_artifacts(
    cfg: Config,
    spec: FeatureSpec,
    seg_pipe: Pipeline,
    prof: pd.DataFrame,
    inertia_table: pd.DataFrame,
    propensity_models: Dict[str, Pipeline],
    metrics: pd.DataFrame,
    decile_reports: Dict[str, pd.DataFrame],
) -> None:
    ensure_dir(cfg.artifact_dir)

    joblib.dump(seg_pipe, os.path.join(cfg.artifact_dir, "segmentation_pipe.joblib"))
    prof.to_csv(os.path.join(cfg.artifact_dir, "segment_profiles.csv"), index=True)
    inertia_table.to_csv(os.path.join(cfg.artifact_dir, "kmeans_inertia_table.csv"), index=False)

    for tgt, model in propensity_models.items():
        joblib.dump(model, os.path.join(cfg.artifact_dir, f"propensity_{tgt}.joblib"))

    metrics.to_csv(os.path.join(cfg.artifact_dir, "propensity_metrics.csv"), index=False)

    for tgt, dec in decile_reports.items():
        dec.to_csv(os.path.join(cfg.artifact_dir, f"decile_lift_{tgt}.csv"), index=False)

    meta = {
        "config": asdict(cfg),
        "feature_spec": asdict(spec),
        "targets": list(propensity_models.keys()),
    }
    with open(os.path.join(cfg.artifact_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_artifacts(artifact_dir: str) -> Tuple[FeatureSpec, Pipeline, pd.DataFrame, Dict[str, Pipeline]]:
    with open(os.path.join(artifact_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    spec_d = meta["feature_spec"]
    spec = FeatureSpec(
        id_col=spec_d["id_col"],
        numeric_cols=spec_d["numeric_cols"],
        categorical_cols=spec_d["categorical_cols"],
        target_cols=spec_d["target_cols"],
    )

    seg_pipe = joblib.load(os.path.join(artifact_dir, "segmentation_pipe.joblib"))
    prof = pd.read_csv(os.path.join(artifact_dir, "segment_profiles.csv"), index_col=0)

    models = {}
    for tgt in meta["targets"]:
        models[tgt] = joblib.load(os.path.join(artifact_dir, f"propensity_{tgt}.joblib"))

    return spec, seg_pipe, prof, models




def score_new_customers(
    new_df: pd.DataFrame,
    spec: FeatureSpec,
    seg_pipe: Pipeline,
    propensity_models: Dict[str, Pipeline],
) -> pd.DataFrame:
    """
    Input: new customers WITHOUT targets (targets can exist; will be ignored)
    Output: segment + product probabilities + recommended offer
    """
    df = new_df.copy()

    
    base_features = spec.numeric_cols + spec.categorical_cols
    df["segment"] = seg_pipe.predict(df[base_features])

   
    model_features = spec.numeric_cols + ["segment"] + spec.categorical_cols

    for tgt, model in propensity_models.items():
        df[f"p_{tgt.replace('target_', '')}"] = model.predict_proba(df[model_features])[:, 1]

    p_cols = [c for c in df.columns if c.startswith("p_")]
    df["recommended_offer"] = df[p_cols].idxmax(axis=1).str.replace("p_", "", regex=False)

    return df



def try_shap_explain(pipe: Pipeline, X_sample: pd.DataFrame, top_n: int = 15) -> Optional[pd.DataFrame]:
    """
    Returns top features by mean(|SHAP|) for explainability.
    Works best for linear model; for HGB it may be slower/heavier.
    """
    try:
        import shap  
    except Exception:
        print("SHAP not installed. Skipping explainability. (pip install shap)")
        return None

    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]

    Xp = prep.transform(X_sample)
    feature_names = prep.get_feature_names_out()

    
    if isinstance(clf, LogisticRegression):
        explainer = shap.LinearExplainer(clf, Xp)
        shap_vals = explainer.shap_values(Xp)
    else:
        
        explainer = shap.Explainer(clf, Xp)
        shap_vals = explainer(Xp).values

    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return imp.sort_values("mean_abs_shap", ascending=False).head(top_n)




def main():
    print_section("STEP 1: GENERATE SYNTHETIC BANK DATA")
    df = generate_synthetic_bank_data(CFG.n_customers, CFG.random_state)
    print(df.head(5))
    print("\nShape:", df.shape)
    print("\nTarget rates:")
    for t in PRODUCT_TARGETS:
        print(f"  {t}: {df[t].mean():.3f}")

    spec = build_feature_spec(df)

    print_section("FEATURE SPEC")
    print("Numeric cols:", spec.numeric_cols)
    print("Categorical cols:", spec.categorical_cols)
    print("Targets:", spec.target_cols)

    print_section("STEP 2: SEGMENTATION (KMEANS)")
    seg_pipe, df_seg, inertia_table = train_segmentation(df, spec, CFG.random_state)
    prof = segment_profiles(df_seg, spec)
    print("\nSegment profiles (head):")
    print(prof.head(20))

    # Add segments distribution
    print("\nSegment distribution:")
    print(df_seg["segment"].value_counts(normalize=True).sort_index().round(4))

    print_section("STEP 3: PROPENSITY MODELING (ONE MODEL PER PRODUCT)")
    propensity_models, metrics_df, decile_reports = train_all_propensity_models(df_seg, spec, CFG.random_state)
    print_section("PROPENSITY METRICS SUMMARY (CHOSEN MODELS)")
    print(metrics_df.round(4))

    print_section("STEP 4: SAVE ARTIFACTS")
    ensure_dir(CFG.artifact_dir)
    save_artifacts(CFG, spec, seg_pipe, prof, inertia_table, propensity_models, metrics_df, decile_reports)
    print(f"Saved artifacts under: {CFG.artifact_dir}/")

    
    if CFG.enable_shap_optional:
        print_section("STEP 5 (OPTIONAL): SHAP EXPLAINABILITY")
        # explain one target (e.g., investment)
        explain_target = "target_investment" if "target_investment" in propensity_models else list(propensity_models.keys())[0]
        sample = df_seg.sample(min(CFG.shap_sample_n, len(df_seg)), random_state=CFG.random_state)
        model_features = spec.numeric_cols + ["segment"] + spec.categorical_cols
        imp = try_shap_explain(propensity_models[explain_target], sample[model_features], top_n=15)
        if imp is not None:
            print(f"\nTop SHAP features for {explain_target}:")
            print(imp)

    print_section("STEP 6: SCORE 'NEW' CUSTOMERS (PROD-LIKE)")
   
    new_customers = df_seg.drop(columns=spec.target_cols).sample(CFG.score_sample_n, random_state=CFG.random_state).reset_index(drop=True)
    scored = score_new_customers(new_customers, spec, seg_pipe, propensity_models)

    pcols = [c for c in scored.columns if c.startswith("p_")]
    show_cols = [spec.id_col, "segment"] + pcols + ["recommended_offer"]
    print(scored[show_cols].head(10).round(4))

    if CFG.save_scored_sample_csv:
        path = os.path.join(CFG.artifact_dir, "scored_new_customers_sample.csv")
        scored.to_csv(path, index=False)
        print(f"\nSaved scored sample: {path}")

    print_section("DONE ✅")
    print("Bu çıktı bankalarda CRM hedefleme akışına çok yakındır:")
    print("- Segment -> persona")
    print("- Ürün bazlı propensity -> decile/lift raporu")
    print("- En yüksek olasılığa göre teklif önerisi")
    print("- Artifact kaydı (model/pipeline) -> prod skorlamaya hazır")


if __name__ == "__main__":
    main()
