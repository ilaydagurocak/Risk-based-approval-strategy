from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ayarlar

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
EPS = 1e-9

# Veri boyutu
N_TX = 120_000
FRAUD_RATE = 0.004  # ~%0.4
N_CARDS = 25_000
N_MERCHANTS = 3_000
N_DEVICES = 35_000
TIME_SPAN_DAYS = 30

# alarm inceleme limiti
MAX_ALERTS_PER_DAY = 500


@dataclass
class Costs:
    fn: float = 500.0
    fp: float = 15.0
    review: float = 1.5

COSTS = Costs(fn=500.0, fp=15.0, review=1.5)

# Hibrit politika
AUTO_BLOCK_RULES = True          
MODEL_FOR_REST = True            
USE_TOPK_CAPACITY = True         

# PSI yorum eşiği
PSI_WARN = 0.10
PSI_ALERT = 0.25

def make_fraud_transactions(
    n_tx: int,
    fraud_rate: float,
    n_cards: int,
    n_merchants: int,
    n_devices: int,
    time_span_days: int,
    drift=False,
    seed=42,
):
    rng = np.random.default_rng(seed)

    total_seconds = time_span_days * 24 * 3600
    ts = rng.integers(0, total_seconds, size=n_tx)
    ts.sort()

    card_id = rng.integers(1, n_cards + 1, size=n_tx)
    merchant_id = rng.integers(1, n_merchants + 1, size=n_tx)

    # device: karta bağlı + bazen switch
    primary_device = rng.integers(1, n_devices + 1, size=n_cards + 1)
    device_id = primary_device[card_id].copy()
    switch = rng.random(n_tx) < 0.08
    device_id[switch] = rng.integers(1, n_devices + 1, size=switch.sum())

    country = rng.choice(
        ["TR", "DE", "NL", "US", "GB", "AE", "RU"],
        p=[0.78, 0.05, 0.03, 0.05, 0.04, 0.03, 0.02],
        size=n_tx,
    )
    channel = rng.choice(["pos", "ecom", "atm"], p=[0.70, 0.25, 0.05], size=n_tx)
    mcc = rng.choice(
        ["grocery", "fuel", "electronics", "travel", "luxury", "gaming", "restaurant", "digital_goods"],
        p=[0.22, 0.10, 0.10, 0.06, 0.06, 0.08, 0.30, 0.08],
        size=n_tx,
    )

    # Amount
    base_amount = rng.lognormal(mean=3.4, sigma=0.8, size=n_tx)
    mcc_mult = np.where(mcc == "travel", 2.0, 1.0)
    mcc_mult = np.where(mcc == "luxury", 2.5, mcc_mult)
    mcc_mult = np.where(mcc == "electronics", 1.8, mcc_mult)
    amount = np.clip(base_amount * mcc_mult, 1, 20_000)

    # Drift simülasyonu: son dönem ecom + high amount + TR dışı biraz artsın
    if drift:
        drift_mask = rng.random(n_tx) < 0.25
        channel = np.where(drift_mask, rng.choice(["ecom", "pos"], p=[0.55, 0.45], size=n_tx), channel)
        country = np.where(drift_mask, rng.choice(["TR", "US", "GB", "DE"], p=[0.65, 0.15, 0.10, 0.10], size=n_tx), country)
        amount = np.where(drift_mask, np.clip(amount * rng.normal(1.10, 0.10, size=n_tx), 1, 20_000), amount)

    df = pd.DataFrame({
        "ts": ts,
        "card_id": card_id,
        "merchant_id": merchant_id,
        "device_id": device_id,
        "country": country,
        "channel": channel,
        "mcc": mcc,
        "amount": amount,
        "device_switched": switch.astype(int),
    })

    # Fraud olasılığı 
    amount_z = (df["amount"] - df["amount"].mean()) / (df["amount"].std() + 1e-9)
    risk = (
        -6.5
        + 0.9 * (df["channel"] == "ecom").astype(int)
        + 0.7 * (df["country"] != "TR").astype(int)
        + 0.8 * df["mcc"].isin(["digital_goods", "gaming"]).astype(int)
        + 0.6 * df["mcc"].isin(["luxury", "travel"]).astype(int)
        + 0.6 * (amount_z > 1.5).astype(int)
        + 0.5 * df["device_switched"].astype(int)
    )
    pd_true = 1 / (1 + np.exp(-risk))

    # hedef fraud_rate’e ölçekledim
    scale = fraud_rate / (pd_true.mean() + 1e-12)
    pd_true = np.clip(pd_true * scale, 0, 0.95)

    df["fraud"] = rng.binomial(1, pd_true).astype(int)
    return df


# Ana veri + “current” driftli veri (monitoring için)
df = make_fraud_transactions(
    n_tx=N_TX,
    fraud_rate=FRAUD_RATE,
    n_cards=N_CARDS,
    n_merchants=N_MERCHANTS,
    n_devices=N_DEVICES,
    time_span_days=TIME_SPAN_DAYS,
    drift=False,
    seed=RANDOM_STATE,
)
current_df = make_fraud_transactions(
    n_tx=int(N_TX * 0.35),
    fraud_rate=FRAUD_RATE * 1.2,  
    n_cards=N_CARDS,
    n_merchants=N_MERCHANTS,
    n_devices=N_DEVICES,
    time_span_days=TIME_SPAN_DAYS,
    drift=True,
    seed=RANDOM_STATE + 1,
)

print("Rows:", len(df), "| Fraud rate:", round(df["fraud"].mean(), 5))
print("Current Rows:", len(current_df), "| Current Fraud rate:", round(current_df["fraud"].mean(), 5))


def add_basic_time_features(df):
    out = df.sort_values("ts").reset_index(drop=True).copy()
    out["hour"] = (out["ts"] // 3600) % 24
    out["day"] = out["ts"] // (24 * 3600)
    # card bazında device değişimi
    out["prev_device"] = out.groupby("card_id")["device_id"].shift(1)
    out["device_changed"] = (out["device_id"] != out["prev_device"]).astype(int)
    out.loc[out["prev_device"].isna(), "device_changed"] = 0
    out = out.drop(columns=["prev_device"])
    return out

def add_time_window_features(df, key="card_id", time_col="ts"):
    """
    Basit sliding window:
    - last 10m tx count
    - last 60m tx count
    - last 60m amount sum
    """
    out = df.copy()
    out["tx_count_10m"] = 0
    out["tx_count_60m"] = 0
    out["amt_sum_60m"] = 0.0

    for cid, g in out.groupby(key, sort=False):
        ts = g[time_col].values
        amt = g["amount"].values

        c10 = np.zeros(len(g), dtype=int)
        c60 = np.zeros(len(g), dtype=int)
        s60 = np.zeros(len(g), dtype=float)

        j10 = 0
        j60 = 0
        sum60 = 0.0

        for i in range(len(g)):
            t = ts[i]
            while ts[j60] < t - 3600:
                sum60 -= amt[j60]
                j60 += 1
            while ts[j10] < t - 600:
                j10 += 1

            c60[i] = i - j60
            c10[i] = i - j10
            s60[i] = sum60
            sum60 += amt[i]

        out.loc[g.index, "tx_count_10m"] = c10
        out.loc[g.index, "tx_count_60m"] = c60
        out.loc[g.index, "amt_sum_60m"] = s60

    return out

df = add_basic_time_features(df)
df = add_time_window_features(df)

current_df = add_basic_time_features(current_df)
current_df = add_time_window_features(current_df)

print("\nFeature snapshot:")
print(df[["amount", "tx_count_10m", "tx_count_60m", "amt_sum_60m", "device_changed"]].head(3))

# kuralların uygulanması

def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bankada fraud genelde:
    - sert kurallar (hard rules) + model skoru (soft)
    Burada rule_flag=1 ise yüksek risk pre-screen yapıyoruz.
    """
    out = df.copy()
    # örnek kurallar:
    rule1 = (out["channel"] == "ecom") & (out["country"] != "TR") & (out["amount"] >= 2500)
    rule2 = (out["tx_count_10m"] >= 4) & (out["amount"] >= 800)
    rule3 = (out["device_changed"] == 1) & out["mcc"].isin(["digital_goods", "gaming"]) & (out["amount"] >= 600)

    out["rule_flag"] = (rule1 | rule2 | rule3).astype(int)

    # rule_score: çok basit puan 
    out["rule_score"] = (
        2.0 * rule1.astype(float)
        + 1.5 * rule2.astype(float)
        + 1.2 * rule3.astype(float)
    )
    return out

df = apply_rules(df)
current_df = apply_rules(current_df)

print("\nRule flag rate:", round(df["rule_flag"].mean(), 4), "| rule-flag fraud rate:", round(df.loc[df["rule_flag"]==1, "fraud"].mean() if (df["rule_flag"]==1).any() else 0, 4))


cut_ts = df["ts"].quantile(0.80)
train_df = df[df["ts"] <= cut_ts].copy()
test_df = df[df["ts"] > cut_ts].copy()

test_days = int(test_df["day"].nunique())
K_ALERTS = MAX_ALERTS_PER_DAY * max(test_days, 1)

print("\nTrain rows:", len(train_df), "Test rows:", len(test_df))
print("Train fraud rate:", round(train_df["fraud"].mean(), 5))
print("Test  fraud rate:", round(test_df["fraud"].mean(), 5))
print("Test days:", test_days, "| Max alerts total (K):", K_ALERTS)


# MODEL Kurallar feature olarak dahil ettiö

target = "fraud"
num_cols = ["amount", "hour", "tx_count_10m", "tx_count_60m", "amt_sum_60m", "device_changed", "rule_score"]
cat_cols = ["country", "channel", "mcc"]

X_train = train_df[num_cols + cat_cols]
y_train = train_df[target].values

X_test = test_df[num_cols + cat_cols]
y_test = test_df[target].values

preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

clf = LogisticRegression(max_iter=2000, class_weight="balanced")
pipe = Pipeline(steps=[("prep", preprocess), ("clf", clf)])
pipe.fit(X_train, y_train)

proba_test = pipe.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, proba_test)
pr_auc = average_precision_score(y_test, proba_test)
print("\n=== Fraud Model Metrics (test) ===")
print("ROC-AUC:", round(roc_auc, 4))
print("PR-AUC :", round(pr_auc, 4))


# THRESHOLD stratejileri

def eval_at_threshold(y_true, proba, thr):
    y_pred = (proba >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    return {
        "thr": float(thr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

prec, rec, thr_arr = precision_recall_curve(y_test, proba_test)
f1_arr = 2 * (prec * rec) / (prec + rec + EPS)
best_idx = int(np.argmax(f1_arr))
thr_f1 = float(thr_arr[max(best_idx - 1, 0)]) if len(thr_arr) else 0.5

print("\nBest threshold (approx F1-max):", round(thr_f1, 6))
print(eval_at_threshold(y_test, proba_test, thr_f1))
print("\n--- Classification @ F1-max ---")
print("Confusion matrix:\n", confusion_matrix(y_test, (proba_test >= thr_f1).astype(int)))
print(classification_report(y_test, (proba_test >= thr_f1).astype(int), digits=4))

def expected_cost_at_threshold(y_true, proba, thr, costs: Costs):
    y_pred = (proba >= thr).astype(int)  # 1=alert
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Alarm sayısı = tp+fp => inceleme maliyeti
    alerts = tp + fp

    # Beklenen maliyet: kaçırılan fraud (fn) + yanlış alarm (fp) + inceleme maliyeti(alerts)
    cost = costs.fn * fn + costs.fp * fp + costs.review * alerts
    return float(cost), int(alerts), int(tp), int(fp), int(fn), int(tn)

# Cost-based threshold taraması
grid = np.unique(np.quantile(proba_test, np.linspace(0.01, 0.99, 200)))
rows = []
for t in grid:
    cost, alerts, tp, fp, fn, tn = expected_cost_at_threshold(y_test, proba_test, t, COSTS)
    rows.append([t, cost, alerts, tp, fp, fn, tn])
cost_df = pd.DataFrame(rows, columns=["thr", "total_cost", "alerts", "tp", "fp", "fn", "tn"])
best_cost = cost_df.iloc[cost_df["total_cost"].idxmin()].to_dict()
thr_cost = float(best_cost["thr"])

print("\n=== Cost-based threshold ===")
print("Costs:", COSTS)
print("Best thr (min cost):", round(thr_cost, 6))
print("Min total cost:", round(best_cost["total_cost"], 2), "| alerts:", int(best_cost["alerts"]),
      "| tp:", int(best_cost["tp"]), "| fp:", int(best_cost["fp"]), "| fn:", int(best_cost["fn"]))

# Top-K (kapasite)
K = min(K_ALERTS, len(test_df))
topk_idx = np.argsort(-proba_test)[:K]
fraud_total = int(y_test.sum())
fraud_caught = int(y_test[topk_idx].sum())
recall_at_k = fraud_caught / (fraud_total + EPS) if fraud_total > 0 else 0.0
precision_at_k = fraud_caught / (K + EPS)

print("\n=== Top-K capacity strategy ===")
print(f"Max alerts/day={MAX_ALERTS_PER_DAY} | test_days={test_days} | K={K}")
print("Fraud total:", fraud_total, "| Fraud caught:", fraud_caught)
print("Recall@K   :", round(float(recall_at_k), 4))
print("Precision@K:", round(float(precision_at_k), 4))

# hibrit karar stratejisi
test_out = test_df.reset_index(drop=True).copy()
test_out["proba_ml"] = proba_test

# Başlangıç: hepsi PASS
test_out["decision"] = "PASS"

# Kural tetiklenenler
if AUTO_BLOCK_RULES:
    rule_alert_idx = test_out.index[test_out["rule_flag"] == 1].to_numpy()
else:
    rule_alert_idx = np.array([], dtype=int)

# Kalanlar için ML
remaining_idx = test_out.index[test_out["rule_flag"] == 0].to_numpy()

if MODEL_FOR_REST:
    if USE_TOPK_CAPACITY:
        # kapasiteyi: önce rule alerts, kalan kapasiteyi ML Top-K ile dolduruyoruz
        capacity_left = max(K - len(rule_alert_idx), 0)

        # remaining içinde en yüksek proba'ları seçiyoruz
        rem_sorted = remaining_idx[np.argsort(-test_out.loc[remaining_idx, "proba_ml"].values)]
        ml_alert_idx = rem_sorted[:capacity_left]
    else:
        # eşik: cost-based veya f1-based seçiyo
        chosen_thr = thr_cost
        ml_alert_idx = remaining_idx[test_out.loc[remaining_idx, "proba_ml"].values >= chosen_thr]
else:
    ml_alert_idx = np.array([], dtype=int)

final_alert_idx = np.unique(np.concatenate([rule_alert_idx, ml_alert_idx]))
test_out.loc[final_alert_idx, "decision"] = "ALERT"

# Hibrit performans:
alerts = (test_out["decision"] == "ALERT").values.astype(int)
cm_hybrid = confusion_matrix(test_out["fraud"].values, alerts)
tn, fp, fn, tp = cm_hybrid.ravel()
precision_h = tp / (tp + fp + EPS)
recall_h = tp / (tp + fn + EPS)
print("\n=== HYBRID (Rules + ML) Performance on test ===")
print("Alerts:", int(alerts.sum()), "| TP:", int(tp), "| FP:", int(fp), "| FN:", int(fn), "| TN:", int(tn))
print("Precision:", round(float(precision_h), 4), "| Recall:", round(float(recall_h), 4))

# ALARM olan ilk 5 işlem için en yüksek katkı yapan feature'ları yazdırıyor

def get_feature_names(pipe: Pipeline, num_cols, cat_cols):
    prep: ColumnTransformer = pipe.named_steps["prep"]
    # num passthrough
    num_features = list(num_cols)
    # cat onehot
    ohe: OneHotEncoder = prep.named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
    return num_features + cat_feature_names

feature_names = get_feature_names(pipe, num_cols, cat_cols)
coefs = pipe.named_steps["clf"].coef_.ravel()
coef_map = dict(zip(feature_names, coefs))

def explain_row(pipe, row_df: pd.DataFrame, topn=8):
    # row_df tek satır DF olmalı
    prep = pipe.named_steps["prep"]
    x = prep.transform(row_df)  # (1, n_features)
    x = np.asarray(x).ravel()
    contrib = x * coefs
    idx = np.argsort(-np.abs(contrib))[:topn]
    items = [(feature_names[i], float(x[i]), float(contrib[i])) for i in idx]
    return items

print("\n=== Explainability (top alerts) ===")
alert_rows = test_out.index[test_out["decision"] == "ALERT"].to_numpy()
for i in alert_rows[:5]:
    row = test_out.loc[[i], num_cols + cat_cols]
    items = explain_row(pipe, row, topn=8)
    print(f"\nTx #{i} | fraud={int(test_out.loc[i,'fraud'])} | proba={test_out.loc[i,'proba_ml']:.4f} | rule_flag={int(test_out.loc[i,'rule_flag'])}")
    for name, val, c in items:
        # val: transformed value (onehot 0/1 or numeric)
        print(f" - {name}: value={val:.4f} | contrib={c:.4f}")


def psi(expected, actual, bins=10):
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, qs))
    if len(edges) < 3:
        return 0.0

    e_bin = pd.cut(expected, bins=edges, include_lowest=True)
    a_bin = pd.cut(actual, bins=edges, include_lowest=True)

    e_counts = e_bin.value_counts().sort_index()
    a_counts = a_bin.value_counts().sort_index()

    e_dist = (e_counts / e_counts.sum()).values + EPS
    a_dist = (a_counts / a_counts.sum()).values + EPS

    return float(np.sum((a_dist - e_dist) * np.log(a_dist / e_dist)))

psi_features = ["amount", "hour", "tx_count_10m", "tx_count_60m", "amt_sum_60m", "device_changed", "rule_score"]
psi_rows = []
for f in psi_features:
    val = psi(train_df[f].values, current_df[f].values, bins=10)
    psi_rows.append((f, val))

psi_df = pd.DataFrame(psi_rows, columns=["feature", "PSI"]).sort_values("PSI", ascending=False)
print("\n=== PSI (train vs current) ===")
print(psi_df.to_string(index=False))

print("\nPSI yorum:")
print(f" - < {PSI_WARN:.2f}: stabil")
print(f" - {PSI_WARN:.2f}–{PSI_ALERT:.2f}: izlenmeli")
print(f" - > {PSI_ALERT:.2f}: drift alarmı")


# ÇIKTI 

show_cols = [
    "ts", "day", "amount", "country", "channel", "mcc",
    "tx_count_10m", "tx_count_60m", "amt_sum_60m",
    "device_changed", "rule_flag", "rule_score",
    "fraud", "proba_ml", "decision"
]
print("\n=== Output preview ===")
print(test_out[show_cols].head(15))
