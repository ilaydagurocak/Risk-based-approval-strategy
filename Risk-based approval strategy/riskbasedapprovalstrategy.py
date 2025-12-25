import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

# ayarlar
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
EPS = 1e-9

TARGET = "default"
FEATURES_NUM = ["income", "utilization", "delinq_12m", "tenure", "bureau_score"]

# scorecard parametreleri
PDO = 20
BaseScore = 600
BaseOdds = 50


# elimde gerçek veri olmadığından sentetik veri üretiyorum
def make_synthetic(n=25000, drift=False):
    income = np.random.normal(45000, 18000, n)
    utilization = np.random.beta(2, 5, n)
    delinq_12m = np.random.choice([0, 1, 2, 3], p=[0.75, 0.17, 0.06, 0.02], size=n)
    tenure = np.random.randint(0, 180, n)

    if drift:
        income = income * np.random.normal(0.98, 0.02, n)
        utilization = np.clip(utilization + np.random.normal(0.03, 0.02, n), 0, 1)
        delinq_12m = np.clip(delinq_12m + np.random.choice([0, 1], p=[0.85, 0.15], size=n), 0, 3)

    bureau_score = np.clip(
        720 - 120 * utilization - 90 * delinq_12m + np.random.normal(0, 30, n),
        300, 900
    )

    # Temerrüt olasılığı üretimi 
    logit = (
        -4
        + 2.3 * utilization
        + 0.9 * delinq_12m
        - 0.005 * bureau_score
        - 0.002 * tenure
    )
    pd_true = 1 / (1 + np.exp(-logit))
    default = np.random.binomial(1, pd_true)

    return pd.DataFrame({
        "income": income,
        "utilization": utilization,
        "delinq_12m": delinq_12m,
        "tenure": tenure,
        "bureau_score": bureau_score,
        "default": default
    })

df = make_synthetic(n=26000, drift=False)
current_df = make_synthetic(n=12000, drift=True)

train_df, test_df = train_test_split(df, test_size=0.30, random_state=RANDOM_STATE, stratify=df[TARGET])

print("Train default rate:", round(train_df[TARGET].mean(), 4))
print("Test  default rate:", round(test_df[TARGET].mean(), 4))
print("Curr  default rate:", round(current_df[TARGET].mean(), 4))


# MONOTONİK BINNING + WOE/IV

def _make_initial_bins(x: pd.Series, q=10):
    return pd.qcut(x, q=q, duplicates="drop")

def _bin_stats(df, binned, target):
    tmp = df[[target]].copy()
    tmp["bin"] = binned
    g = tmp.groupby("bin")[target].agg(["count", "sum"]).rename(columns={"sum": "bad"})
    g["good"] = g["count"] - g["bad"]
    g["bad_rate"] = g["bad"] / (g["count"] + EPS)
    return g.reset_index()

def _is_monotonic(arr, increasing=True):
    diffs = np.diff(arr)
    return np.all(diffs >= -1e-12) if increasing else np.all(diffs <= 1e-12)

def monotonic_binning(df, feature, target, q=10, increasing=None, min_bin_size=0.03):
    x = df[feature]
    binned = _make_initial_bins(x, q=q)
    stats = _bin_stats(df, binned, target).sort_values("bin").reset_index(drop=True)
    br = stats["bad_rate"].values

    if increasing is None:
        inc = np.sum(np.diff(br) >= 0)
        dec = np.sum(np.diff(br) <= 0)
        increasing = inc >= dec

    intervals = list(stats["bin"])
    edges = sorted(set([iv.left for iv in intervals] + [intervals[-1].right]))
    edges = [edges[0]] + [e for e in edges[1:]]

    def make_cut(edges_):
        return pd.cut(x, bins=edges_, include_lowest=True, duplicates="drop")

    for _ in range(300):
        b2 = make_cut(edges)
        s2 = _bin_stats(df, b2, target).sort_values("bin").reset_index(drop=True)

        total = s2["count"].sum()
        too_small = (s2["count"] / total) < min_bin_size
        br2 = s2["bad_rate"].values
        mono_ok = _is_monotonic(br2, increasing=increasing)

        if mono_ok and not too_small.any():
            return b2, s2, increasing

        if too_small.any():
            idx = int(np.where(too_small)[0][0])
        else:
            diffs = np.diff(br2)
            viol = np.where(diffs < 0)[0] if increasing else np.where(diffs > 0)[0]
            idx = int(viol[0] + 1) if len(viol) else 0

        if len(edges) <= 3:
            return make_cut(edges), s2, increasing

        if idx == 0:
            remove_edge_pos = 1
        elif idx >= (len(edges) - 2):
            remove_edge_pos = len(edges) - 2
        else:
            left_gap = abs(br2[idx] - br2[idx - 1])
            right_gap = abs(br2[idx] - br2[idx + 1])
            remove_edge_pos = idx if left_gap <= right_gap else (idx + 1)

        edges.pop(remove_edge_pos)

    b_final = make_cut(edges)
    s_final = _bin_stats(df, b_final, target).sort_values("bin").reset_index(drop=True)
    return b_final, s_final, increasing

def woe_table_from_stats(stats_df):
    total_good = stats_df["good"].sum()
    total_bad = stats_df["bad"].sum()

    out = stats_df.copy()
    out["dist_good"] = out["good"] / (total_good + EPS)
    out["dist_bad"] = out["bad"] / (total_bad + EPS)
    out["woe"] = np.log((out["dist_good"] + EPS) / (out["dist_bad"] + EPS))
    out["iv_bin"] = (out["dist_good"] - out["dist_bad"]) * out["woe"]
    iv = out["iv_bin"].sum()
    return out, iv

binning_artifacts = {}
iv_rows = []
for f in FEATURES_NUM:
    binned, stats2, inc = monotonic_binning(train_df, f, TARGET, q=10, increasing=None, min_bin_size=0.03)
    woe_tbl, iv = woe_table_from_stats(stats2)
    binning_artifacts[f] = {"woe_table": woe_tbl, "increasing": inc}
    iv_rows.append([f, iv])

iv_df = pd.DataFrame(iv_rows, columns=["feature", "IV"]).sort_values("IV", ascending=False)
print("\n=== IV Summary (train) ===")
print(iv_df.to_string(index=False))

def apply_woe_numeric(df, feature, woe_tbl):
    intervals = woe_tbl["bin"].tolist()
    woe_map = dict(zip(intervals, woe_tbl["woe"]))
    edges = [intervals[0].left] + [iv.right for iv in intervals]
    b = pd.cut(df[feature], bins=edges, include_lowest=True)
    return b.map(woe_map).astype(float)

def build_woe_matrix(df, artifacts):
    X = pd.DataFrame(index=df.index)
    for f, art in artifacts.items():
        X[f] = apply_woe_numeric(df, f, art["woe_table"])
    return X.fillna(0.0)

X_train = build_woe_matrix(train_df, binning_artifacts)
X_test  = build_woe_matrix(test_df, binning_artifacts)
X_curr  = build_woe_matrix(current_df, binning_artifacts)

y_train = train_df[TARGET].values
y_test  = test_df[TARGET].values
y_curr  = current_df[TARGET].values


#  MODEL (WOE -> Logistic) + Kalibrasyon

base_model = LogisticRegression(max_iter=2000)
base_model.fit(X_train, y_train)

cal_model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
cal_model.fit(X_train, y_train)

pd_test = cal_model.predict_proba(X_test)[:, 1]
pd_curr = cal_model.predict_proba(X_curr)[:, 1]

auc = roc_auc_score(y_test, pd_test)
fpr, tpr, thr = roc_curve(y_test, pd_test)
ks = np.max(tpr - fpr)
ks_thr = thr[np.argmax(tpr - fpr)]

print("\n=== Model Performance (test) ===")
print("AUC:", round(auc, 4))
print("KS :", round(ks, 4), "| KS-max thr:", round(float(ks_thr), 6))

# scorecard 
Factor = PDO / np.log(2)
Offset = BaseScore - Factor * np.log(BaseOdds)
def pd_to_score(pd):
    pd = np.clip(pd, 1e-6, 1 - 1e-6)
    odds = (1 - pd) / pd
    return Offset + Factor * np.log(odds)
score_test = pd_to_score(pd_test)



def attach_pricing_vars(df):
    
    ead = np.clip(
        (df["income"] * 0.25) * (0.6 + 0.8*df["utilization"]) * (900 - df["bureau_score"]) / 600,
        500, 50000
    )

    
    lgd = np.clip(0.35 + 0.10*df["utilization"] + 0.05*(df["delinq_12m"]>0), 0.2, 0.9)

    
    tenor_years = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15], size=len(df))

    
    interest_rate = np.clip(0.12 + 0.55*pd_test[:len(df)], 0.10, 0.55)

    funding_rate = 0.08

    
    op_cost = 50.0

    out = df.copy()
    out["EAD"] = ead
    out["LGD"] = lgd
    out["TENOR_Y"] = tenor_years
    out["INT_RATE"] = interest_rate
    out["FUND_RATE"] = funding_rate
    out["OP_COST"] = op_cost
    return out


test_prc = attach_pricing_vars(test_df.reset_index(drop=True))
test_prc["PD_MODEL"] = pd_test
test_prc["SCORE"] = score_test
test_prc["Y_TRUE"] = y_test

def expected_profit(row):
    pdm = row["PD_MODEL"]
    ead = row["EAD"]
    lgd = row["LGD"]
    t = row["TENOR_Y"]
    r = row["INT_RATE"]
    f = row["FUND_RATE"]
    op = row["OP_COST"]

    revenue = (r - f) * ead * t
    el = pdm * lgd * ead
    return revenue - el - op

test_prc["EXP_PROFIT"] = test_prc.apply(expected_profit, axis=1)


def profit_at_threshold(df_prc, thr):
    approved = df_prc["PD_MODEL"] < thr
    total_profit = df_prc.loc[approved, "EXP_PROFIT"].sum()
    approval_rate = approved.mean()

    
    bad_rate_approved = df_prc.loc[approved, "Y_TRUE"].mean() if approved.any() else np.nan
    return float(total_profit), float(approval_rate), float(bad_rate_approved)


grid = np.quantile(test_prc["PD_MODEL"], np.linspace(0.01, 0.99, 99))
rows = []
for t in grid:
    totp, appr, badr = profit_at_threshold(test_prc, t)
    rows.append([t, totp, appr, badr])

perf = pd.DataFrame(rows, columns=["thr", "total_profit", "approval_rate", "bad_rate_approved"])
best = perf.iloc[perf["total_profit"].idxmax()].to_dict()

best_thr = float(best["thr"])
print("\n=== PROFIT OPTIMIZATION (test) ===")
print("Best threshold:", round(best_thr, 6))
print("Max total profit:", round(best["total_profit"], 2))
print("Approval rate @ best:", round(best["approval_rate"], 4))
print("Bad rate among approved @ best:", round(best["bad_rate_approved"], 4))


y_pred = (test_prc["PD_MODEL"].values >= best_thr).astype(int)  # 1=RED riskli, 0=ONAY
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix [[TN,FP],[FN,TP]]:\n", cm)
print(classification_report(y_test, y_pred, digits=4))


coefs = base_model.coef_.ravel()
feature_names = X_train.columns.tolist()
coef_map = dict(zip(feature_names, coefs))

def reason_codes_for_row(x_woe_row: pd.Series, topn=3):
    contrib = {f: coef_map[f] * float(x_woe_row[f]) for f in feature_names}
    sorted_feats = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
    top = [(f, v) for f, v in sorted_feats if v > 0][:topn]
    return top

def human_reason(feature, value, woe_tbl):
    intervals = woe_tbl["bin"].tolist()
    edges = [intervals[0].left] + [iv.right for iv in intervals]
    b = pd.cut(pd.Series([value]), bins=edges, include_lowest=True).iloc[0]
    row = woe_tbl[woe_tbl["bin"] == b].iloc[0]
    return f"{feature}: bin={str(b)}, bad_rate~{row['bad_rate']:.3f}, woe={row['woe']:.3f}"

print("\n=== Reason Codes (sample REJECT cases @ profit-optimal thr) ===")
reject_mask = test_prc["PD_MODEL"] >= best_thr
reject_idx = np.where(reject_mask.values)[0]
if len(reject_idx) == 0:
    print("Bu threshold ile REJECT yok.")
else:
    sample = reject_idx[:5]
    for i in sample:
        raw = test_df.reset_index(drop=True).iloc[i]
        woe_row = X_test.reset_index(drop=True).iloc[i]
        top = reason_codes_for_row(woe_row, topn=3)
        print(f"\nCase #{i} | PD={test_prc.loc[i,'PD_MODEL']:.4f} | Score={test_prc.loc[i,'SCORE']:.1f} | ExpProfit(approve)={test_prc.loc[i,'EXP_PROFIT']:.2f}")
        for f, c in top:
            txt = human_reason(f, raw[f], binning_artifacts[f]["woe_table"])
            print(f" - {txt} | contribution={c:.4f}")


def psi_for_feature(train_series, curr_series, woe_tbl):
    intervals = woe_tbl["bin"].tolist()
    edges = [intervals[0].left] + [iv.right for iv in intervals]
    tr_bin = pd.cut(train_series, bins=edges, include_lowest=True)
    cu_bin = pd.cut(curr_series, bins=edges, include_lowest=True)

    tr_dist = tr_bin.value_counts(normalize=True).reindex(intervals).fillna(0.0).values + EPS
    cu_dist = cu_bin.value_counts(normalize=True).reindex(intervals).fillna(0.0).values + EPS
    return float(np.sum((cu_dist - tr_dist) * np.log(cu_dist / tr_dist)))

psi_rows = []
for f in FEATURES_NUM:
    psi_val = psi_for_feature(train_df[f], current_df[f], binning_artifacts[f]["woe_table"])
    psi_rows.append([f, psi_val])

psi_df = pd.DataFrame(psi_rows, columns=["feature", "PSI"]).sort_values("PSI", ascending=False)
print("\n=== PSI (train vs current) ===")
print(psi_df.to_string(index=False))
print("\nPSI kuralı (genel): <0.10 stabil | 0.10-0.25 izlenmeli | >0.25 drift")


# final output
out = test_prc.copy()
out["DECISION"] = np.where(out["PD_MODEL"] < best_thr, "APPROVE", "REJECT")

print("\n=== Final Output Preview ===")
print(out[["PD_MODEL","SCORE","EAD","LGD","TENOR_Y","INT_RATE","EXP_PROFIT","DECISION","Y_TRUE"]].head(10))
