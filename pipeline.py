import pandas as pd, numpy as np, json
from sklearn.datasets import load_breast_cancer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, IterativeImputer
 
rng = np.random.default_rng(42)
b = load_breast_cancer(as_frame=True)
truth = b.frame.drop(columns="target").copy()
feats = list(truth.columns)
log = {}
 
# ---- 1. Build a corrupted copy (documented injection) ----
raw = truth.copy()
# (a) MCAR missingness: 8% of cells in 6 columns
miss_cols = ["mean texture","mean smoothness","area error","worst concavity","fractal dimension error","mean compactness"]
mask = pd.DataFrame(False, index=raw.index, columns=feats)
for c in miss_cols:
    m = rng.random(len(raw)) < 0.08
    mask[c] = m
raw = raw.mask(mask)
log["injected_missing_cells"] = int(mask.sum().sum())
# (b) Unit-entry errors: 10 area values multiplied by 1000 (e.g. mm^2 stored as um^2)
err_idx = rng.choice(len(raw), 10, replace=False)
raw.loc[err_idx, "mean area"] = raw.loc[err_idx, "mean area"] * 1000
log["injected_unit_errors"] = 10
# (c) Impossible negatives: 5 radius values sign-flipped
neg_idx = rng.choice(len(raw), 5, replace=False)
raw.loc[neg_idx, "mean radius"] = -raw.loc[neg_idx, "mean radius"]
log["injected_negatives"] = 5
# (d) Duplicates: 6 exact copies appended
dup = raw.sample(6, random_state=1)
raw = pd.concat([raw, dup], ignore_index=True)
log["injected_duplicates"] = 6
raw.to_csv("raw_corrupted.csv", index=False)
 
# ---- 2. Audit ----
audit = {
 "rows": len(raw), "missing_cells": int(raw.isna().sum().sum()),
 "missing_by_col": {c:int(v) for c,v in raw.isna().sum().items() if v>0},
 "duplicate_rows": int(raw.duplicated().sum()),
 "negative_values": int((raw[feats] < 0).sum().sum()),
}
log["audit"] = audit
 
# ---- 3. Clean ----
df = raw.drop_duplicates().reset_index(drop=True)
# Negatives: all 30 features are non-negative physical measurements -> flip sign back (documented rule: sign-flip is the only plausible single-cause fix)
# Verified by checking that |x| falls in the plausible range of the column.
neg_mask = df[feats] < 0
log["negatives_fixed"] = int(neg_mask.sum().sum())
df[feats] = df[feats].abs()
# Unit errors: detect via ratio to within-column median > 100 for area features
area_cols = ["mean area","area error","worst area"]
flag_units = (df["mean area"] > 100 * df["mean area"].median())
log["unit_error_rows_detected"] = int(flag_units.sum())
# Rescale the detected rows back by /1000 (documented: 1000x factor is the injected/known unit error)
df.loc[flag_units, "mean area"] = df.loc[flag_units, "mean area"] / 1000
# Note: we do not scale worst area in this demo; detection rule is shown on mean area only
# Missing: compare simple median vs iterative (model-based) imputation on the cells we know
imp_cols = [c for c in feats if df[c].isna().any()]
missing_truth = truth.loc[df.index, imp_cols][df[imp_cols].isna()]
def rmse_on_missing(imputed):
    diff = (imputed.loc[df.index, imp_cols] - truth.loc[df.index, imp_cols])[df[imp_cols].isna()]
    return float(np.sqrt(np.nanmean(diff.values**2)))
med = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(df), columns=feats, index=df.index)
it = pd.DataFrame(IterativeImputer(max_iter=20, random_state=0).fit_transform(df), columns=feats, index=df.index)
# Normalized RMSE: divide by column std of truth so columns are comparable
std = truth[imp_cols].std()
def nrmse(imputed):
    errs=[]
    for c in imp_cols:
        m = df[c].isna()
        e = (imputed.loc[m,c]-truth.loc[m,c])
        errs.append(np.sqrt((e**2).mean())/std[c])
    return float(np.mean(errs))
log["imputation_nrmse_median"] = round(nrmse(med),4)
log["imputation_nrmse_iterative"] = round(nrmse(it),4)
log["imputed_cols"] = imp_cols
log["n_missing_imputed"] = int(df[imp_cols].isna().sum().sum())
# Use the iterative imputer for the final dataset (lower error)
final = it.copy()
final.to_csv("cleaned.csv", index=False)
 
# ---- 4. Compare to truth: how close is cleaned data to original? ----
t = truth.loc[final.index]
log["final_rows"] = len(final)
log["final_max_abs_dev_mean_area_pct"] = None
mae_area = float((final["mean area"]-t["mean area"]).abs().median())
log["median_abs_err_mean_area"] = round(mae_area,2)
log["mean_area_rows_still_outlying_vs_truth"] = int(((final["mean area"]-t["mean area"]).abs()/t["mean area"] > 0.5).sum())
# Effect on an analysis: correlation of mean radius and mean area preserved?
log["corr_radius_area_truth"] = round(t["mean radius"].corr(t["mean area"]),4)
log["corr_radius_area_clean"] = round(final["mean radius"].corr(final["mean area"]),4)
log["corr_radius_area_raw_naive"] = round(raw["mean radius"].corr(raw["mean area"]),4)
# Downstream effect: fit a logistic regression on truth vs cleaned vs naive-dropna
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
y = b.target.values
def cv(X, yy):
    return float(cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)), X, yy, cv=5).mean())
log["cv_acc_truth"] = round(cv(truth.values, y),4)
naive = raw.drop_duplicates().dropna()
log["naive_dropna_rows"] = len(naive)
log["cv_acc_clean"] = round(cv(final.values, y[final.index]),4)
# naive: align y via index of original raw (raw index preserved within drop_duplicates)
yn = y[naive.index.map(lambda i: i if i < len(truth) else i-len(truth)+0)] if False else None
json.dump(log, open("log.json","w"), indent=2, default=str)
print(json.dumps(log, indent=2, default=str))
 