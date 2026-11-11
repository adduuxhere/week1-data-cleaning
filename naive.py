"""
Naive baseline: drop rows with missing values (and duplicates), no other cleaning.

Rebuilds the same corrupted dataset as pipeline.py (same seed, same order of
random draws). Unit-scale and sign errors are deliberately NOT corrected here.
Output: naive_results.json
"""
import json
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

# ---- Rebuild the corrupted dataset (identical to pipeline.py) ----
rng = np.random.default_rng(SEED)
b = load_breast_cancer(as_frame=True)
raw = b.frame.drop(columns="target").copy()
feats = list(raw.columns)

miss_cols = ["mean texture", "mean smoothness", "area error",
             "worst concavity", "fractal dimension error", "mean compactness"]
mask = pd.DataFrame(False, index=raw.index, columns=feats)
for c in miss_cols:
    mask[c] = rng.random(len(raw)) < 0.08
raw = raw.mask(mask)

err_idx = rng.choice(len(raw), 10, replace=False)
raw.loc[err_idx, "mean area"] = raw.loc[err_idx, "mean area"] * 1000

neg_idx = rng.choice(len(raw), 5, replace=False)
raw.loc[neg_idx, "mean radius"] = -raw.loc[neg_idx, "mean radius"]

dup = raw.sample(6, random_state=1)
raw = pd.concat([raw, dup], ignore_index=True)

# Labels follow the same row order: original 569 rows, then the 6 duplicated rows
labels = np.concatenate([b.target, b.target[dup.index]])

# ---- Naive cleaning: duplicates + drop rows with any missing value ----
df = raw.copy()
df["label"] = labels
df = df.drop_duplicates(subset=feats + ["label"]).dropna()

X = df[feats].values
y = df["label"].values

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
cv_acc = float(cross_val_score(model, X, y, cv=5).mean())

print("rows used:", len(df))
print("5-fold CV accuracy:", round(cv_acc, 4))

with open("naive_results.json", "w") as f:
    json.dump({"naive_dropna_rows": len(df), "cv_acc_naive": round(cv_acc, 4)}, f, indent=2)