import pandas as pd, numpy as np
from sklearn.datasets import load_breast_cancer
b = load_breast_cancer(as_frame=True)
df = b.frame.copy()
df["diagnosis"] = b.target_names[b.target]
print("shape", df.shape)
print("dtypes:", df.dtypes.value_counts().to_dict())
print("missing total:", int(df.isna().sum().sum()))
print("duplicates:", int(df.duplicated().sum()))
print(df["diagnosis"].value_counts().to_dict())
feats = b.feature_names
d = df[feats]
print("min<=0 cols:", [c for c in feats if (d[c] <= 0).any()])
print("zero-var:", [c for c in feats if d[c].nunique()==1])
q1, q3 = d.quantile(.25), d.quantile(.75); iqr = q3-q1
out = ((d < q1-1.5*iqr) | (d > q3+1.5*iqr))
print("IQR outlier cells:", int(out.sum().sum()), "rows with >=1:", int(out.any(axis=1).sum()))
print((out.sum()).sort_values(ascending=False).head(6).to_dict())
# consistency: mean radius vs worst radius, area ~ radius^2 check
print("worst>=mean radius violations:", int((df["worst radius"] < df["mean radius"]).sum()))
print("worst>=mean area violations:", int((df["worst area"] < df["mean area"]).sum()))
print("corr radius-perimeter:", round(d["mean radius"].corr(d["mean perimeter"]),3))
print("corr radius-area:", round(d["mean radius"].corr(d["mean area"]),3))
print(d[["mean radius","mean area","mean concavity","worst area"]].describe().round(2).to_string())
print("max abs skew top:", d.skew().abs().sort_values(ascending=False).head(4).round(2).to_dict())
 