"""
02_log_transform_and_scale.py

Step 2 of the respiratory disease classification project.
  Phase 1: import cleaned.csv and log2-transform the metabolite columns
  Phase 2: autoscale (z-score) the log-transformed metabolites

Notes
  - Only metabolite columns are transformed/scaled. ID, batch, class, Sex
    and Age are passed through untouched.
  - Missing values stay missing (NaN). StandardScaler ignores them when
    computing means/SDs and leaves them as NaN. Impute later.
  - The scaled file is for EXPLORATION (PCA, plots). For model evaluation,
    redo imputation + scaling inside a scikit-learn Pipeline so they are fit
    on training folds only (otherwise test data leaks into the scaling).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ------------------------------ CONFIG ----------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent      # use .parent.parent if script is in scripts/
IN_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "results"

ID_COL = "Sample.Name"
BATCH_COL = "Batch.Number"
LABEL_COL = "Class.name"
META_COLS = ["Sex", "Age"]          # passed through unchanged
# ------------------------------------------------------------------------


# ===== PHASE 1: IMPORT AND LOG TRANSFORM =================================
print("=" * 60 + "\nPHASE 1: IMPORT AND LOG2 TRANSFORM\n" + "=" * 60)

df = pd.read_csv(IN_PATH)
print(f"Loaded {IN_PATH}: {df.shape[0]} samples x {df.shape[1]} columns")

non_met = [ID_COL, BATCH_COL, LABEL_COL] + META_COLS
missing_cols = [c for c in non_met if c not in df.columns]
if missing_cols:
    raise ValueError(f"Expected columns not found: {missing_cols}")

metabolites = [c for c in df.columns if c not in non_met]
print(f"Metabolite columns: {len(metabolites)}")

# All metabolite columns must be numeric by now (script 1 converted them)
not_numeric = [c for c in metabolites if not pd.api.types.is_numeric_dtype(df[c])]
if not_numeric:
    raise TypeError(f"Non-numeric metabolite columns (re-run script 1): {not_numeric}")

X = df[metabolites]
print(f"Missing metabolite cells before transform (kept as NaN): {int(X.isna().sum().sum())}")

# Log needs strictly positive values. Script 1 already treated '< 0' text as
# "below range" -> NaN, so any remaining zero/negative numbers are handled the
# same way (set to NaN, imputed later) rather than inventing a replacement value.
nonpos = X <= 0
n_nonpos = int(nonpos.sum().sum())
print(f"Zero or negative values: {n_nonpos}")
if n_nonpos > 0:
    for c in X.columns[nonpos.any()]:
        print(f"  {c}: {int(nonpos[c].sum())} value(s) <= 0 (min {X[c].min():.4g}) -> set to NaN")
    X = X.mask(nonpos)

X_log = np.log2(X)

# Skewness before vs after (closer to 0 = more symmetric)
skew_before = X.skew()
skew_after = X_log.skew()
print(f"\nMedian skewness before log: {skew_before.median():.2f}   after log2: {skew_after.median():.2f}")
print(f"Metabolites with |skew| > 1: before {int((skew_before.abs() > 1).sum())}, "
      f"after {int((skew_after.abs() > 1).sum())}")

# Save log-transformed data (useful for interpreting effect sizes)
df_log = pd.concat([df[non_met], X_log], axis=1)
OUT_DIR.mkdir(parents=True, exist_ok=True)
df_log.to_csv(OUT_DIR / "log2_transformed.csv", index=False)
print(f"Saved {OUT_DIR / 'log2_transformed.csv'}")


# ===== PHASE 2: AUTOSCALE (Z-SCORE) ======================================
print("\n" + "=" * 60 + "\nPHASE 2: AUTOSCALE (Z-SCORE)\n" + "=" * 60)

scaler = StandardScaler()                       # (x - mean) / SD per metabolite, NaNs ignored
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_log), columns=metabolites, index=df.index
)

# Sanity check: every metabolite should now have mean ~0 and SD ~1
means = X_scaled.mean()
sds = X_scaled.std(ddof=0)
print(f"Max |mean| across metabolites: {means.abs().max():.2e}")
print(f"SD range across metabolites:   {sds.min():.3f} to {sds.max():.3f}")
print(f"Missing cells still NaN: {int(X_scaled.isna().sum().sum())}")

df_scaled = pd.concat([df[non_met], X_scaled], axis=1)
df_scaled.to_csv(OUT_DIR / "scaled_for_exploration.csv", index=False)

# Save scaling parameters so the transform is documented / reproducible
pd.DataFrame({"mean_log2": scaler.mean_, "sd_log2": scaler.scale_},
             index=metabolites).to_csv(OUT_DIR / "scaling_parameters.csv")
print(f"Saved scaled_for_exploration.csv and scaling_parameters.csv to {OUT_DIR}/")


# ===== QUICK DIAGNOSTIC PLOT =============================================
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(skew_before.dropna(), bins=30, color="tab:red", alpha=0.8)
axes[0].set_title("Skewness per metabolite: raw")
axes[1].hist(skew_after.dropna(), bins=30, color="tab:blue", alpha=0.8)
axes[1].set_title("Skewness per metabolite: log2")
axes[2].boxplot([X_scaled[c].dropna() for c in metabolites[:30]], showfliers=False)
axes[2].set_title("Z-scores, first 30 metabolites")
axes[2].set_xticks([])
for ax in axes[:2]:
    ax.set_xlabel("Skewness")
    ax.set_ylabel("Number of metabolites")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_transform_diagnostics.png", dpi=150)
print(f"Saved {FIG_DIR / '02_transform_diagnostics.png'}")