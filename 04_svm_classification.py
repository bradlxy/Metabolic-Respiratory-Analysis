"""
04_svm_classification.py

Step 4 of the respiratory disease classification project: SVM classifier.

Design (to give honest performance estimates)
  - Input is the LOG2 data (not the scaled file): imputation and scaling are
    refit inside every training fold via a scikit-learn Pipeline, so no
    information from test samples leaks into training.
  - Nested cross-validation: the inner loop tunes the SVM (kernel, C, gamma),
    the outer loop (5-fold x 3 repeats, stratified) measures performance.
  - Runs twice: raw log2 data, and log2 data centred within each batch
    (script 3 showed a strong batch effect). Centring uses no labels, and every
    batch contains all classes in similar numbers, but it does use the means of
    all samples in a batch, including those later held out for testing.
  - Baselines that the SVM has to beat: majority-class guess, random guess,
    an age-only SVM (age is confounded with disease), and a label-permutation test.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_score, permutation_test_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------ CONFIG ----------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent      # use .parent.parent if script is in scripts/
IN_PATH = PROJECT_ROOT / "data" / "processed" / "log2_transformed.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "results"

ID_COL, BATCH_COL, LABEL_COL = "Sample.Name", "Batch.Number", "Class.name"
META_COLS = ["Sex", "Age"]
INCLUDE_CLASSES = None        # None = all classes, or e.g. ["COVID19", "Influenza", "RSV"]
USE_METADATA_FEATURES = False # False = metabolites only (keeps Age/Sex out of the model)
OUTER_SPLITS, OUTER_REPEATS, INNER_SPLITS = 5, 3, 5
N_PERMUTATIONS = 100
RANDOM_STATE = 42
# ------------------------------------------------------------------------

FIG_DIR.mkdir(parents=True, exist_ok=True)

# ===== 1. LOAD ===========================================================
print("=" * 60 + "\n1. LOAD\n" + "=" * 60)
df = pd.read_csv(IN_PATH)
if INCLUDE_CLASSES is not None:
    df = df[df[LABEL_COL].isin(INCLUDE_CLASSES)].reset_index(drop=True)
non_met = [ID_COL, BATCH_COL, LABEL_COL] + META_COLS
metabolites = [c for c in df.columns if c not in non_met]
y = df[LABEL_COL].values
classes = sorted(np.unique(y))
batch = df[BATCH_COL].astype(str)
print(f"Samples: {len(df)}   metabolites: {len(metabolites)}   classes: {pd.Series(y).value_counts().to_dict()}")
print(f"Majority-class accuracy: {pd.Series(y).value_counts(normalize=True).max():.1%}")

X_raw = df[metabolites].copy()
grp = X_raw.groupby(batch.values)
X_centred = (X_raw - grp.transform("mean")) / grp.transform("std").replace(0, 1)

datasets = {"raw_log2": X_raw, "batch_centred": X_centred}
if USE_METADATA_FEATURES:
    meta = pd.DataFrame({"Age": pd.to_numeric(df["Age"], errors="coerce"),
                         "Male": df["Sex"].map({"Male": 1, "Female": 0})})
    datasets = {k: pd.concat([v, meta], axis=1) for k, v in datasets.items()}

pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                 ("scale", StandardScaler()),
                 ("svc", SVC(class_weight="balanced"))])
param_grid = [
    {"svc__kernel": ["linear"], "svc__C": [0.001, 0.01, 0.1, 1]},
    {"svc__kernel": ["rbf"], "svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.001, 0.01]},
]
outer = RepeatedStratifiedKFold(n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=RANDOM_STATE)
inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# ===== 2. NESTED CV ======================================================
results, confusions = [], {}
for name, X in datasets.items():
    print("\n" + "=" * 60 + f"\n2. NESTED CV SVM: {name}\n" + "=" * 60)
    bal, f1m, best_params = [], [], []
    y_true_all, y_pred_all = [], []
    for fold, (tr, te) in enumerate(outer.split(X, y), 1):
        gs = GridSearchCV(pipe, param_grid, cv=inner, scoring="f1_macro", n_jobs=-1)
        gs.fit(X.iloc[tr], y[tr])
        pred = gs.predict(X.iloc[te])
        bal.append(balanced_accuracy_score(y[te], pred))
        f1m.append(f1_score(y[te], pred, average="macro"))
        best_params.append(str(gs.best_params_))
        y_true_all.extend(y[te]); y_pred_all.extend(pred)
    print(f"Balanced accuracy: {np.mean(bal):.3f} +/- {np.std(bal):.3f}")
    print(f"Macro F1:          {np.mean(f1m):.3f} +/- {np.std(f1m):.3f}   ({len(bal)} outer folds)")
    print("Most often selected hyperparameters:")
    print(pd.Series(best_params).value_counts().head(3).to_string())

    cm = confusion_matrix(y_true_all, y_pred_all, labels=classes)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    confusions[name] = cm_norm
    print("Per-class recall: " + ", ".join(f"{c} {r:.2f}" for c, r in zip(classes, np.diag(cm_norm))))
    results.append({"dataset": name, "balanced_acc_mean": np.mean(bal), "balanced_acc_sd": np.std(bal),
                    "macro_f1_mean": np.mean(f1m), "macro_f1_sd": np.std(f1m)})

# ===== 3. BASELINES AND SANITY CHECKS ====================================
print("\n" + "=" * 60 + "\n3. BASELINES AND SANITY CHECKS\n" + "=" * 60)
simple_cv = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for strat in ["most_frequent", "stratified"]:
    sc = cross_val_score(DummyClassifier(strategy=strat, random_state=RANDOM_STATE),
                         X_raw, y, cv=outer, scoring="balanced_accuracy")
    print(f"Dummy ({strat:13s}) balanced accuracy: {sc.mean():.3f}")
    results.append({"dataset": f"dummy_{strat}", "balanced_acc_mean": sc.mean(), "balanced_acc_sd": sc.std()})

# Age-only model: how much of the 'signal' could age alone explain?
age = pd.to_numeric(df["Age"], errors="coerce").to_frame("Age")
age_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("svc", SVC(class_weight="balanced"))])
sc = cross_val_score(age_pipe, age, y, cv=outer, scoring="balanced_accuracy")
print(f"Age-only SVM balanced accuracy: {sc.mean():.3f} +/- {sc.std():.3f}   "
      "(if the metabolite model is not clearly above this, age may be driving the result)")
results.append({"dataset": "age_only_svm", "balanced_acc_mean": sc.mean(), "balanced_acc_sd": sc.std()})

# Permutation test on the batch-centred data (default SVM, no tuning, so it stays fast)
fixed = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                  ("svc", SVC(kernel="rbf", C=1, class_weight="balanced"))])
score, perm_scores, pval = permutation_test_score(
    fixed, datasets["batch_centred"], y, cv=simple_cv, scoring="balanced_accuracy",
    n_permutations=N_PERMUTATIONS, n_jobs=-1, random_state=RANDOM_STATE)
print(f"\nPermutation test (batch-centred, default RBF SVM, {N_PERMUTATIONS} shuffles): "
      f"score {score:.3f} vs shuffled {perm_scores.mean():.3f} +/- {perm_scores.std():.3f}, p = {pval:.3f}")

# ===== 4. SAVE ===========================================================
summary = pd.DataFrame(results).round(3)
summary.to_csv(OUT_DIR / "svm_results.csv", index=False)
print("\nSummary:\n" + summary.to_string(index=False))

fig, axes = plt.subplots(1, len(confusions), figsize=(6 * len(confusions), 5))
for ax, (name, cm_norm) in zip(np.atleast_1d(axes), confusions.items()):
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=classes, yticklabels=classes, ax=ax, cbar=False)
    ax.set_title(f"SVM confusion matrix ({name})\nrows = true class, row-normalised")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout()
plt.savefig(FIG_DIR / "04_svm_confusion.png", dpi=150)
print(f"\nSaved {FIG_DIR / '04_svm_confusion.png'} and {OUT_DIR / 'svm_results.csv'}")