"""
03_pca_and_clustering.py

Step 3 of the respiratory disease classification project (EXPLORATION).
  1. Load the scaled data and impute missing values (for exploration only)
  2. PCA: scree plot, score plots coloured by class / batch / age, loadings
  3. Check how much each PC is explained by class, batch and age
  4. Clustering: hierarchical (Ward) heatmap + k-means, compared with the
     true classes using the adjusted Rand index (ARI)
  5. Repeat PCA and clustering after centring each metabolite within each
     batch, to see whether batch effects are hiding the disease signal

PCA and clustering are UNSUPERVISED: labels are only used to colour plots and
to judge the results afterwards, never to build them.
Imputation here is for exploration. For model evaluation (script 4) the
imputer and scaler are refit inside each cross-validation fold.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score

# ------------------------------ CONFIG ----------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent      # use .parent.parent if script is in scripts/
IN_PATH = PROJECT_ROOT / "data" / "processed" / "scaled_for_exploration.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "results"

ID_COL, BATCH_COL, LABEL_COL = "Sample.Name", "Batch.Number", "Class.name"
META_COLS = ["Sex", "Age"]
INCLUDE_CLASSES = None            # None = all classes, or e.g. ["COVID19", "Influenza", "RSV"]
IMPUTE_METHOD = "median"          # "median" or "min" (min suits below-detection missingness)
N_PCS = 10
K_RANGE = range(2, 9)             # k values to try for k-means
RANDOM_STATE = 42
# ------------------------------------------------------------------------

FIG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=150)
    plt.close()
    print(f"  saved {FIG_DIR / name}")


# ===== 1. LOAD AND IMPUTE ================================================
print("=" * 60 + "\n1. LOAD AND IMPUTE\n" + "=" * 60)
df = pd.read_csv(IN_PATH)
if INCLUDE_CLASSES is not None:
    df = df[df[LABEL_COL].isin(INCLUDE_CLASSES)].reset_index(drop=True)
print(f"Samples: {len(df)}   classes: {df[LABEL_COL].value_counts().to_dict()}")

non_met = [ID_COL, BATCH_COL, LABEL_COL] + META_COLS
metabolites = [c for c in df.columns if c not in non_met]
X = df[metabolites]
print(f"Metabolites: {len(metabolites)}   missing cells to impute: {int(X.isna().sum().sum())}")

if IMPUTE_METHOD == "min":
    X_imp = X.fillna(X.min())
else:
    X_imp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X),
                         columns=metabolites, index=df.index)
print(f"Imputed with column {IMPUTE_METHOD}")

labels = df[LABEL_COL].values
batch = df[BATCH_COL].astype(str).values
age = pd.to_numeric(df["Age"], errors="coerce")


# ===== 2. PCA ============================================================
print("\n" + "=" * 60 + "\n2. PCA\n" + "=" * 60)
n_pcs = min(N_PCS, X_imp.shape[0] - 1, X_imp.shape[1])
pca = PCA(n_components=n_pcs, random_state=RANDOM_STATE)
scores = pca.fit_transform(X_imp)
evr = pca.explained_variance_ratio_
pc_names = [f"PC{i + 1}" for i in range(n_pcs)]
print("Explained variance: " + ", ".join(f"{n} {v:.1%}" for n, v in zip(pc_names[:5], evr[:5])))
print(f"First {n_pcs} PCs together: {evr.sum():.1%}")

scores_df = pd.DataFrame(scores, columns=pc_names)
scores_df.insert(0, ID_COL, df[ID_COL])
scores_df[LABEL_COL] = labels
scores_df.to_csv(OUT_DIR / "pca_scores.csv", index=False)

loadings = pd.DataFrame(pca.components_.T, index=metabolites, columns=pc_names)
loadings.to_csv(OUT_DIR / "pca_loadings.csv")

# Scree plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(range(1, n_pcs + 1), evr * 100, color="tab:blue")
ax.plot(range(1, n_pcs + 1), np.cumsum(evr) * 100, "o-", color="tab:red", label="cumulative")
ax.set_xlabel("Principal component"); ax.set_ylabel("Explained variance (%)")
ax.set_title("Scree plot"); ax.legend()
savefig("03_pca_scree.png")


def score_plot(ax, xpc, ypc, colour_by, title, S=None, E=None):
    S = scores if S is None else S
    E = evr if E is None else E
    i, j = pc_names.index(xpc), pc_names.index(ypc)
    xl, yl = f"{xpc} ({E[i]:.1%})", f"{ypc} ({E[j]:.1%})"
    if colour_by == "age":
        ax.scatter(S[:, i], S[:, j], c="lightgrey", s=25)           # missing age
        ok = age.notna().values
        sc = ax.scatter(S[ok, i], S[ok, j], c=age[ok], cmap="viridis", s=30)
        plt.colorbar(sc, ax=ax, label="Age")
    else:
        vals = labels if colour_by == "class" else batch
        sns.scatterplot(x=S[:, i], y=S[:, j], hue=vals, s=35, ax=ax)
        ax.legend(title=colour_by.capitalize(), fontsize=8)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title)


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
score_plot(axes[0], "PC1", "PC2", "class", "PCA coloured by class")
score_plot(axes[1], "PC1", "PC2", "batch", "PCA coloured by batch")
score_plot(axes[2], "PC1", "PC2", "age", "PCA coloured by age (grey = missing)")
savefig("03_pca_scores_pc1_pc2.png")

if n_pcs >= 3:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    score_plot(axes[0], "PC1", "PC3", "class", "PC1 vs PC3 by class")
    score_plot(axes[1], "PC2", "PC3", "class", "PC2 vs PC3 by class")
    savefig("03_pca_scores_pc3.png")

# Top loadings for PC1 and PC2
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, pc in zip(axes, ["PC1", "PC2"]):
    top = loadings[pc].reindex(loadings[pc].abs().sort_values(ascending=False).index[:12])
    ax.barh(top.index[::-1], top.values[::-1], color="tab:purple")
    ax.set_title(f"Top 12 loadings: {pc}")
savefig("03_pca_top_loadings.png")
print("Top 5 metabolites by |loading| on PC1:", loadings["PC1"].abs().nlargest(5).index.tolist())
print("Top 5 metabolites by |loading| on PC2:", loadings["PC2"].abs().nlargest(5).index.tolist())

# Outlier flag: distance from the centre in the first 5 PCs
k5 = min(5, n_pcs)
dist = np.sqrt(((scores[:, :k5] / scores[:, :k5].std(axis=0)) ** 2).sum(axis=1))
cut = dist.mean() + 3 * dist.std()
outliers = df.loc[dist > cut, [ID_COL, LABEL_COL, BATCH_COL]]
print(f"\nPotential outliers (distance > mean + 3 SD in first {k5} PCs): {len(outliers)}")
if len(outliers):
    print(outliers.to_string(index=False))


# ===== 3. WHAT DOES EACH PC REFLECT? =====================================
print("\n" + "=" * 60 + "\n3. WHAT DRIVES EACH PC (class vs batch vs age)\n" + "=" * 60)


def eta_squared(values, groups):
    """Share of a PC's variance explained by group membership (0-1)."""
    grand = values.mean()
    ss_total = ((values - grand) ** 2).sum()
    ss_between = sum(len(values[groups == g]) * (values[groups == g].mean() - grand) ** 2
                     for g in np.unique(groups))
    return ss_between / ss_total


def driver_table(S, E):
    rows = []
    ok = age.notna().values
    for k, pc in enumerate(pc_names[:5]):
        rho = spearmanr(S[ok, k], age[ok])[0]
        rows.append({"PC": pc, "var_explained": f"{E[k]:.1%}",
                     "class_eta2": round(eta_squared(S[:, k], labels), 2),
                     "batch_eta2": round(eta_squared(S[:, k], batch), 2),
                     "age_spearman": round(rho, 2)})
    return pd.DataFrame(rows)


drivers = driver_table(scores, evr)
print(drivers.to_string(index=False))
print("(eta2 near 1 = that PC is mostly explained by the grouping; "
      "age_spearman near +/-1 = strongly related to age)")
drivers.to_csv(OUT_DIR / "pca_drivers.csv", index=False)


# ===== 4. CLUSTERING =====================================================
print("\n" + "=" * 60 + "\n4. CLUSTERING\n" + "=" * 60)
n_classes = len(np.unique(labels))

# 4a. k-means on the scaled, imputed data: which k fits best?
sil = {}
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X_imp)
    sil[k] = silhouette_score(X_imp, km.labels_)
print("Silhouette score by k: " + ", ".join(f"k={k}: {v:.3f}" for k, v in sil.items()))
best_k = max(sil, key=sil.get)
print(f"Best silhouette at k={best_k}; true number of classes = {n_classes}")

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(list(sil), list(sil.values()), "o-")
ax.set_xlabel("k"); ax.set_ylabel("Silhouette score"); ax.set_title("k-means: choosing k")
savefig("03_kmeans_silhouette.png")

assign = pd.DataFrame({ID_COL: df[ID_COL], "true_class": labels, "batch": batch})
for name, k in [("kmeans_k_best", best_k), ("kmeans_k_classes", n_classes)]:
    assign[name] = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(X_imp)

# 4b. Hierarchical clustering (Ward), cut at the number of classes
hc = AgglomerativeClustering(n_clusters=n_classes, linkage="ward").fit(X_imp)
assign["hierarchical"] = hc.labels_

print("\nAgreement with true classes (ARI: 1 = perfect, ~0 = no better than chance):")
for col in ["kmeans_k_best", "kmeans_k_classes", "hierarchical"]:
    print(f"  {col:18s} ARI vs class: {adjusted_rand_score(labels, assign[col]):.3f}   "
          f"ARI vs batch: {adjusted_rand_score(batch, assign[col]):.3f}")

print(f"\nk-means (k={n_classes}) clusters vs true class:")
print(pd.crosstab(assign["kmeans_k_classes"], assign["true_class"]))
assign.to_csv(OUT_DIR / "cluster_assignments.csv", index=False)

# PCA coloured by cluster
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(x=scores[:, 0], y=scores[:, 1], hue=assign["kmeans_k_classes"].astype(str),
                s=35, ax=axes[0]).set(title=f"k-means (k={n_classes}) on PCA map",
                                      xlabel="PC1", ylabel="PC2")
sns.scatterplot(x=scores[:, 0], y=scores[:, 1], hue=labels, s=35, ax=axes[1]
                ).set(title="True classes", xlabel="PC1", ylabel="PC2")
savefig("03_kmeans_vs_true.png")

# Heatmap with hierarchical clustering of samples and metabolites
class_pal = dict(zip(np.unique(labels), sns.color_palette("Set1", n_classes)))
batch_pal = dict(zip(np.unique(batch), sns.color_palette("Set2", len(np.unique(batch)))))
row_colors = pd.DataFrame({"Class": pd.Series(labels).map(class_pal),
                           "Batch": pd.Series(batch).map(batch_pal)}).set_index(X_imp.index)
cg = sns.clustermap(X_imp, method="ward", cmap="vlag", center=0, vmin=-3, vmax=3,
                    row_colors=row_colors, yticklabels=False, xticklabels=True,
                    figsize=(18, 12), cbar_kws={"label": "z-score"})
cg.ax_heatmap.tick_params(axis="x", labelsize=5)
for lab, col in class_pal.items():
    cg.ax_col_dendrogram.bar(0, 0, color=col, label=lab, linewidth=0)
cg.ax_col_dendrogram.legend(title="Class", loc="center", ncol=n_classes, fontsize=8)
cg.savefig(FIG_DIR / "03_clustermap.png", dpi=150)
plt.close("all")
print(f"  saved {FIG_DIR / '03_clustermap.png'}")



# ===== 5. AFTER BATCH CENTRING ===========================================
print("\n" + "=" * 60 + "\n5. REPEAT AFTER CENTRING EACH METABOLITE WITHIN EACH BATCH\n" + "=" * 60)
# Each metabolite is z-scored within its own batch. This uses no disease labels.
# It is only safe when every batch contains all classes in similar proportions
# (check the class-by-batch table from script 1), otherwise it removes real signal.
print("Class by batch:")
print(pd.crosstab(pd.Series(batch, name="Batch"), pd.Series(labels, name="Class")))
grp = X_imp.groupby(batch)
X_bc = (X_imp - grp.transform("mean")) / grp.transform("std").replace(0, 1)

pca_bc = PCA(n_components=n_pcs, random_state=RANDOM_STATE)
scores_bc = pca_bc.fit_transform(X_bc)
evr_bc = pca_bc.explained_variance_ratio_
print("\nExplained variance after batch centring: "
      + ", ".join(f"{n} {v:.1%}" for n, v in zip(pc_names[:5], evr_bc[:5])))
drivers_bc = driver_table(scores_bc, evr_bc)
print(drivers_bc.to_string(index=False))
drivers_bc.to_csv(OUT_DIR / "pca_drivers_batch_centred.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
score_plot(axes[0], "PC1", "PC2", "class", "Batch-centred PCA: class", scores_bc, evr_bc)
score_plot(axes[1], "PC1", "PC2", "batch", "Batch-centred PCA: batch", scores_bc, evr_bc)
score_plot(axes[2], "PC1", "PC2", "age", "Batch-centred PCA: age", scores_bc, evr_bc)
savefig("03_pca_batch_centred.png")

print("\nClustering after batch centring (ARI vs class / vs batch):")
for name, model in [("kmeans_k_classes", KMeans(n_clusters=n_classes, n_init=20, random_state=RANDOM_STATE)),
                    ("hierarchical", AgglomerativeClustering(n_clusters=n_classes, linkage="ward"))]:
    lab = model.fit_predict(X_bc)
    assign[name + "_batch_centred"] = lab
    print(f"  {name:18s} ARI vs class: {adjusted_rand_score(labels, lab):.3f}   "
          f"ARI vs batch: {adjusted_rand_score(batch, lab):.3f}")
assign.to_csv(OUT_DIR / "cluster_assignments.csv", index=False)

X_bc.insert(0, ID_COL, df[ID_COL].values)
X_bc.insert(1, LABEL_COL, labels)
X_bc.to_csv(OUT_DIR / "batch_centred_for_exploration.csv", index=False)

print("\nDone. Outputs: pca_scores.csv, pca_loadings.csv, pca_drivers.csv, "
      "cluster_assignments.csv and figures in results/")