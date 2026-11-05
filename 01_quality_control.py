
from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------------ CONFIG ----------------------------------
RAW_PATH = Path("/data/RawData.csv")
OUT_DIR = Path("/data/processed")

ID_COL = "Sample.Name"
BATCH_COL = "Batch.Number"
LABEL_COL = "Class.name"
META_COLS = ["Sex", "Age"]          # kept as features/covariates
LEAKAGE_COLS = ["CT"]               # PCR cycle threshold: only exists for infected patients
BLANK_LABEL = "VTM"                 # viral transport medium blanks, not patients
MISSING_COL_THRESHOLD = 0.5         # drop metabolites invalid in >50% of samples

# Text entries that mean "no usable measurement"
PLACEHOLDERS = ["< 0", "No Peak", "FAILED", "#DIV/0!"]
# ------------------------------------------------------------------------


# ===== 1. IMPORT THE DATA ================================================
print("=" * 60, "\n1. IMPORTING DATA\n", "=" * 60, sep="")
df = pd.read_csv(RAW_PATH)
print(f"Loaded {RAW_PATH}: {df.shape[0]} rows x {df.shape[1]} columns")
print(df.iloc[:5, :8])


# ===== 2. INSPECT COLUMN NAMES ===========================================
print("\n" + "=" * 60 + "\n2. COLUMN NAMES\n" + "=" * 60)
print(f"Number of columns: {len(df.columns)}")
print(list(df.columns))

expected = [ID_COL, BATCH_COL, LABEL_COL] + META_COLS + LEAKAGE_COLS
print(f"\nExpected non-metabolite columns missing from file: "
      f"{[c for c in expected if c not in df.columns]}")
print(f"Names with stray spaces: {[c for c in df.columns if c != c.strip()]}")
lowered = pd.Series([c.strip().lower() for c in df.columns])
print(f"Duplicate names (ignoring case): {lowered[lowered.duplicated()].tolist()}")

non_met = set(expected)
metabolites = [c for c in df.columns if c not in non_met]
print(f"Metabolite columns: {len(metabolites)}")


# ===== 3. DATA TYPES / NON-NUMERIC ENTRIES ===============================
print("\n" + "=" * 60 + "\n3. DATA TYPES\n" + "=" * 60)
print(df.dtypes.value_counts().to_string())
print(f"\nMetabolite columns stored as text: "
      f"{sum(df[c].dtype == object or str(df[c].dtype) == 'str' for c in metabolites)} "
      f"of {len(metabolites)}  (caused by placeholder entries)")

# Every distinct non-numeric entry in the metabolite block
tokens = {}
for c in metabolites:
    for v in df[c].dropna():
        if isinstance(v, str):
            try:
                float(v)
            except ValueError:
                tokens[v] = tokens.get(v, 0) + 1
print("\nNon-numeric entries found in metabolite columns:")
print(pd.Series(tokens).sort_values(ascending=False).to_string())

# Check Age and CT for corrupted entries (e.g. Excel turning '1-4' into '04-Jan')
for c in ["Age", "CT"]:
    bad = df[pd.to_numeric(df[c], errors="coerce").isna() & df[c].notna()]
    print(f"\nNon-numeric values in {c}: {bad[c].value_counts().to_dict()}")
    if c == "Age" and len(bad):
        print("  -> rows:", bad[ID_COL].tolist(), "(check original source; likely Excel date corruption)")


# ===== 4. MISSING VALUES =================================================
print("\n" + "=" * 60 + "\n4. MISSING VALUES\n" + "=" * 60)
print(f"True NaN cells by column:\n{df.isna().sum()[df.isna().sum() > 0].to_string()}")

# Treat placeholders as missing for the purposes of QC
met_raw = df[metabolites].replace(PLACEHOLDERS, np.nan)
met_num = met_raw.apply(pd.to_numeric, errors="coerce")
frac_invalid = met_num.isna().mean().sort_values(ascending=False)

print(f"\nMetabolites with NO invalid entries: {(frac_invalid == 0).sum()}")
print(f"Metabolites invalid in >20% of samples: {(frac_invalid > 0.2).sum()}")
print(f"Metabolites invalid in >{MISSING_COL_THRESHOLD:.0%} of samples: "
      f"{(frac_invalid > MISSING_COL_THRESHOLD).sum()}")
print("\nTop 15 metabolites by fraction invalid:")
print(frac_invalid.head(15).round(3).to_string())

# Placeholder type matters: '< 0' = below calibration range, 'No Peak' = not detected, 'FAILED' = run failure
placeholder_counts = pd.DataFrame({p: (df[metabolites] == p).sum() for p in PLACEHOLDERS})
print("\nPlaceholder type counts across all metabolites:")
print(placeholder_counts.sum().to_string())

# Is missingness different between classes?
by_class = met_num.isna().groupby(df[LABEL_COL]).mean().T
by_class["max_diff"] = by_class.max(axis=1) - by_class.min(axis=1)
print("\nMetabolites whose missingness differs most between classes (potential signal or artifact):")
print(by_class.sort_values("max_diff", ascending=False).head(10).round(2).to_string())


# ===== 5. OTHER CHECKS ===================================================
print("\n" + "=" * 60 + "\n5. OTHER CHECKS\n" + "=" * 60)
print(f"Duplicate rows: {df.duplicated().sum()}   Duplicate sample IDs: {df[ID_COL].duplicated().sum()}")

print("\nClass counts:")
print(df[LABEL_COL].value_counts(dropna=False).to_string())

print("\nClass by batch:")
print(pd.crosstab(df[BATCH_COL], df[LABEL_COL]))

df["_age"] = pd.to_numeric(df["Age"], errors="coerce")
print("\nAge by class (confounder check):")
print(df.groupby(LABEL_COL)["_age"].describe()[["count", "mean", "min", "50%", "max"]].round(1))
print("\nSex by class:")
print(pd.crosstab(df[LABEL_COL], df["Sex"].fillna("missing")))

print("\nCT (PCR cycle threshold) availability by class:")
print(df.groupby(LABEL_COL)["CT"].apply(
    lambda s: f"{pd.to_numeric(s, errors='coerce').notna().sum()} numeric, "
              f"{(s == 'Negative').sum()} 'Negative', {s.isna().sum()} blank"))
df = df.drop(columns="_age")


# ===== 6. CLEAN THE DATA =================================================
print("\n" + "=" * 60 + "\n6. CLEANING\n" + "=" * 60)
drop_log = []

# 6a. Tidy labels
df[LABEL_COL] = df[LABEL_COL].str.strip()
df[ID_COL] = df[ID_COL].str.strip()

# 6b. Separate blank (VTM) samples from patient samples
blanks = df[df[LABEL_COL] == BLANK_LABEL].copy()
df = df[df[LABEL_COL] != BLANK_LABEL].copy()
print(f"Separated {len(blanks)} {BLANK_LABEL} blank samples (saved separately, not used as patients)")
for s in blanks[ID_COL]:
    drop_log.append((s, "row", f"{BLANK_LABEL} blank sample"))

# 6c. Placeholders -> NaN, metabolites -> numeric
df[metabolites] = df[metabolites].replace(PLACEHOLDERS, np.nan).apply(pd.to_numeric, errors="coerce")
print(f"Converted placeholders ({PLACEHOLDERS}) to NaN in {len(metabolites)} metabolite columns")

# 6d. Age / CT: numeric conversion (corrupted entries -> NaN)
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["CT"] = pd.to_numeric(df["CT"], errors="coerce")   # 'Negative' becomes NaN

# 6e. Drop metabolites with too much missing data (computed on patient samples only)
frac = df[metabolites].isna().mean()
too_missing = frac[frac > MISSING_COL_THRESHOLD].index.tolist()
for c in too_missing:
    drop_log.append((c, "column", f"invalid in {frac[c]:.0%} of samples"))
df = df.drop(columns=too_missing)
metabolites = [c for c in metabolites if c not in too_missing]
print(f"Dropped {len(too_missing)} metabolites invalid in >{MISSING_COL_THRESHOLD:.0%}: {too_missing}")

# 6f. Drop constant metabolites
constant = [c for c in metabolites if df[c].nunique(dropna=True) <= 1]
for c in constant:
    drop_log.append((c, "column", "constant"))
df = df.drop(columns=constant)
metabolites = [c for c in metabolites if c not in constant]
print(f"Dropped {len(constant)} constant metabolites")

# 6g. Keep CT out of the modeling table (leakage), but save it separately
ct = df[[ID_COL, "CT"]].copy()
df = df.drop(columns=LEAKAGE_COLS)
drop_log.append(("CT", "column", "leakage: PCR viral load only exists for infected patients"))
print("Removed CT from the modeling table (saved separately)")

# 6h. Sex -> consistent text, Age kept numeric (missing values left for the pipeline)
df["Sex"] = df["Sex"].str.strip().str.title()

df = df.reset_index(drop=True)
print(f"\nFinal shape: {df.shape[0]} samples x {df.shape[1]} columns "
      f"({len(metabolites)} metabolites + metadata)")
print(f"Remaining missing metabolite cells: {int(df[metabolites].isna().sum().sum())}")
print(f"Missing Age: {df['Age'].isna().sum()}   Missing Sex: {df['Sex'].isna().sum()}")


# ===== 7. SAVE ===========================================================
OUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_DIR / "cleaned.csv", index=False)
blanks.to_csv(OUT_DIR / "vtm_blanks_raw.csv", index=False)
ct.to_csv(OUT_DIR / "ct_values.csv", index=False)
pd.DataFrame(drop_log, columns=["item", "type", "reason"]).to_csv(OUT_DIR / "drop_log.csv", index=False)
print(f"\nSaved cleaned.csv, vtm_blanks_raw.csv, ct_values.csv, drop_log.csv to {OUT_DIR}/")