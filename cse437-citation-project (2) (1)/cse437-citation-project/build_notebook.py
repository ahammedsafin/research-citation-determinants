import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent

CELLS = []

def md(src):
    CELLS.append(("markdown", src))

def code(src):
    CELLS.append(("code", src))

# ============================================================
md("""# Predicting Highly Cited Research Papers
## CSE437: Data Science - Research Project

**Research question:** Which factors make a research paper highly cited - and do textual (TF-IDF / transformer-embedding) signals add predictive value beyond bibliometric features?

**Dataset:** AMiner `dblp-v10` (1,000,000 papers, 1956-2017). Columns: `abstract, authors, n_citation, references, title, venue, year, id`.

**Notebook map**
| # | Section | Report phase |
|---|---------|--------------|
| 1 | Setup & configuration | - |
| 2 | Data loading | Phase 2 |
| 3 | Cleaning & deduplication | Phase 2 |
| 4 | Feature engineering | Phase 2 |
| 5 | Target construction + citation artifact check | Phase 2 |
| 6 | EDA | Phase 2 |
| 7 | Temporal split | Phase 3 |
| 8 | Baseline, Logistic Regression, Random Forest, XGBoost | Phase 3/4 |
| 9 | TF-IDF text model | Phase 3/4 |
| 10 | Model comparison | Phase 4 |
| 11 | Feature importance + SHAP | Phase 4 |
| 12 | Ablation study (mirrors Fu & Aliferis 2008) | Phase 4 |
| 13 | Statistical significance tests | Phase 4 |
| 14 | Error analysis | Phase 4 |
| 15 | Transformer embeddings + hybrid model (optional) | Phase 3/4 |
| 16 | Sensitivity analysis (thresholds + artifact) | Phase 4 |

**How to run in Colab:** `Runtime > Change runtime type` is not required (everything runs on CPU).
Upload the dataset zip to `/content`, then run the data cell below which unzips it.
""")

# ============================================================
code('''# ============================================================
# SECTION 1: SETUP & CONFIGURATION
# ============================================================
import os, sys, subprocess, ast, re, json, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

def pip_install(pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)

pip_install(["scikit-learn", "matplotlib", "xgboost"])

RUN_EMBEDDINGS = os.environ.get("RUN_EMBEDDINGS", "0") == "1"   # set True (or env=1) for Section 15; needs ~15 min CPU
RUN_SHAP       = os.environ.get("RUN_SHAP", "1") == "1"

if RUN_EMBEDDINGS:
    pip_install(["sentence-transformers"])
if RUN_SHAP:
    try:
        import shap  # noqa
    except ImportError:
        try:
            pip_install(["shap"])
        except Exception as e:
            print("SHAP install failed, continuing without it:", e)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
)
from scipy.stats import binomtest

pd.set_option("display.max_columns", None)

try:
    from IPython.display import display  # noqa
except ImportError:
    def display(x):
        print(x if not hasattr(x, "to_string") else x.to_string())

RANDOM_STATE = 42

print("Setup complete.")
print("RUN_EMBEDDINGS =", RUN_EMBEDDINGS, "| RUN_SHAP =", RUN_SHAP)
''')

# ============================================================
code('''# ============================================================
# SECTION 1b: CONFIGURATION
# ============================================================
IS_COLAB = os.path.exists("/content")
BASE     = Path("/content") if IS_COLAB else Path.cwd()

DATA_DIR = BASE / "data"; FIG_DIR = BASE / "figures"; RES_DIR = BASE / "results"
for d in (DATA_DIR, FIG_DIR, RES_DIR):
    d.mkdir(exist_ok=True)

COHORT_YEARS = range(2005, 2015)
TARGET_N     = 100_000
TOP_PCT      = 0.90
TRAIN_YEARS  = set(range(2005, 2013))
TEST_YEARS   = {2013, 2014}

USECOLS = ["abstract", "authors", "n_citation", "references", "title", "venue", "year", "id"]

CANDIDATES = [
    Path("/content/dblp-v10.csv"),
    Path("/content/data/dblp-v10.csv"),
    Path("/content/dblp/dblp-v10.csv"),
    Path(DATA_DIR) / "dblp-v10.csv",
]
if os.environ.get("LOCAL_DBLP"):
    CANDIDATES.append(Path(os.environ["LOCAL_DBLP"]))
CSV_PATH = next((p for p in CANDIDATES if p.exists()), None)

if CSV_PATH is None:
    if IS_COLAB:
        z = Path("/content/dblp-v10.csv.zip")
        if not z.exists():
            from google.colab import files  # type: ignore
            print("Choose dblp-v10.csv.zip to upload (~620 MB)...")
            up = files.upload()
            z = Path("/content") / list(up.keys())[0] if up else z
        print("Unzipping...")
        subprocess.run(["unzip", "-o", str(z), "-d", "/content"], check=True)
        CSV_PATH = next((p for p in CANDIDATES if p.exists()), None)
    else:
        local = DATA_DIR / "dblp-v10.csv"
        CSV_PATH = local if local.exists() else None

assert CSV_PATH is not None and Path(CSV_PATH).exists(), (
    "Could not find dblp-v10.csv - upload the zip or place the csv and re-run this cell."
)
print("Using data file:", CSV_PATH)
''')

# ============================================================
code('''# ============================================================
# SECTION 2: DATA LOADING (chunked stream over ~1 GB csv)
# Only papers from the cohort window are kept in memory.
# ============================================================
CACHE = DATA_DIR / "cohort.pkl"

if CACHE.exists():
    df = pd.read_pickle(CACHE)
    print(f"Loaded cached cohort: {df.shape}")
else:
    chunks, n_raw, n_kept = [], 0, 0
    y_lo, y_hi = min(COHORT_YEARS), max(COHORT_YEARS)
    for ch in pd.read_csv(CSV_PATH, usecols=USECOLS, chunksize=150_000):
        n_raw += len(ch)
        ch = ch[ch["year"].between(y_lo, y_hi)]
        n_kept += len(ch)
        chunks.append(ch)
    cohort_full = pd.concat(chunks, ignore_index=True)
    del chunks

    frac = min(1.0, TARGET_N / len(cohort_full))
    parts = [g.sample(frac=frac, random_state=RANDOM_STATE)
             for _, g in cohort_full.groupby("year")]
    df = pd.concat(parts, ignore_index=True)
    del parts
    del cohort_full
    print(f"Raw rows read: {n_raw:,} | rows in {y_lo}-{y_hi}: {n_kept:,} | sampled cohort: {df.shape}")

print(df["year"].value_counts().sort_index())
''')

# ============================================================
code('''# ============================================================
# SECTION 3: CLEANING & DEDUPLICATION
# ============================================================
missing_before = df.isna().mean().mul(100).round(2)
print("Missing % before cleaning:")
print(missing_before)

n_before = len(df)
dup_id    = int(df["id"].duplicated().sum())
df["title_key"] = (df["title"].fillna("").str.lower()
                          .str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip())
dup_title = int(df.duplicated(subset=["title_key", "year"]).sum())
df = df.drop_duplicates(subset=["id"]).drop_duplicates(subset=["title_key", "year"], keep="first")
df = df.drop(columns="title_key").reset_index(drop=True)

df["abstract"] = df["abstract"].fillna("")
df["venue"]    = df["venue"].astype("string").str.strip().str.lower()
df["text"]     = df["title"].fillna("") + ". " + df["abstract"]

df["has_abstract"]  = (df["abstract"].str.len() > 0).astype(int)
df["has_refs"]      = df["references"].notna().astype(int)
df["venue_missing"] = df["venue"].isna().astype(int)

print(f"Rows before dedup: {n_before:,} | duplicate ids removed: {dup_id:,} "
      f"| duplicate title+year removed: {dup_title:,} | final rows: {len(df):,}")
print("Missing % after cleaning:")
print(df.isna().mean().mul(100).round(2))
''')

# ============================================================
code('''# ============================================================
# SECTION 4: FEATURE ENGINEERING
# ============================================================
parse_failures = {"authors": 0, "references": 0}

def parse_list(x, key):
    if pd.isna(x):
        return []
    try:
        v = ast.literal_eval(x)
        return v if isinstance(v, list) else []
    except Exception:
        parse_failures[key] += 1
        return []

df["author_count"] = df["authors"].map(lambda x: len(parse_list(x, "authors"))).clip(upper=20)
df["ref_count"]    = df["references"].map(lambda x: len(parse_list(x, "references")))
df.loc[df["has_refs"] == 0, "ref_count"] = 0

df["title_word_count"]    = df["title"].fillna("").str.split().str.len()
df["title_char_len"]      = df["title"].fillna("").str.len()
df["abstract_word_count"] = df["abstract"].str.split().str.len()
df["abstract_word_count"] = df["abstract_word_count"].where(df["has_abstract"] == 1, 0)

print("List-parse failures:", parse_failures)
display(df[["author_count", "ref_count", "title_word_count",
            "abstract_word_count"]].describe().round(2))
''')

# ============================================================
code('''# ============================================================
# SECTION 5: TARGET CONSTRUCTION + CITATION-COUNT ARTIFACT CHECK
# Label = top 10% of papers WITHIN their publication year
# (percentile ranking controls for publication age / exposure time)
# ============================================================
df["citation_percentile"] = df.groupby("year")["n_citation"].rank(pct=True)
df["highly_cited"] = (df["citation_percentile"] >= TOP_PCT).astype(int)

print("Citation quantiles:")
print(df["n_citation"].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

artifact_share = (df["n_citation"] == 50).mean()
print(f"\\nARTIFACT CHECK: {(df['n_citation'] == 50).sum():,} papers "
      f"({artifact_share:.1%}) have n_citation EXACTLY 50 - a known AMiner imputation "
      f"default. This mass sits right at our top-10% boundary, so we run a robustness "
      f"check excluding these rows in Section 16.")

print("\\nClass balance (1 = highly cited, top 10% of its year):")
print(df["highly_cited"].value_counts(normalize=True).round(3))

df.to_pickle(CACHE)
print("\\nCohort cached ->", CACHE)
''')

# ============================================================
code('''# ============================================================
# SECTION 6a: EDA - citation distribution + papers per year + artifact
# ============================================================
fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))

ax[0].hist(df["n_citation"].clip(upper=500), bins=60, color="#4c72b0")
ax[0].set_title("Citation counts (clipped at 500)")
ax[0].set_xlabel("n_citation"); ax[0].set_ylabel("papers")

ax[1].hist(np.log1p(df["n_citation"]), bins=60, color="#55a868")
ax[1].set_title("log(1+citations): still right-skewed")
ax[1].set_xlabel("log1p(n_citation)")

spike = df.groupby("year").apply(lambda g: (g["n_citation"] == 50).mean(), include_groups=False)
ax[2].bar(spike.index.astype(str), spike.values * 100, color="#c44e52")
ax[2].set_title("% of papers with n_citation EXACTLY 50 (imputation artifact)")
ax[2].set_ylabel("% of year's papers"); ax[2].tick_params(axis="x", rotation=90)

plt.tight_layout(); plt.savefig(FIG_DIR / "fig1_citation_distribution.png", dpi=150); plt.show()

cnt = df["year"].value_counts().sort_index()
plt.figure(figsize=(9, 4))
plt.bar(cnt.index.astype(str), cnt.values, color="#4c72b0")
plt.title(f"Papers per year in cohort (N={len(df):,})")
plt.ylabel("papers"); plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_papers_per_year.png", dpi=150); plt.show()

print("Observation: citations are extremely skewed (median",
      int(df['n_citation'].median()), "; p99 =", int(df['n_citation'].quantile(0.99)),
      "). The exact-50 spike confirms an imputation artifact that must be handled.")
''')

# ============================================================
code('''# ============================================================
# SECTION 6b: EDA - missingness, venues, feature relationships
# ============================================================
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
miss = (df[["venue", "references", "abstract"]].isna().mean() * 100)
miss_labels = [f"{c} missing" for c in miss.index]
ax[0].bar(miss_labels, miss.fillna(0).values, color="#8172b2")
ax[0].set_ylabel("% missing"); ax[0].set_title("Missingness by field")

top_v = df["venue_norm"] if "venue_norm" in df else df["venue"]
tv = top_v.value_counts().head(15)[::-1]
ax[1].barh(tv.index, tv.values, color="#4c72b0")
ax[1].set_title("Top 15 venues by paper count"); ax[1].tick_params(labelsize=8)
plt.tight_layout(); plt.savefig(FIG_DIR / "fig3_missingness_venues.png", dpi=150); plt.show()

bins = [1, 2, 3, 4, 5, 8, 21]
lab = ["1", "2", "3", "4", "5-7", "8+"]
grp = pd.cut(df["author_count"], bins=[0, 1, 2, 3, 4, 7, 20], labels=lab)
med = df.groupby(grp, observed=True)["n_citation"].apply(lambda s: np.log1p(s).median())

plt.figure(figsize=(8, 4))
med.plot(kind="bar", color="#55a868")
plt.title("Median log(1+citations) by team size")
plt.xlabel("number of authors"); plt.ylabel("median log1p(citations)")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig4_teamsize_vs_impact.png", dpi=150); plt.show()

q = pd.qcut(df["ref_count"], 10, duplicates="drop")
rate_ref = df.groupby(q, observed=True)["n_citation"].apply(lambda s: (s > 0).mean())
plt.figure(figsize=(8, 4))
rate_ref.plot(kind="line", marker="o", color="#c44e52")
plt.title("Share of cited papers across reference-count deciles")
plt.xlabel("reference-count decile"); plt.ylabel("share with >=1 citation")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig5_refcount_vs_impact.png", dpi=150); plt.show()

NUM_EDA = ["n_citation", "author_count", "ref_count", "title_word_count",
           "title_char_len", "abstract_word_count", "year"]
corr = df[NUM_EDA].assign(logcites=lambda d: np.log1p(d["n_citation"])).corr(method="spearman")
plt.figure(figsize=(7, 6))
im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046)
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title("Spearman correlation heatmap")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig6_corr_heatmap.png", dpi=150); plt.show()

print("EDA observations written to results/eda_notes.txt")
with open(RES_DIR / "eda_notes.txt", "w", encoding="utf-8") as f:
    f.write(f"""EDA KEY NUMBERS
Cohort size: {len(df):,}
Years: {df['year'].min()}-{df['year'].max()}
Median citations: {df['n_citation'].median()} | Mean: {df['n_citation'].mean():.1f} | Max: {df['n_citation'].max()}
Exact-50 artifact share: {artifact_share:.1%}
Highly-cited prevalence (top 10%/year): {df['highly_cited'].mean():.1%}
Venue missing: {df['venue_missing'].mean():.1%} | References missing: {1-df['has_refs'].mean():.1%} | Abstract missing: {1-df['has_abstract'].mean():.1%}
Median team size: {df['author_count'].median()} | Median reference count: {df['ref_count'].median()}
Spearman(ref_count, log cites): {np.corrcoef(df['ref_count'], np.log1p(df['n_citation']))[0,1]:.3f}
Spearman(author_count, log cites): {np.corrcoef(df['author_count'], np.log1p(df['n_citation']))[0,1]:.3f}
""")
''')

# ============================================================
code('''# ============================================================
# SECTION 6c: VENUE FEATURES (leak-safe: past years only)
# venue_count_past : how many papers the venue published BEFORE this year
# venue_prior_rate : share of LAST year's papers at this venue that were highly cited
# Both use only information available at publication time.
# ============================================================
vc = df["venue"].fillna("__missing__").value_counts()
common = vc[vc >= 20].index
df["venue_norm"] = df["venue"].where(df["venue"].isin(common), "__rare__").fillna("__missing__")

vy_n = (df.groupby(["venue_norm", "year"]).size().rename("n").reset_index()
          .sort_values(["venue_norm", "year"]))
vy_n["cum_prev"] = vy_n.groupby("venue_norm")["n"].cumsum() - vy_n["n"]
past_map = dict(zip(zip(vy_n["venue_norm"], vy_n["year"]), vy_n["cum_prev"]))
df["venue_count_past"] = [past_map.get((v, y), 0)
                          for v, y in zip(df["venue_norm"], df["year"])]

vy_r = (df.groupby(["venue_norm", "year"])["highly_cited"].mean().rename("r").reset_index()
          .sort_values(["venue_norm", "year"]))
vy_r["prev_r"] = vy_r.groupby("venue_norm")["r"].shift(1)
rate_map = dict(zip(zip(vy_r["venue_norm"], vy_r["year"]), vy_r["prev_r"]))

glob_prev = df.groupby("year")["highly_cited"].mean().shift(1).to_dict()
valid_priors = [v for v in glob_prev.values() if pd.notna(v)]
GLOBAL_PRIOR = float(np.mean(valid_priors)) if valid_priors else 0.10

def _venue_prior(v, y):
    r = rate_map.get((v, y))
    if r is None or pd.isna(r):
        r = glob_prev.get(y)
    if r is None or pd.isna(r):
        r = GLOBAL_PRIOR
    return float(r)

df["venue_prior_rate"] = [_venue_prior(v, y)
                          for v, y in zip(df["venue_norm"], df["year"])]

NUM_FEATURES = ["year", "author_count", "ref_count", "has_refs", "has_abstract",
                "venue_missing", "title_word_count", "title_char_len",
                "abstract_word_count", "venue_count_past", "venue_prior_rate"]
G_TEXTSTAT = ["title_word_count", "title_char_len", "abstract_word_count", "has_abstract"]
G_BIBLIO   = ["year", "author_count", "ref_count", "has_refs"]
G_VENUE    = ["venue_missing", "venue_count_past", "venue_prior_rate"]

assert int(df[NUM_FEATURES].isna().sum().sum()) == 0, "NaN left in features"
print("Venue features built. Global prior-rate fallback:",
      round(GLOBAL_PRIOR, 4))
print("Feature NaN check passed.")
df[["venue_norm", "venue_count_past", "venue_prior_rate"]].head()
''')

# ============================================================
code('''# ============================================================
# SECTION 7: TEMPORAL SPLIT
# Train on 2005-2012, test on 2013-2014 (simulates real deployment:
# predict future impact using only information available at publication)
# ============================================================
train = df[df["year"].isin(TRAIN_YEARS)].reset_index(drop=True)
test  = df[df["year"].isin(TEST_YEARS)].reset_index(drop=True)

X_train, y_train = train[NUM_FEATURES], train["highly_cited"]
X_test,  y_test  = test[NUM_FEATURES],  test["highly_cited"]

pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
print(f"Train: {train.shape} | Test: {test.shape}")
print(f"Train prevalence: {y_train.mean():.3f} | Test prevalence: {y_test.mean():.3f}")
print(f"scale_pos_weight for XGBoost: {pos_weight:.2f}")
print("""NOTE: prevalence drifts between periods because the exact-50 artifact and
sparse citation records in recent years shift tie-blocks across the top-10%
boundary. This is realistic label noise - Section 16 quantifies its impact.""")
''')

# ============================================================
code('''# ============================================================
# SECTIONS 8-10: MODELS + COMPARISON
# Baseline (majority class) | Logistic Regression | Random Forest |
# XGBoost | TF-IDF + Logistic Regression
# ============================================================
results, preds_store = [], {}

def evaluate(name, y_true, pred, prob):
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, prob) if prob is not None else np.nan,
        "PR_AUC": average_precision_score(y_true, prob) if prob is not None else np.nan,
    }
    results.append(row)
    preds_store[name] = (pred, prob)
    print(f"{name:<38s} F1={row['F1']:.3f} ROC-AUC={row['ROC_AUC']:.3f}")
    return row

majority = int(y_train.mode()[0])
evaluate("00 Majority-class baseline", y_test,
         np.full(len(y_test), majority), np.full(len(y_test), y_train.mean()))

logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=3000, class_weight="balanced",
                               random_state=RANDOM_STATE)),
])
logreg.fit(X_train, y_train)
evaluate("01 Logistic Regression (tabular)", y_test,
         logreg.predict(X_test), logreg.predict_proba(X_test)[:, 1])

rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                            n_jobs=-1, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)
evaluate("02 Random Forest (tabular)", y_test,
         rf.predict(X_test), rf.predict_proba(X_test)[:, 1])

try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        scale_pos_weight=pos_weight, eval_metric="logloss",
                        random_state=RANDOM_STATE, n_jobs=-1)
    xgb.fit(X_train, y_train)
    evaluate("03 XGBoost (tabular)", y_test,
             xgb.predict(X_test), xgb.predict_proba(X_test)[:, 1])
except Exception as e:
    print("XGBoost unavailable:", e)

tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=3,
                        sublinear_tf=True)
Xtr_txt = tfidf.fit_transform(train["text"])
Xte_txt = tfidf.transform(test["text"])

txt_clf = LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear",
                             random_state=RANDOM_STATE)
txt_clf.fit(Xtr_txt, y_train)
evaluate("04 TF-IDF + LogReg (text only)", y_test,
         txt_clf.predict(Xte_txt), txt_clf.predict_proba(Xte_txt)[:, 1])

res_df = pd.DataFrame(results)
res_df.to_csv(RES_DIR / "model_comparison.csv", index=False)
print("\\n", res_df.round(3))
''')

# ============================================================
code('''# ============================================================
# SECTION 10b: ROC curves + confusion matrices
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for name, (pred, prob) in preds_store.items():
    if prob is not None:
        fpr, tpr, _ = roc_curve(y_test, prob)
        axes[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_test, prob):.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC curves"); axes[0].legend(fontsize=8)

best_name = res_df.sort_values("F1", ascending=False).iloc[0]["Model"]
pred_best, _ = preds_store[best_name]
ConfusionMatrixDisplay(confusion_matrix(y_test, pred_best)).plot(ax=axes[1], colorbar=False)
axes[1].set_title(f"Confusion matrix - best F1 model\\n{best_name}")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig7_roc_confusion.png", dpi=150); plt.show()
print("Best F1 model:", best_name)
''')

# ============================================================
code('''# ============================================================
# SECTION 11: FEATURE IMPORTANCE + PERMUTATION IMPORTANCE (+ optional SHAP)
# ============================================================
imp = pd.Series(rf.feature_importances_, index=NUM_FEATURES).sort_values()
imp.plot(kind="barh", figsize=(8, 5), color="#4c72b0")
plt.title("Random Forest impurity importance")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig8_rf_importance.png", dpi=150); plt.show()

pi = permutation_importance(xgb, X_test, y_test, scoring="roc_auc",
                            n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
pi_s = pd.Series(pi.importances_mean, index=NUM_FEATURES).sort_values()
pi_s.plot(kind="barh", figsize=(8, 5), color="#c44e52")
plt.title("Permutation importance (XGBoost, Delta ROC-AUC on test)")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig9_permutation_importance.png", dpi=150); plt.show()
print(pi_s.sort_values(ascending=False).round(4))

if RUN_SHAP:
    try:
        import shap
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X_test),
                                                         min(2000, len(X_test)),
                                                         replace=False)
        Xs = X_test.iloc[idx]
        explainer = shap.TreeExplainer(xgb)
        sv = explainer.shap_values(Xs)
        plt.figure()
        shap.summary_plot(sv, Xs, show=False)
        plt.tight_layout(); plt.savefig(FIG_DIR / "fig10_shap_summary.png", dpi=150); plt.show()
    except Exception as e:
        print("SHAP skipped:", e)
else:
    print("RUN_SHAP=False -> skipped (set env RUN_SHAP=1 to enable)")
''')

# ============================================================
code('''# ============================================================
# SECTION 12: ABLATION STUDY
# Design mirrors Fu & Aliferis (2008): content-only vs bibliometric-only
# vs combined, all evaluated on the SAME temporal test set.
# ============================================================
svd = TruncatedSVD(n_components=150, random_state=RANDOM_STATE)
Xtr_svd = svd.fit_transform(Xtr_txt)
Xte_svd = svd.transform(Xte_txt)
svd_cols = [f"svd_{i}" for i in range(Xtr_svd.shape[1])]
Xtr_svd = pd.DataFrame(Xtr_svd, columns=svd_cols, index=train.index)
Xte_svd = pd.DataFrame(Xte_svd, columns=svd_cols, index=test.index)
print(f"SVD explained variance ratio (sum of {len(svd_cols)} comps): "
      f"{svd.explained_variance_ratio_.sum():.3f}")

EXPERIMENTS = {
    "E1 Content stats only":        G_TEXTSTAT,
    "E2 Bibliometric core":         G_BIBLIO,
    "E3 Bibliometric + venue":      G_BIBLIO + G_VENUE,
    "E4 Text TF-IDF only (LR)":     None,
    "E5 E2 + TF-IDF SVD":           G_BIBLIO,
    "E6 E3 + TF-IDF SVD":           G_BIBLIO + G_VENUE,
    "E7 FULL hybrid":               G_BIBLIO + G_VENUE + G_TEXTSTAT,
}

abl_rows, abl_preds = [], {}

def run_xgb(cols):
    m = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8,
                      scale_pos_weight=pos_weight, eval_metric="logloss",
                      random_state=RANDOM_STATE, n_jobs=-1)
    m.fit(train[cols], y_train)
    prob = m.predict_proba(test[cols])[:, 1]
    return prob, m

for name, cols in EXPERIMENTS.items():
    tag = name.split()[0]
    if tag == "E4":
        prob = txt_clf.predict_proba(Xte_txt)[:, 1]
    elif tag in ("E5", "E6"):
        feats_tr = pd.concat([train[cols].reset_index(drop=True),
                              Xtr_svd.reset_index(drop=True)], axis=1)
        feats_te = pd.concat([test[cols].reset_index(drop=True),
                              Xte_svd.reset_index(drop=True)], axis=1)
        m = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=pos_weight, eval_metric="logloss",
                          random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(feats_tr, y_train)
        prob = m.predict_proba(feats_te)[:, 1]
    else:
        prob, _ = run_xgb(cols)
    pred = (prob >= 0.5).astype(int)
    abl_rows.append({
        "Experiment": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, prob),
        "PR_AUC": average_precision_score(y_test, prob),
    })
    abl_preds[name] = (pred, prob)

abl_df = pd.DataFrame(abl_rows)
abl_df.to_csv(RES_DIR / "ablation_results.csv", index=False)
print(abl_df.round(3))

fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
x = np.arange(len(abl_df))
ax[0].bar(x - 0.2, abl_df["F1"], width=0.4, label="F1", color="#4c72b0")
ax[0].bar(x + 0.2, abl_df["PR_AUC"], width=0.4, label="PR-AUC", color="#dd8452")
ax[0].set_xticks(x); ax[0].set_xticklabels([e.split()[0] for e in abl_df["Experiment"]])
ax[0].legend(); ax[0].set_title("Ablation study - F1 and PR-AUC")
ax[1].bar(x, abl_df["ROC_AUC"], color="#55a868")
ax[1].set_xticks(x); ax[1].set_xticklabels([e.split()[0] for e in abl_df["Experiment"]])
ax[1].set_ylim(0.5, min(1.0, abl_df["ROC_AUC"].max() + 0.05))
ax[1].set_title("Ablation study - ROC-AUC")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig11_ablation.png", dpi=150); plt.show()
''')

# ============================================================
code('''# ============================================================
# SECTION 13: STATISTICAL SIGNIFICANCE
# Bootstrap CI for Delta ROC-AUC (FULL vs bibliometric-only) + McNemar test
# ============================================================
rng = np.random.RandomState(RANDOM_STATE)
prob_full = abl_preds["E7 FULL hybrid"][1]
prob_base = abl_preds["E3 Bibliometric + venue"][1]
n = len(y_test)
y_arr = y_test.to_numpy()

def auc(p, idx):
    return roc_auc_score(y_arr[idx], p[idx])

boot = []
idx_all = np.arange(n)
for _ in range(500):
    idx = rng.choice(idx_all, n, replace=True)
    boot.append(auc(prob_full, idx) - auc(prob_base, idx))
boot = np.array(boot)
lo, hi = np.percentile(boot, [2.5, 97.5])
delta = auc(prob_full, idx_all) - auc(prob_base, idx_all)
print(f"Delta ROC-AUC (FULL - biblio+venue) = {delta:+.4f}, 95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")

pf = (prob_full >= 0.5).astype(int)
pb = (prob_base >= 0.5).astype(int)
b = int(((pf == y_arr) & (pb != y_arr)).sum())
c = int(((pf != y_arr) & (pb == y_arr)).sum())
if b + c > 0:
    pval = binomtest(b, b + c, 0.5).pvalue
    print(f"McNemar discordant pairs: b={b} (full better), c={c} (base better), p={pval:.4g}")
else:
    print("No discordant predictions - models identical at threshold 0.5.")

with open(RES_DIR / "significance_tests.txt", "w") as f:
    f.write(f"Delta ROC-AUC full vs biblio: {delta:.4f} CI [{lo:.4f},{hi:.4f}]\\n"
            f"McNemar b={b} c={c}\\n")
''')

# ============================================================
code('''# ============================================================
# SECTION 14: ERROR ANALYSIS
# What kinds of papers does the best model misclassify?
# ============================================================
err = test[["id", "title", "year", "venue_norm", "n_citation", "author_count",
            "ref_count", "abstract_word_count", "venue_count_past"]].copy()
err["true"] = y_arr
err["pred"] = pf
err["prob"] = prob_full
err["error_type"] = np.select(
    [(err["true"] == 1) & (err["pred"] == 0), (err["true"] == 0) & (err["pred"] == 1)],
    ["false_negative", "false_positive"], default="correct")

summary = err.groupby("year")["error_type"].value_counts(normalize=True).unstack().round(3)
print("Error mix by year:"); print(summary)

err["venue_q"] = pd.qcut(err["venue_count_past"], 4, duplicates="drop")
print("\\nError mix by venue-popularity quartile:")
print(err.groupby("venue_q", observed=True)["error_type"].value_counts(normalize=True).unstack().round(3))

err["abs_q"] = pd.qcut(err["abstract_word_count"].replace(0, 1), 4, duplicates="drop")
print("\\nError mix by abstract-length quartile:")
print(err.groupby("abs_q", observed=True)["error_type"].value_counts(normalize=True).unstack().round(3))

fn = err[err["error_type"] == "false_negative"].nlargest(8, "prob")
fp = err[err["error_type"] == "false_positive"].nsmallest(8, "prob")
print("\\nSample FALSE NEGATIVES (model missed high-impact):")
print(fn[["title", "year", "n_citation", "prob"]].to_string(index=False, max_colwidth=70))
print("\\nSample FALSE POSITIVES (model over-predicted):")
print(fp[["title", "year", "n_citation", "prob"]].to_string(index=False, max_colwidth=70))

err.to_csv(RES_DIR / "error_analysis.csv", index=False)
''')

# ============================================================
code('''# ============================================================
# SECTION 15: TRANSFORMER EMBEDDINGS + HYBRID MODEL (optional, CPU ~15 min)
# Sentence-BERT family model (all-MiniLM-L6-v2) - NOT GPT.
# Set RUN_EMBEDDINGS=True above (or env RUN_EMBEDDINGS=1) to execute.
# ============================================================
if RUN_EMBEDDINGS:
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    emb_tr = st_model.encode(train["text"].tolist(), batch_size=128,
                             convert_to_numpy=True, normalize_embeddings=True,
                             show_progress_bar=True)
    emb_te = st_model.encode(test["text"].tolist(), batch_size=128,
                             convert_to_numpy=True, normalize_embeddings=True)
    np.save(DATA_DIR / "emb_train.npy", emb_tr)
    np.save(DATA_DIR / "emb_test.npy", emb_te)
    print("Embeddings:", emb_tr.shape, emb_te.shape)

    emb_cols = [f"emb_{i}" for i in range(emb_tr.shape[1])]
    tr_e = pd.DataFrame(emb_tr, columns=emb_cols, index=train.index)
    te_e = pd.DataFrame(emb_te, columns=emb_cols, index=test.index)

    lr_emb = LogisticRegression(max_iter=3000, class_weight="balanced",
                                random_state=RANDOM_STATE)
    lr_emb.fit(tr_e, y_train)
    evaluate("05 SBERT embedding + LogReg (text)", y_test,
             lr_emb.predict(te_e), lr_emb.predict_proba(te_e)[:, 1])

    hyb_cols = G_BIBLIO + G_VENUE
    tr_h = pd.concat([train[hyb_cols].reset_index(drop=True),
                      tr_e.reset_index(drop=True)], axis=1)
    te_h = pd.concat([test[hyb_cols].reset_index(drop=True),
                      te_e.reset_index(drop=True)], axis=1)
    xgb_h = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=pos_weight, eval_metric="logloss",
                          random_state=RANDOM_STATE, n_jobs=-1)
    xgb_h.fit(tr_h, y_train)
    evaluate("06 Biblio+venue + SBERT hybrid", y_test,
             xgb_h.predict(te_h), xgb_h.predict_proba(te_h)[:, 1])

    res_df = pd.DataFrame(results)
    res_df.to_csv(RES_DIR / "model_comparison_with_embeddings.csv", index=False)
    print(res_df.round(3))
else:
    print("Skipped (RUN_EMBEDDINGS=False). In Colab: set RUN_EMBEDDINGS = True in Section 1.")
''')

# ============================================================
code('''# ============================================================
# SECTION 16: SENSITIVITY ANALYSIS
# 16a Different thresholds: top 5% / 10% / 20% within year
# 16b Artifact robustness: exclude n_citation==50 rows, rebuild labels
# Uses the E3 (bibliometric + venue) XGBoost config throughout.
# ============================================================
def make_labels(frame, pct):
    q = frame.groupby("year")["n_citation"].rank(pct=True)
    return (q >= pct).astype(int)

def run_e3(frame, labels, tag):
    pw = float((labels == 0).sum() / max((labels == 1).sum(), 1))
    tr_m = frame["year"].isin(TRAIN_YEARS); te_m = frame["year"].isin(TEST_YEARS)
    cols = G_BIBLIO + G_VENUE
    m = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pw,
                      eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)
    m.fit(frame.loc[tr_m, cols], labels[tr_m.values])
    prob = m.predict_proba(frame.loc[te_m, cols])[:, 1]
    yt = labels[te_m.values]
    pred = (prob >= 0.5).astype(int)
    return {"Variant": tag,
            "Prevalence": round(float(yt.mean()), 3),
            "F1": f1_score(yt, pred, zero_division=0),
            "ROC_AUC": roc_auc_score(yt, prob),
            "PR_AUC": average_precision_score(yt, prob)}

sens_rows = [
    run_e3(df, make_labels(df, 0.95), "Top 5% within year"),
    run_e3(df, make_labels(df, 0.90), "Top 10% within year"),
    run_e3(df, make_labels(df, 0.80), "Top 20% within year"),
]

mask50 = df["n_citation"] != 50
df_no50 = df[mask50].reset_index(drop=True)
sens_rows.append(run_e3(df_no50, make_labels(df_no50, 0.90),
                        "Top 10%, exact-50 rows excluded"))

sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(RES_DIR / "sensitivity_analysis.csv", index=False)
print(sens_df)
print("\\nIf conclusions hold across thresholds and after removing the artifact, "
      "findings are robust; discuss any drift in the report's limitations section.")
''')

# ============================================================
code('''# ============================================================
# SECTION 17: FINAL SUMMARY (paste into Results section)
# ============================================================
final = pd.concat([
    res_df.assign(Table="main"),
    abl_df.rename(columns={"Experiment": "Model"}).assign(Table="ablation"),
    sens_df.rename(columns={"Variant": "Model"}).assign(Table="sensitivity"),
], ignore_index=True, sort=False)
final.to_csv(RES_DIR / "ALL_RESULTS.csv", index=False)

print("=" * 70)
print("FILES PRODUCED")
print("=" * 70)
for sub in (DATA_DIR, FIG_DIR, RES_DIR):
    for p in sorted(sub.glob("*")):
        print("  ", p)

print("""
KEY REPORT SENTENCES (fill numbers from tables above)
1. Temporal split: trained 2005-2012, tested 2013-2014 (n_test = {}).
2. Best tabular model: see ablation table (E3) vs text-only (E4/E5) vs full hybrid (E7).
3. Adding textual features changed ROC-AUC by ... (Section 12/13, significance test).
4. Top drivers (permutation importance): ...
5. Sensitivity: conclusions stable/unstable across top 5/10/20% and after artifact removal.
""".format(n))
''')

# ============================================================
NB_META = {
    "colab": {"provenance": [], "toc_visible": True},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}

def build(path):
    nb = {
        "cells": [
            {"cell_type": t,
             "metadata": {},
             **({"execution_count": None, "outputs": []} if t == "code" else {}),
             "source": s}
            for t, s in CELLS
        ],
        "metadata": NB_META,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    Path(path).write_text(json.dumps(nb, indent=1, ensure_ascii=False),
                          encoding="utf-8")
    print("Notebook written:", path, f"({len(CELLS)} cells)")

def run_cells():
    g = {"__name__": "__notebook__"}
    for i, (t, s) in enumerate(CELLS):
        if t != "code":
            continue
        lines = [ln for ln in s.split("\n")
                 if not ln.lstrip().startswith(("!", "%"))
                 and "google.colab" not in ln]
        src = "\n".join(lines)
        print(f"\n{'='*30} CELL EXEC {i} {'='*30}", flush=True)
        exec(compile(src, f"<cell{i}>", "exec"), g)

if __name__ == "__main__":
    out = BASE.parent / "citation_prediction_project.ipynb"
    build(out)
    if "--exec" in sys.argv:
        run_cells()
