"""Stage 2: Models, ablation, statistical tests, figures (fast, ~5 min)"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, accuracy_score
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import mannwhitneyu, chi2_contingency, spearmanr
from xgboost import XGBClassifier

print("=== STAGE 2: Models + Ablation + Stats ===", flush=True)

BASE = Path(r"D:\cse360\cse437-citation-project")
DATA_DIR = BASE / "data"
FIG_DIR = BASE / "figures"
RES_DIR = BASE / "results"
for d in (FIG_DIR, RES_DIR):
    d.mkdir(exist_ok=True)

# --- LOAD EXPANDED DATA ---
print("Loading expanded features ...", flush=True)
df = pd.read_pickle(DATA_DIR / "cohort_expanded.pkl")
print(f"Loaded: {df.shape}")

TRAIN_YEARS = set(range(2005, 2013))
G_BIBLIO = ["year", "author_count", "ref_count", "has_refs"]
G_VENUE  = ["venue_missing", "venue_count_past", "venue_prior_rate"]
G_TEXTSTAT = ["title_word_count", "title_char_len", "abstract_word_count", "has_abstract"]
G_BASE = G_BIBLIO + G_VENUE + G_TEXTSTAT
G_NLP = [c for c in df.columns if c.startswith("nlp_")]
G_LDA = [c for c in df.columns if c.startswith("topic_")]
G_FE = ["log_ref_count", "log_author_count", "author_count_sq",
        "ref_x_author", "title_len_x_sentiment", "ref_x_venue_prestige"]
G_TS = ["venue_growth_rate", "venue_3yr_avg", "venue_trend_slope"]

tr_m = df["year"].isin(TRAIN_YEARS).values
te_m = ~tr_m
y = df["highly_cited"]
pos_w = float((y[tr_m] == 0).sum() / max((y[tr_m] == 1).sum(), 1))
print(f"Train: {tr_m.sum()} | Test: {te_m.sum()} | prevalence: {y[te_m].mean():.3f}", flush=True)

def make_xgb():
    return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=pos_w, eval_metric="logloss",
                         random_state=42, n_jobs=-1)

# --- STATISTICAL TESTS ---
print("\n=== Statistical tests ===", flush=True)
stats_rows = []
for feat in ["ref_count", "author_count", "abstract_word_count",
             "venue_prior_rate", "nlp_sentiment", "nlp_fk_grade", "nlp_keyword_count"]:
    grp0 = df.loc[y == 0, feat].dropna()
    grp1 = df.loc[y == 1, feat].dropna()
    u_stat, u_p = mannwhitneyu(grp0, grp1, alternative="two-sided")
    r_val, r_p = spearmanr(df[feat].fillna(0), np.log1p(df["n_citation"]))
    stats_rows.append({
        "Feature": feat,
        "MW_U_stat": round(u_stat, 0),
        "MW_U_p": f"{u_p:.2e}",
        "Spearman_r": round(r_val, 4),
        "Spearman_p": f"{r_p:.2e}",
        "Group0_median": round(grp0.median(), 2),
        "Group1_median": round(grp1.median(), 2),
    })

ct = pd.crosstab(df["venue_missing"], y)
chi2, chi_p, _, _ = chi2_contingency(ct)
stats_rows.append({
    "Feature": "venue_missing x highly_cited",
    "MW_U_stat": chi2, "MW_U_p": f"{chi_p:.2e}",
    "Spearman_r": "", "Spearman_p": "",
    "Group0_median": "", "Group1_median": "",
})

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(RES_DIR / "statistical_tests.csv", index=False)
print(stats_df.to_string(index=False))

# --- BASELINE MODELS ---
print("\n=== Model training ===", flush=True)

results = []
preds_all = {}

def evaluate(name, y_true, pred, prob):
    row = {
        "Model": name,
        "Accuracy": round(accuracy_score(y_true, pred), 4),
        "F1": round(f1_score(y_true, pred, zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_true, prob), 4) if prob is not None else 0,
        "PR_AUC": round(average_precision_score(y_true, prob), 4) if prob is not None else 0,
    }
    results.append(row)
    preds_all[name] = (pred, prob)
    return row

# 1. LR
print("  LR ...", flush=True)
lr = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42))])
lr.fit(df.loc[tr_m, G_BASE], y_tr := y[tr_m].values)
y_te = y[te_m].values
evaluate("01 LR (base)", y_te, lr.predict(df.loc[te_m, G_BASE]),
         lr.predict_proba(df.loc[te_m, G_BASE])[:, 1])

# 2. RF
print("  RF ...", flush=True)
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42)
rf.fit(df.loc[tr_m, G_BASE], y_tr)
evaluate("02 RF (base)", y_te, rf.predict(df.loc[te_m, G_BASE]),
         rf.predict_proba(df.loc[te_m, G_BASE])[:, 1])

# 3. XGB
print("  XGB ...", flush=True)
xgb = make_xgb()
xgb.fit(df.loc[tr_m, G_BASE], y_tr)
evaluate("03 XGB (base)", y_te, xgb.predict(df.loc[te_m, G_BASE]),
         xgb.predict_proba(df.loc[te_m, G_BASE])[:, 1])

# 4. LinearSVC
print("  LinearSVC ...", flush=True)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(df.loc[tr_m, G_BASE])
X_te_s = scaler.transform(df.loc[te_m, G_BASE])
svc = CalibratedClassifierCV(LinearSVC(max_iter=5000, class_weight="balanced", random_state=42))
svc.fit(X_tr_s, y_tr)
evaluate("04 LinearSVC (base)", y_te, svc.predict(X_te_s), svc.predict_proba(X_te_s)[:, 1])

# 5. GradientBoosting
print("  GBM ...", flush=True)
gbm = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, random_state=42)
gbm.fit(df.loc[tr_m, G_BASE], y_tr)
evaluate("05 GradientBoosting (base)", y_te, gbm.predict(df.loc[te_m, G_BASE]),
         gbm.predict_proba(df.loc[te_m, G_BASE])[:, 1])

# 6. Stacking
print("  Stacking ...", flush=True)
estimators = [("lr", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
              ("rf", RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42)),
              ("xgb", make_xgb())]
stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000),
                           cv=3, n_jobs=1)
stack.fit(X_tr_s, y_tr)
evaluate("06 Stacking (base)", y_te, stack.predict(X_te_s), stack.predict_proba(X_te_s)[:, 1])

# 7. XGB + NLP features (new!)
print("  XGB + NLP ...", flush=True)
xgb_nlp = make_xgb()
xgb_nlp.fit(df.loc[tr_m, G_BASE + G_NLP], y_tr)
evaluate("07 XGB (base+NLP)", y_te, xgb_nlp.predict(df.loc[te_m, G_BASE + G_NLP]),
         xgb_nlp.predict_proba(df.loc[te_m, G_BASE + G_NLP])[:, 1])

# 8. XGB + ALL features (new!)
print("  XGB + ALL ...", flush=True)
G_ALL = G_BASE + G_NLP + G_LDA + G_FE + G_TS
xgb_all = make_xgb()
xgb_all.fit(df.loc[tr_m, G_ALL], y_tr)
evaluate("08 XGB (ALL features)", y_te, xgb_all.predict(df.loc[te_m, G_ALL]),
         xgb_all.predict_proba(df.loc[te_m, G_ALL])[:, 1])

res_base = pd.DataFrame(results)
res_base.to_csv(RES_DIR / "model_comparison_expanded.csv", index=False)
print("\nModel comparison:")
print(res_base.to_string(index=False))

# --- TF-IDF BASELINE ---
print("\n=== TF-IDF SVD baseline ===", flush=True)
tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
Xtr_txt = tfidf.fit_transform(df.loc[tr_m, "text"])
Xte_txt = tfidf.transform(df.loc[te_m, "text"])
svd = TruncatedSVD(n_components=150, random_state=42)
S_tr = svd.fit_transform(Xtr_txt)
S_te = svd.transform(Xte_txt)
print(f"SVD explained variance: {svd.explained_variance_ratio_.sum():.3f}")

xgb_txt = make_xgb()
xgb_txt.fit(S_tr, y_tr)
evaluate("09 XGB (TF-IDF SVD)", y_te, xgb_txt.predict(S_te), xgb_txt.predict_proba(S_te)[:, 1])

# XGB biblio+venue+TF-IDF
F_tr_bv = np.hstack([df.loc[tr_m, G_BIBLIO + G_VENUE].values, S_tr])
F_te_bv = np.hstack([df.loc[te_m, G_BIBLIO + G_VENUE].values, S_te])
xgb_bv = make_xgb()
xgb_bv.fit(F_tr_bv, y_tr)
evaluate("10 XGB (biblio+venue+TFIDF)", y_te, xgb_bv.predict(F_te_bv),
         xgb_bv.predict_proba(F_te_bv)[:, 1])

# --- EXPANDED ABLATION ---
print("\n=== Expanded ablation E1-E11 ===", flush=True)

ABLATION = {
    "E1  Content stats only":          G_TEXTSTAT,
    "E2  Bibliometric core":           G_BIBLIO,
    "E3  Bibliometric + venue":        G_BIBLIO + G_VENUE,
    "E6  E3 + TF-IDF SVD":            G_BIBLIO + G_VENUE,  # special handling
    "E8  E3 + NLP features":           G_BIBLIO + G_VENUE + G_NLP,
    "E9  E3 + NLP + LDA topics":       G_BIBLIO + G_VENUE + G_NLP + G_LDA,
    "E10 E9 + feature engineering":    G_BIBLIO + G_VENUE + G_NLP + G_LDA + G_FE,
    "E11 FULL (E10 + time series)":    G_BIBLIO + G_VENUE + G_NLP + G_LDA + G_FE + G_TS,
}

abl_rows = []
for name, cols in ABLATION.items():
    tag = name.split()[0]
    if tag == "E6":
        F_tr = np.hstack([df.loc[tr_m, G_BIBLIO + G_VENUE].values, S_tr])
        F_te = np.hstack([df.loc[te_m, G_BIBLIO + G_VENUE].values, S_te])
        m = make_xgb(); m.fit(F_tr, y_tr); prob = m.predict_proba(F_te)[:, 1]
        n_feat = len(G_BIBLIO + G_VENUE) + 150
    else:
        m = make_xgb()
        m.fit(df.loc[tr_m, cols], y_tr)
        prob = m.predict_proba(df.loc[te_m, cols])[:, 1]
        n_feat = len(cols)
    pred = (prob >= 0.5).astype(int)
    abl_rows.append({
        "Experiment": name,
        "n_features": n_feat,
        "Accuracy": round(accuracy_score(y_te, pred), 4),
        "F1": round(f1_score(y_te, pred, zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_te, prob), 4),
        "PR_AUC": round(average_precision_score(y_te, prob), 4),
    })

abl_df = pd.DataFrame(abl_rows)
abl_df.to_csv(RES_DIR / "ablation_expanded.csv", index=False)
print(abl_df.to_string(index=False))

# --- FIGURES ---
print("\n=== Generating figures ===", flush=True)

# Fig 16: Ablation comparison
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(abl_df))
ax[0].bar(x, abl_df["F1"], color="#4c72b0")
ax[0].set_xticks(x); ax[0].set_xticklabels([e.split()[0] for e in abl_df["Experiment"]], rotation=45)
ax[0].set_title("F1 by experiment"); ax[0].set_ylabel("F1")
ax[1].bar(x, abl_df["ROC_AUC"], color="#55a868")
ax[1].set_xticks(x); ax[1].set_xticklabels([e.split()[0] for e in abl_df["Experiment"]], rotation=45)
ax[1].set_title("ROC-AUC by experiment"); ax[1].set_ylabel("AUC")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig16_ablation_expanded.png", dpi=150); plt.close()

# Fig 17: Mutual information
all_feats = G_BASE + G_NLP + G_LDA + G_FE + G_TS
all_feats = [f for f in all_feats if f in df.columns]
X_all = df.loc[tr_m, all_feats].values
mi = mutual_info_classif(X_all, y_tr, random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({"Feature": all_feats, "MI_score": mi}).sort_values("MI_score", ascending=False)
mi_df.to_csv(RES_DIR / "mutual_information_ranking.csv", index=False)

plt.figure(figsize=(10, 8))
top20 = mi_df.head(20)
plt.barh(top20["Feature"][::-1], top20["MI_score"][::-1], color="#8172b2")
plt.xlabel("Mutual Information score"); plt.title("Top 20 features by MI with highly_cited")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig17_mutual_information.png", dpi=150); plt.close()

# Fig 18: Feature category importance
cat_mi = {}
for cat_name, cat_feats in [("Bibliometric", G_BIBLIO), ("Venue", G_VENUE),
                              ("Text stats", G_TEXTSTAT), ("NLP", G_NLP),
                              ("LDA topics", G_LDA), ("Feature eng", G_FE),
                              ("Time series", G_TS)]:
    valid = [f for f in cat_feats if f in mi_df["Feature"].values]
    if valid:
        cat_mi[cat_name] = mi_df[mi_df["Feature"].isin(valid)]["MI_score"].sum()

plt.figure(figsize=(8, 5))
cats = sorted(cat_mi.keys(), key=lambda k: cat_mi[k], reverse=True)
plt.barh(cats[::-1], [cat_mi[c] for c in cats[::-1]], color="#cc99cd")
plt.xlabel("Total MI score"); plt.title("Feature category importance (sum of MI)")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig18_category_importance.png", dpi=150); plt.close()

# Fig 19: Model comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
models = res_base["Model"]
x = np.arange(len(models))
axes[0].barh(x, res_base["F1"], color="#4c72b0")
axes[0].set_yticks(x); axes[0].set_yticklabels(models, fontsize=8)
axes[0].set_title("F1 Score"); axes[0].set_xlabel("F1")
axes[1].barh(x, res_base["ROC_AUC"], color="#55a868")
axes[1].set_yticks(x); axes[1].set_yticklabels(models, fontsize=8)
axes[1].set_title("ROC-AUC"); axes[1].set_xlabel("AUC")
axes[2].barh(x, res_base["PR_AUC"], color="#c44e52")
axes[2].set_yticks(x); axes[2].set_yticklabels(models, fontsize=8)
axes[2].set_title("PR-AUC"); axes[2].set_xlabel("PR-AUC")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig19_model_comparison.png", dpi=150); plt.close()

# --- PYSPARK SHOWCASE ---
PYSPARK_CODE = '''# PySpark ETL pipeline (requires Java + pyspark)
# pip install pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, length, regexp_replace, count, mean as spark_mean

spark = SparkSession.builder.appName("CitationETL").config("spark.driver.memory", "4g").getOrCreate()
sdf = spark.read.csv("dblp-v10.csv", header=True, inferSchema=True)
sdf = sdf.filter(col("year").between(2005, 2014))
sdf = sdf.withColumn("abstract_clean", regexp_replace(col("abstract"), r"[^a-zA-Z0-9 ]", " "))
sdf = sdf.withColumn("abstract_len", length(col("abstract_clean")))
venue_stats = sdf.groupBy("venue", "year").agg(
    count("*").alias("paper_count"), spark_mean("n_citation").alias("mean_citations"))
pdf = sdf.toPandas()
spark.stop()
'''
with open(RES_DIR / "pyspark_etl_showcase.py", "w") as f:
    f.write(PYSPARK_CODE)

# --- SUMMARY ---
best_abl = abl_df.loc[abl_df["ROC_AUC"].idxmax()]
best_model = res_base.loc[res_base["ROC_AUC"].idxmax()]
print(f"\n=== SUMMARY ===", flush=True)
print(f"Best model:    {best_model['Model']} | AUC {best_model['ROC_AUC']} | F1 {best_model['F1']}")
print(f"Best ablation: {best_abl['Experiment']} | AUC {best_abl['ROC_AUC']} | F1 {best_abl['F1']}")
print(f"Total features: {len(all_feats)}")
print(f"\nCourse mapping:")
print(f"  PySpark:        ETL pipeline in results/pyspark_etl_showcase.py")
print(f"  Statistics:     Mann-Whitney + chi-squared + Spearman in results/statistical_tests.csv")
print(f"  Feature Eng:    Polynomial interactions, log transforms in E10/E11")
print(f"  NLP:            Tokenization, sentiment, readability, POS, keywords, LDA (E8/E9)")
print(f"  Time Series:    Venue growth rate, rolling avg, trend slope (E11)")
print(f"  Supervised ML:  LR, RF, XGB, LinearSVC, GBM, Stacking (Section F)")
print(f"  EDA:            Figures fig16-19")
