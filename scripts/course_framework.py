import sys, subprocess, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

def pip_install(pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)

BASE = Path(r"D:\cse360\cse437-citation-project")
DATA_DIR, FIG_DIR, RES_DIR = BASE / "data", BASE / "figures", BASE / "results"
for d in (FIG_DIR, RES_DIR):
    d.mkdir(exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
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

print("=== COURSE FRAMEWORK: Advanced Features + Models + Statistical Tests ===", flush=True)

TRAIN_YEARS = set(range(2005, 2013))
G_BIBLIO = ["year", "author_count", "ref_count", "has_refs"]
G_VENUE  = ["venue_missing", "venue_count_past", "venue_prior_rate"]
G_TEXTSTAT = ["title_word_count", "title_char_len", "abstract_word_count", "has_abstract"]
G_BASE = G_BIBLIO + G_VENUE + G_TEXTSTAT

print("Loading data ...", flush=True)
df = pd.read_pickle(DATA_DIR / "cohort.pkl")
df["venue"] = df["venue"].astype("string").str.strip().str.lower()
df["venue_missing"] = df["venue"].isna().astype(int)
vc = df["venue"].fillna("__missing__").value_counts()
common = vc[vc >= 20].index
df["venue_norm"] = df["venue"].where(df["venue"].isin(common), "__rare__").fillna("__missing__")

vy_r = (df.groupby(["venue_norm", "year"])["highly_cited"].mean().rename("r").reset_index()
          .sort_values(["venue_norm", "year"]))
vy_r["prev_r"] = vy_r.groupby("venue_norm")["r"].shift(1)
rate_map = dict(zip(zip(vy_r["venue_norm"], vy_r["year"]), vy_r["prev_r"]))
glob_prev = df.groupby("year")["highly_cited"].mean().shift(1).to_dict()
valid = [v for v in glob_prev.values() if pd.notna(v)]
GP = float(np.mean(valid)) if valid else 0.10

def _vp(v, y):
    r = rate_map.get((v, y))
    if r is None or pd.isna(r): r = glob_prev.get(y)
    if r is None or pd.isna(r): r = GP
    return float(r)

df["venue_prior_rate"] = [_vp(v, y) for v, y in zip(df["venue_norm"], df["year"])]

vy_n = (df.groupby(["venue_norm", "year"]).size().rename("n").reset_index()
          .sort_values(["venue_norm", "year"]))
vy_n["cum_prev"] = vy_n.groupby("venue_norm")["n"].cumsum() - vy_n["n"]
past_map = dict(zip(zip(vy_n["venue_norm"], vy_n["year"]), vy_n["cum_prev"]))
df["venue_count_past"] = [past_map.get((v, y), 0) for v, y in zip(df["venue_norm"], df["year"])]

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

# ============================================================
# A: NLP FEATURES (tokenization, sentiment, readability, POS, keywords)
# Course: NLP + Feature Engineering
# ============================================================
print("\n=== A: NLP features (NLTK + textstat) ===", flush=True)
t0 = time.time()

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import textstat

stop_words = set(stopwords.words("english"))
sid = SentimentIntensityAnalyzer()

def nlp_features(text):
    if not text or len(text) < 10:
        return {"sentiment": 0, "fk_grade": 0, "gunning_fog": 0,
                "noun_frac": 0, "verb_frac": 0, "adj_frac": 0,
                "sent_count": 0, "avg_sent_len": 0, "unique_word_ratio": 0,
                "keyword_count": 0}
    tokens = word_tokenize(text.lower())
    tokens_clean = [t for t in tokens if t.isalpha() and t not in stop_words]
    tags = pos_tag(tokens_clean)
    nouns = sum(1 for _, t in tags if t.startswith("NN"))
    verbs = sum(1 for _, t in tags if t.startswith("VB"))
    adjs  = sum(1 for _, t in tags if t.startswith("JJ"))
    total = max(len(tags), 1)
    sents = sent_tokenize(text)
    sent_lens = [len(word_tokenize(s)) for s in sents]
    return {
        "sentiment": sid.polarity_scores(text)["compound"],
        "fk_grade": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        "noun_frac": nouns / total,
        "verb_frac": verbs / total,
        "adj_frac": adjs / total,
        "sent_count": len(sents),
        "avg_sent_len": np.mean(sent_lens) if sent_lens else 0,
        "unique_word_ratio": len(set(tokens_clean)) / max(len(tokens_clean), 1),
        "keyword_count": sum(1 for t in tokens_clean if t in TOP_KEYWORDS),
    }

TOP_KEYWORDS = set()
abstracts = df["abstract"].fillna("").tolist()
cv_kw = CountVectorizer(max_features=500, stop_words="english")
try:
    kw_matrix = cv_kw.fit_transform(abstracts[:50000])
    TOP_KEYWORDS = set(cv_kw.get_feature_names_out())
except Exception:
    pass

nlp_rows = []
for i, txt in enumerate(abstracts):
    nlp_rows.append(nlp_features(txt))
    if (i + 1) % 20000 == 0:
        print(f"  NLP features: {i+1}/{len(abstracts)} ({time.time()-t0:.0f}s)", flush=True)

nlp_df = pd.DataFrame(nlp_rows, index=df.index)
for col in nlp_df.columns:
    df[f"nlp_{col}"] = nlp_df[col].values

G_NLP = [c for c in df.columns if c.startswith("nlp_")]
print(f"NLP features done: {len(G_NLP)} features in {time.time()-t0:.0f}s", flush=True)

# ============================================================
# B: LDA TOPIC MODELING (Course: NLP)
# ============================================================
print("\n=== B: LDA topic modeling (20 topics) ===", flush=True)
t0 = time.time()

cv_lda = CountVectorizer(max_features=10000, stop_words="english", min_df=5)
X_lda_all = cv_lda.fit_transform(abstracts)
lda = LatentDirichletAllocation(n_components=20, random_state=42, max_iter=15, n_jobs=-1)
X_lda_all = lda.fit_transform(X_lda_all)

for i in range(20):
    df[f"topic_{i}"] = X_lda_all[:, i]

G_LDA = [f"topic_{i}" for i in range(20)]
print(f"LDA done: {X_lda_all.shape} in {time.time()-t0:.0f}s", flush=True)
print("Top words per topic:")
terms = cv_lda.get_feature_names_out()
for i, comp in enumerate(lda.components_):
    top = [terms[j] for j in comp.argsort()[-6:][::-1]]
    print(f"  Topic {i:2d}: {', '.join(top)}")

# ============================================================
# C: FEATURE ENGINEERING (Course: Feature Engineering)
# ============================================================
print("\n=== C: Feature engineering ===", flush=True)

df["log_ref_count"] = np.log1p(df["ref_count"])
df["log_author_count"] = np.log1p(df["author_count"])
df["author_count_sq"] = df["author_count"] ** 2
df["ref_x_author"] = df["ref_count"] * df["author_count"]
df["abstract_x_venue"] = df["abstract_word_count"] * df["venue_prior_rate"]
df["title_len_x_sentiment"] = df["title_char_len"] * df["nlp_sentiment"]
df["ref_x_venue_prestige"] = df["ref_count"] * df["venue_prior_rate"]

G_FE = ["log_ref_count", "log_author_count", "author_count_sq",
        "ref_x_author", "abstract_x_venue", "title_len_x_sentiment",
        "ref_x_venue_prestige"]
print(f"Feature engineering done: {len(G_FE)} interaction features")

# ============================================================
# D: TIME SERIES VENUE TRENDS (Course: Time Series Analysis)
# ============================================================
print("\n=== D: Time series venue trends ===", flush=True)

vym = df.groupby(["venue_norm", "year"]).agg(
    vc=("highly_cited", "mean"),
    pn=("id", "count")).reset_index().sort_values(["venue_norm", "year"])

vym["growth_rate"] = vym.groupby("venue_norm")["pn"].pct_change()
vym["rolling_3yr"] = vym.groupby("venue_norm")["pn"].transform(
    lambda s: s.rolling(3, min_periods=1).mean())
vym["trend_slope"] = vym.groupby("venue_norm")["pn"].transform(
    lambda s: s.rolling(3, min_periods=2).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False))

ts_map_gr = dict(zip(zip(vym["venue_norm"], vym["year"]), vym["growth_rate"]))
ts_map_r3 = dict(zip(zip(vym["venue_norm"], vym["year"]), vym["rolling_3yr"]))
ts_map_ts = dict(zip(zip(vym["venue_norm"], vym["year"]), vym["trend_slope"]))

df["venue_growth_rate"] = [ts_map_gr.get((v, y), 0) for v, y in zip(df["venue_norm"], df["year"])]
df["venue_3yr_avg"]     = [ts_map_r3.get((v, y), 0) for v, y in zip(df["venue_norm"], df["year"])]
df["venue_trend_slope"] = [ts_map_ts.get((v, y), 0) for v, y in zip(df["venue_norm"], df["year"])]
df["venue_growth_rate"] = df["venue_growth_rate"].fillna(0).clip(-5, 5)
df["venue_trend_slope"] = df["venue_trend_slope"].fillna(0)

G_TS = ["venue_growth_rate", "venue_3yr_avg", "venue_trend_slope"]
print(f"Time series features done: {len(G_TS)} features")

# ============================================================
# E: STATISTICAL TESTS (Course: Statistics)
# ============================================================
print("\n=== E: Statistical tests ===", flush=True)

stats_rows = []
for feat in ["ref_count", "author_count", "abstract_word_count",
             "venue_prior_rate", "nlp_sentiment", "nlp_fk_grade"]:
    grp0 = df.loc[y == 0, feat].dropna()
    grp1 = df.loc[y == 1, feat].dropna()
    u_stat, u_p = mannwhitneyu(grp0, grp1, alternative="two-sided")
    r_val, r_p = spearmanr(df[feat], np.log1p(df["n_citation"]))
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

# ============================================================
# F: RETRAIN BASELINE MODELS + NEW MODELS
# ============================================================
print("\n=== F: Model training (LR, RF, XGB, LinearSVC, GBM, Stacking) ===", flush=True)

X_train_base = df.loc[tr_m, G_BASE].values
X_test_base  = df.loc[te_m, G_BASE].values
y_tr = y[tr_m].values
y_te = y[te_m].values
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train_base)
X_te_s = scaler.transform(X_test_base)

results = []
preds_all = {}

def evaluate(name, y_true, pred, prob):
    row = {
        "Model": name,
        "Accuracy": round(accuracy_score(y_true, pred), 4),
        "Precision": round(average_precision_score(y_true, prob), 4) if prob is not None else 0,
        "Recall": round(float(((pred == 1) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)), 4),
        "F1": round(f1_score(y_true, pred, zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_true, prob), 4) if prob is not None else 0,
        "PR_AUC": round(average_precision_score(y_true, prob), 4) if prob is not None else 0,
    }
    results.append(row)
    preds_all[name] = (pred, prob)
    return row

print("  LR ...", flush=True)
lr = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42))])
lr.fit(df.loc[tr_m, G_BASE], y_tr)
evaluate("01 LR (base features)", y_te, lr.predict(df.loc[te_m, G_BASE]),
         lr.predict_proba(df.loc[te_m, G_BASE])[:, 1])

print("  RF ...", flush=True)
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42)
rf.fit(df.loc[tr_m, G_BASE], y_tr)
evaluate("02 RF (base features)", y_te, rf.predict(df.loc[te_m, G_BASE]),
         rf.predict_proba(df.loc[te_m, G_BASE])[:, 1])

print("  XGB ...", flush=True)
xgb = make_xgb()
xgb.fit(df.loc[tr_m, G_BASE], y_tr)
evaluate("03 XGB (base features)", y_te, xgb.predict(df.loc[te_m, G_BASE]),
         xgb.predict_proba(df.loc[te_m, G_BASE])[:, 1])

print("  LinearSVC ...", flush=True)
svc = CalibratedClassifierCV(LinearSVC(max_iter=5000, class_weight="balanced", random_state=42))
svc.fit(X_tr_s, y_tr)
evaluate("04 LinearSVC (base features)", y_te, svc.predict(X_te_s), svc.predict_proba(X_te_s)[:, 1])

print("  GradientBoosting ...", flush=True)
gbm = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, random_state=42)
gbm.fit(df.loc[tr_m, G_BASE], y_tr)
evaluate("05 GradientBoosting (base)", y_te, gbm.predict(df.loc[te_m, G_BASE]),
         gbm.predict_proba(df.loc[te_m, G_BASE])[:, 1])

print("  Stacking ensemble ...", flush=True)
estimators = [("lr", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
              ("rf", RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42)),
              ("xgb", make_xgb())]
stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000),
                           cv=3, n_jobs=1)
stack.fit(X_tr_s, y_tr)
evaluate("06 Stacking (base features)", y_te, stack.predict(X_te_s), stack.predict_proba(X_te_s)[:, 1])

res_base = pd.DataFrame(results)
print("\nBase feature models:")
print(res_base.to_string(index=False))

# ============================================================
# G: EXPANDED ABLATION (E8-E11)
# ============================================================
print("\n=== G: Expanded ablation E8-E11 ===", flush=True)

ABLATION = {
    "E1  Content stats only":          G_TEXTSTAT,
    "E2  Bibliometric core":           G_BIBLIO,
    "E3  Bibliometric + venue":        G_BIBLIO + G_VENUE,
    "E6  E3 + TF-IDF SVD":            G_BIBLIO + G_VENUE,
    "E8  E3 + NLP features":           G_BIBLIO + G_VENUE + G_NLP,
    "E9  E3 + NLP + LDA topics":       G_BIBLIO + G_VENUE + G_NLP + G_LDA,
    "E10 E9 + feature engineering":    G_BIBLIO + G_VENUE + G_NLP + G_LDA + G_FE,
    "E11 FULL (E10 + time series)":    G_BIBLIO + G_VENUE + G_NLP + G_LDA + G_FE + G_TS,
}

tfidf_exp = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=3,
                            sublinear_tf=True)
Xtr_txt = tfidf_exp.fit_transform(df.loc[tr_m, "text"])
Xte_txt = tfidf_exp.transform(df.loc[te_m, "text"])
svd = TruncatedSVD(n_components=150, random_state=42)
S_tr = svd.fit_transform(Xtr_txt)
S_te = svd.transform(Xte_txt)

abl_rows = []
for name, cols in ABLATION.items():
    tag = name.split()[0]
    if tag == "E6":
        F_tr = pd.concat([df.loc[tr_m, G_BIBLIO + G_VENUE].reset_index(drop=True),
                          pd.DataFrame(S_tr, columns=[f"svd_{i}" for i in range(150)])], axis=1)
        F_te = pd.concat([df.loc[te_m, G_BIBLIO + G_VENUE].reset_index(drop=True),
                          pd.DataFrame(S_te, columns=[f"svd_{i}" for i in range(150)])], axis=1)
        m = make_xgb(); m.fit(F_tr, y_tr); prob = m.predict_proba(F_te)[:, 1]
    else:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"  {name}: MISSING {missing[:3]}, skipping", flush=True)
            continue
        m = make_xgb()
        m.fit(df.loc[tr_m, cols], y_tr)
        prob = m.predict_proba(df.loc[te_m, cols])[:, 1]
    pred = (prob >= 0.5).astype(int)
    abl_rows.append({
        "Experiment": name,
        "n_features": len(cols) if tag != "E6" else len(G_BIBLIO + G_VENUE) + 150,
        "Accuracy": round(accuracy_score(y_te, pred), 4),
        "F1": round(f1_score(y_te, pred, zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_te, prob), 4),
        "PR_AUC": round(average_precision_score(y_te, prob), 4),
    })

abl_df = pd.DataFrame(abl_rows)
abl_df.to_csv(RES_DIR / "ablation_expanded.csv", index=False)
print(abl_df.to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(abl_df))
ax[0].bar(x, abl_df["F1"], color="#4c72b0")
ax[0].set_xticks(x); ax[0].set_xticklabels([e.split()[0] for e in abl_df["Experiment"]], rotation=45)
ax[0].set_title("F1 by experiment")
ax[1].bar(x, abl_df["ROC_AUC"], color="#55a868")
ax[1].set_xticks(x); ax[1].set_xticklabels([e.split()[0] for e in abl_df["Experiment"]], rotation=45)
ax[1].set_title("ROC-AUC by experiment")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig16_ablation_expanded.png", dpi=150); plt.close()

# ============================================================
# H: MUTUAL INFORMATION FEATURE RANKING
# ============================================================
print("\n=== H: Mutual information ranking ===", flush=True)
all_feats = G_BASE + G_NLP + G_LDA + G_FE + G_TS
all_feats = [f for f in all_feats if f in df.columns]
X_all = df.loc[tr_m, all_feats].values
mi = mutual_info_classif(X_all, y_tr, random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({"Feature": all_feats, "MI_score": mi}).sort_values("MI_score", ascending=False)
mi_df.to_csv(RES_DIR / "mutual_information_ranking.csv", index=False)
print(mi_df.head(20).to_string(index=False))

plt.figure(figsize=(10, 8))
top20 = mi_df.head(20)
plt.barh(top20["Feature"][::-1], top20["MI_score"][::-1], color="#8172b2")
plt.xlabel("Mutual Information score")
plt.title("Top 20 features by mutual information with highly_cited")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig17_mutual_information.png", dpi=150); plt.close()

# ============================================================
# I: PYSPARK SHOWCASE (Course: PySpark - code block for report)
# ============================================================
PYSPARK_CODE = '''
# PySpark ETL pipeline (requires Java + pyspark)
# pip install pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, length, regexp_replace, count, mean as spark_mean

spark = SparkSession.builder.appName("CitationETL").config("spark.driver.memory", "4g").getOrCreate()

# Load full 1M rows
sdf = spark.read.csv("dblp-v10.csv", header=True, inferSchema=True)

# Filter to cohort years
sdf = sdf.filter(col("year").between(2005, 2014))

# Clean text
sdf = sdf.withColumn("abstract_clean", regexp_replace(col("abstract"), r"[^a-zA-Z0-9 ]", " "))
sdf = sdf.withColumn("abstract_len", length(col("abstract_clean")))

# Compute venue stats (distributed)
venue_stats = sdf.groupBy("venue", "year").agg(
    count("*").alias("paper_count"),
    spark_mean("n_citation").alias("mean_citations")
)

# Convert to pandas for ML (only after aggregation)
pdf = sdf.toPandas()
spark.stop()
'''
with open(RES_DIR / "pyspark_etl_showcase.py", "w") as f:
    f.write(PYSPARK_CODE)

# ============================================================
# SUMMARY
# ============================================================
print("\n=== SUMMARY ===", flush=True)
best_abl = abl_df.loc[abl_df["ROC_AUC"].idxmax()]
print(f"Best ablation: {best_abl['Experiment']} | AUC {best_abl['ROC_AUC']} | F1 {best_abl['F1']}")
print(f"Total features available: {len(all_feats)}")
print(f"Course mapping:")
print(f"  PySpark:        ETL pipeline code in results/pyspark_etl_showcase.py")
print(f"  Statistics:     Mann-Whitney + chi-squared + Spearman in results/statistical_tests.csv")
print(f"  Feature Eng:    Polynomial interactions, log transforms in E10/E11")
print(f"  NLP:            Tokenization, sentiment, readability, POS, keywords, LDA (E8/E9)")
print(f"  Time Series:    Venue growth rate, rolling avg, trend slope (E11)")
print(f"  Supervised ML:  LR, RF, XGB, LinearSVC, GBM, Stacking (Section F)")
print(f"  EDA:            Updated figures fig16-17")

print("\nAll outputs saved to:")
for p in sorted((RES_DIR).glob("ablation_expanded*")) + sorted(RES_DIR.glob("statistical*")) + sorted(RES_DIR.glob("mutual*")):
    print(f"  {p}")
print(f"  {FIG_DIR / 'fig16_ablation_expanded.png'}")
print(f"  {FIG_DIR / 'fig17_mutual_information.png'}")
