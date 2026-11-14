"""Stage 1: NLP features + LDA topic modeling (slow, ~15-20 min)"""
import sys, time, warnings, pickle
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

print("=== STAGE 1: NLP + LDA features ===", flush=True)

BASE = Path(r"D:\cse360\cse437-citation-project")
DATA_DIR = BASE / "data"

print("Loading data ...", flush=True)
df = pd.read_pickle(DATA_DIR / "cohort.pkl")
df["venue"] = df["venue"].astype("string").str.strip().str.lower()
df["venue_missing"] = df["venue"].isna().astype(int)

print(f"Loaded {len(df)} rows", flush=True)

# --- LEAK-SAFE VENUE FEATURES ---
print("\n=== Computing venue features ===", flush=True)
TRAIN_YEARS = set(range(2005, 2013))
vc = df["venue"].fillna("__missing__").value_counts()
common = vc[vc >= 20].index
df["venue_norm"] = df["venue"].where(df["venue"].isin(common), "__rare__").fillna("__missing__")

# venue_prior_rate
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

# venue_count_past
vy_n = (df.groupby(["venue_norm", "year"]).size().rename("n").reset_index()
          .sort_values(["venue_norm", "year"]))
vy_n["cum_prev"] = vy_n.groupby("venue_norm")["n"].cumsum() - vy_n["n"]
past_map = dict(zip(zip(vy_n["venue_norm"], vy_n["year"]), vy_n["cum_prev"]))
df["venue_count_past"] = [past_map.get((v, y), 0) for v, y in zip(df["venue_norm"], df["year"])]

# abstract_x_venue interaction
df["abstract_x_venue"] = df["abstract_word_count"] * df["venue_prior_rate"]

print(f"Venue features done. Columns: venue_missing, venue_count_past, venue_prior_rate, abstract_x_venue", flush=True)

# --- NLP FEATURES ---
print("\n=== NLP features (tokenization, sentiment, readability, POS) ===", flush=True)
t0 = time.time()

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import textstat

stop_words = set(stopwords.words("english"))
sid = SentimentIntensityAnalyzer()

TOP_KEYWORDS = {"method", "approach", "novel", "propose", "algorithm", "model",
                "network", "data", "result", "performance", "system", "framework",
                "analysis", "effect", "significantly", "improve", "show", "based",
                "present", "study", "paper", "existing", "new", "feature",
                "training", "learning", "deep", "classif", "cluster", "regression",
                "neural", "graph", "image", "text", "generat", "optimi", "predict"}

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

abstracts = df["abstract"].fillna("").tolist()
nlp_rows = []
for i, txt in enumerate(abstracts):
    nlp_rows.append(nlp_features(txt))
    if (i + 1) % 20000 == 0:
        print(f"  NLP: {i+1}/{len(abstracts)} ({time.time()-t0:.0f}s)", flush=True)

nlp_df = pd.DataFrame(nlp_rows, index=df.index)
for col in nlp_df.columns:
    df[f"nlp_{col}"] = nlp_df[col].values

print(f"NLP done: {len([c for c in df.columns if c.startswith('nlp_')])} features in {time.time()-t0:.0f}s", flush=True)

# --- LDA TOPICS ---
print("\n=== LDA topic modeling (15 topics, reduced for speed) ===", flush=True)
t0 = time.time()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

cv_lda = CountVectorizer(max_features=8000, stop_words="english", min_df=5, max_df=0.95)
X_lda_all = cv_lda.fit_transform(abstracts)
lda = LatentDirichletAllocation(n_components=15, random_state=42, max_iter=10, n_jobs=-1)
X_lda_all = lda.fit_transform(X_lda_all)

for i in range(15):
    df[f"topic_{i}"] = X_lda_all[:, i]

terms = cv_lda.get_feature_names_out()
print("Top words per topic:")
for i, comp in enumerate(lda.components_):
    top = [terms[j] for j in comp.argsort()[-6:][::-1]]
    print(f"  Topic {i:2d}: {', '.join(top)}")

print(f"LDA done in {time.time()-t0:.0f}s", flush=True)

# --- FEATURE ENGINEERING ---
print("\n=== Feature engineering ===", flush=True)
df["log_ref_count"] = np.log1p(df["ref_count"])
df["log_author_count"] = np.log1p(df["author_count"])
df["author_count_sq"] = df["author_count"] ** 2
df["ref_x_author"] = df["ref_count"] * df["author_count"]
df["title_len_x_sentiment"] = df["title_char_len"] * df["nlp_sentiment"]
df["ref_x_venue_prestige"] = df["ref_count"] * df["venue_prior_rate"]

# --- TIME SERIES VENUE TRENDS ---
print("\n=== Time series venue trends ===", flush=True)
vym = df.groupby(["venue_norm", "year"]).agg(
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

print(f"Time series done", flush=True)

# --- SAVE ---
out = DATA_DIR / "cohort_expanded.pkl"
df.to_pickle(out)
print(f"\nSaved expanded features to {out} ({df.shape[1]} columns)", flush=True)
print(f"New feature columns: {[c for c in df.columns if c.startswith(('nlp_', 'topic_', 'log_', 'author_count_sq', 'ref_x_', 'venue_growth', 'venue_3yr', 'venue_trend'))]}")
