# Predicting Highly Cited Research Papers (CSE437 Project)

**Research question:** Which factors make a research paper highly cited - and do textual signals
(TF-IDF / transformer embeddings) add predictive value beyond bibliometric features?

**Data:** AMiner dblp-v10 (`dblp-v10.csv.zip`, ~620 MB zipped / 1.37 GB csv, 1,000,000 papers,
1956-2017). Cohort used: papers **2005-2014**, stratified sample of **100,000** papers.

---

## Files

```
citation_prediction_project.ipynb   <- THE deliverable (runs in Colab or locally)
build_notebook.py                   <- generator script (optional; regenerates the ipynb)
data/      cohort.pkl + saved embeddings
figures/   all charts (fig1..fig11, auto-saved by the notebook)
results/   model_comparison.csv, ablation_results.csv, sensitivity_analysis.csv,
           ALL_RESULTS.csv, error_analysis.csv, eda_notes.txt, significance_tests.txt
requirements.txt
```

## How to run

### Google Colab (recommended)
1. Upload `citation_prediction_project.ipynb` and open it.
2. Run cells top-to-bottom. When prompted, upload `dblp-v10.csv.zip` (~620 MB).
3. Optional: set `RUN_EMBEDDINGS = True` in Section 1 to also train the SBERT hybrid
   (adds ~15 min CPU). `RUN_SHAP = True` adds SHAP plots automatically.
4. Download `results/ALL_RESULTS.csv` + `figures/` for your report.

### Locally
```powershell
pip install -r requirements.txt
$env:LOCAL_DBLP = "C:\path\to\dblp-v10.csv"
jupyter lab citation_prediction_project.ipynb
```

## Pipeline design (maps to report phases)

| Notebook section | Report phase | Notes |
|---|---|---|
| Load (chunked stream) | Phase 2 | Only 2005-2014 kept in memory |
| Cleaning/dedup | Phase 2 | id dedup + normalized-title+year dedup, missing-value handling |
| Features | Phase 2 | team size, reference count, title/abstract stats, leak-safe venue features |
| Target | Phase 2 | **top 10% of citations WITHIN publication year** (controls paper age) |
| EDA | Phase 2 | figs 1-6 incl. the n_citation==50 artifact chart |
| Temporal split | Phase 3 | train 2005-2012, test 2013-2014 (simulates deployment) |
| Baseline/LR/RF/XGB/TF-IDF | Phase 3-4 | required >= 3 models + NLP baseline |
| Importance + SHAP | Phase 4 | permutation importance (leak-free view) + SHAP |
| Ablation E1-E7 | Phase 4 | mirrors Fu & Aliferis (2008): content-only vs biblio-only vs combined |
| Bootstrap CI + McNemar | Phase 4 | statistical comparison of models |
| Error analysis | Phase 4 | FP/FN breakdown by year, venue popularity, abstract length |
| SBERT hybrid (optional) | Phase 3-4 | Sentence-BERT embeddings, NOT GPT |
| Sensitivity | Phase 4 | top 5/10/20% thresholds; artifact-exclusion robustness |

## Methodology rules baked into the code

- `n_citation` is NEVER a feature (would leak the label).
- Venue features use **past years only** (venue_count_past, venue_prior_rate).
- Temporal test set: no future papers are mixed into training folds.
- Class imbalance handled via class_weight / scale_pos_weight; never trust accuracy alone.

## Headline verified results (temporal test, n=19,198)

- Majority baseline: F1 0.000, ROC-AUC 0.500
- Logistic Regression (tabular): F1 0.263, ROC-AUC 0.850
- Random Forest: F1 0.256, ROC-AUC 0.826
- XGBoost: F1 0.262, ROC-AUC 0.833
- TF-IDF text-only: F1 0.203, ROC-AUC 0.810
- **Best: E6 = bibliometric + venue + TF-IDF(SVD): F1 0.307, ROC-AUC 0.862, PR-AUC 0.264**
- Delta AUC text contribution: +0.0081, bootstrap 95% CI [+0.0001, +0.0158]; McNemar p=0.62
- Top permutation importances: venue_prior_rate, ref_count, abstract_word_count
- Excluding the n_citation==50 artifact: F1 rises to 0.44 while AUC stays ~0.83

## Known data caveat (discuss in Limitations)

~30% of cohort papers have `n_citation` EXACTLY 50 - an AMiner imputation default.
The notebook quantifies it (fig1), avoids using it as a feature, and runs a robustness
check excluding these rows (Section 16).

## Ideas to strengthen further (ranked)

1. Topic-growth feature (Sjogarde & Didegah 2022): yearly keyword frequency growth per topic cluster.
2. Interdisciplinarity score (Lyutov et al. 2024): train a field classifier on abstracts; P(misclassified) = interdisciplinarity.
3. Author collaboration network: degree/past productivity of authors from the full 1M graph.
4. Calibration curves + Brier score for the best model.
5. LightGBM/CatBoost as extra model family; hyperparameter search (RandomizedSearchCV).
6. Regression head: predict log1p(citations) and compare against classification framing.
7. LLM zero-shot arm (Ye et al. 2026): prompt an API LLM on 500 test abstracts, compare AUC.
