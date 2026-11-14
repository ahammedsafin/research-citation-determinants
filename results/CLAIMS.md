# Claim Tests (add-on experiments)

## H1. Where does textual information live? Text value conditioned on venue visibility - VERDICT: NOT SUPPORTED
Delta-AUC from adding TF-IDF text by venue-visibility stratum (train-quartile buckets):
- V1_lowest: +0.0491 (CI [+0.0115, +0.0884])
- V2_low: +0.0386 (CI [+0.0225, +0.0539])
- V3_mid: +0.0196 (CI [+0.0015, +0.0390])
- V4_high: +0.1142 (CI [+0.0766, +0.1548])
Text adds the most where bibliometric signals are WEAKEST within the stratum
(biblio-only AUC 0.705 in V4_high
vs 0.801 at its best elsewhere); text restores discrimination to
0.819.
Reframes Lyutov et al. (2024): venue prestige conditions not just citation PAYOFF but how much
PREDICTIVE INFORMATION the metadata channel carries; semantic content compensates exactly there.
See claims_H1_stratified.csv, fig12.

## H2. The exact-50 artifact distorts reported metrics, not rankings - VERDICT: SUPPORTED
30.4% of the cohort has n_citation == 50 (imputation default).
Under a +-1 citation perturbation, labels of exact-50 papers flip 5.2%
of the time vs 0.2% for other papers.
Excluding artifact rows: F1 0.2608 -> 0.4405,
AUC 0.8249 -> 0.8289.
See claims_H2_*.csv, fig13.

## H3. Semantic novelty adds independent predictive signal - VERDICT: WEAK/NULL - discuss honestly
novelty = 1 - mean cosine similarity to prior-two-years' abstract centroids (leak-safe).
Alone AUC 0.5061; added to E6: AUC 0.8622 -> 0.8624,
bootstrap Delta CI [-0.0034, +0.0035], XGB gain share 0.0092.
See claims_H3_novelty.csv, fig14. Calibration of best model: fig15 (Brier=0.0823).
