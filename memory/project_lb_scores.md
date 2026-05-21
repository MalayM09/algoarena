---
name: All LB Scores — Public Leaderboard History
description: Every submission result on the 24% public leaderboard, in order
type: project
---

Public LB is 24% of test data. Final private LB on 100% revealed at competition end.

| Submission | Public LB | Notes |
|---|---|---|
| lgbm_baseline_v1 | -0.00002 | GroupKFold on time_ID, all features |
| fold_safe_v1 | **+0.00005** | **BEST** — GroupKFold on SO3_T quintiles, accidentally cross-sectional |
| true_time_baseline | -0.00011 | Time-ordered split, temporal overfit |
| fold_safe_v4c | -0.00005 | Low-drift features + inv_SO3_T, worse than v1 |
| transductive_v4_005 | +0.00003 | Per-day Ridge, z-scored, top-50 IC, 5% scaled |
| transductive_v3_tenth | -0.00018 | Same as above but 10% scaled (too large) |
| knn_K3_3pct | -0.00042 | KNN K=3, z-scored top-50, 3% scaled — catastrophic |

**Selected for private LB:** fold_safe_v1 + transductive_v4_005

**Inverse OOF-LB pattern:** Higher OOF R² → worse LB, without exception. The only model with positive LB has tiny OOF (+0.000544). fold_safe_v1 is the only reliable baseline.
