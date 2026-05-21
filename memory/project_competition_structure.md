---
name: Competition Structure — Core Finding
description: The competition is cross-sectional imputation, not time-series prediction. 83.6% of test days overlap with training days.
type: project
---

The competition is NOT a temporal forecasting problem. It is a cross-sectional imputation problem.

**Why:** SO3_T.round(5) reveals 428 unique trading days in train and 512 in test — 428 overlap (83.6%). The task is: "predict the TARGET for unlabeled assets on the SAME day as labeled ones."

**How to apply:** All future modelling must exploit same-day labeled peers. The key signal is cross-sectional (within-day), not temporal. 92.1% of TARGET variance is within-day.

**Public LB is 24% of test data.** Final scores on 100% (private LB) at competition end. Variance on public scores is ~2× higher than private.

**Lesson from KNN failure:** Within-day OOF validation (80/20 split) is NOT a valid estimator of LB performance. It overfits to same-day structure. KNN got OOF R²=+0.093 but LB=-0.00042. The inverse OOF-LB pattern applies to ALL validation schemes tried so far.
