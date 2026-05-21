---
name: OOF Validation Is Unreliable for This Competition
description: Every high-OOF model got worse LB. Within-day OOF also fails. Only trust LB directly.
type: feedback
---

Do NOT use OOF R² to select submissions for this competition.

**Why:** The inverse OOF-LB pattern held without exception across 7+ experiments. Higher OOF always meant worse LB. This includes:
- Temporal GroupKFold: high OOF → LB negative
- Random KFold on test-like rows: OOF=+0.023 → never submitted (known bad)
- Within-day 80/20 split (transductive): OOF=+0.015 → LB=+0.00003 (barely positive)
- Within-day KNN OOF: OOF=+0.093 → LB=-0.00042 (worst ever)

**How to apply:** When choosing between submissions, prefer the one with LOWER OOF R². Models with OOF > 0.001 should not be submitted. fold_safe_v1 (OOF=+0.000544) is the ceiling of what's trustworthy.
