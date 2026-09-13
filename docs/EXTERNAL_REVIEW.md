# Technical Review

**Note on provenance**: unlike a genuine third-party review, this was self-conducted — applying the same adversarial scrutiny an outside reviewer would, rather than presenting it as independent verification it isn't. If a real external review happens later, it should replace this document rather than be appended alongside it.

## Summary

The project's strongest asset is its negative-result discipline: three tuning interventions were tried, ruled out with real evidence, and reported rather than discarded. Its weakest points are statistical (small holdout N, no independent test set for the fusion method's hyperparameters) and scope (no real-hardware validation, a fixed montage requirement that limits practical reuse). Neither weakness invalidates the headline claim, but both bound how strongly it should be stated.

## Findings

### 1. Holdout set is small (2 subjects per dataset)

**Issue**: Every reported few-shot number is a mean over exactly 2 held-out subjects. Subject-level variance is visible everywhere in this project's own data (BNCI subject 1: 80.7% vs. subject 4: 63.9% on the classical baseline; PhysionetMI subject 6: 49.4% vs. subject 7: 85.7%) — a 2-subject mean cannot distinguish "this method generalizes" from "we got a favorable pair of holdout subjects."

**Response**: Accurate criticism, not previously stated plainly enough in the README. The `n_holdout=2` split was fixed early (Phase 2) as a reasonable default at the time and never revisited once the project's scope grew. This should be read as evidence from 2 subjects per dataset, not a population-level claim. A fair fix is either more holdout subjects (fewer pretrain subjects, a real tradeoff) or repeating the whole pipeline across several different random holdout splits and reporting variance across splits, not just across k-shot draws within one split.

### 2. The C-regularization sweep is not a proper validation set

**Issue**: The C sweep that "confirmed" 58.2% wasn't a regularization artifact was run and evaluated on the same 2 holdout subjects used to report the final number. This is a mild form of the same test-set-reuse concern that caused a real leakage bug in the reference project this documentation style is modeled on (see that project's D-013) — here it's milder (hyperparameter selection, not model training, touched the test set) but the same category of concern.

**Response**: Correct, and worth being explicit about rather than letting the "confirmed not an artifact" framing overstate its rigor. The flatness of the sweep (58.0-58.2% across C=0.05-1.0) makes overfitting-to-the-holdout-set via this specific sweep unlikely in practice, but the methodology doesn't rule it out cleanly. A proper fix would hold out a third split (pretrain / hyperparameter-validation / final-test) — not done here, and worth flagging as a real gap rather than asserting confidence the evidence doesn't fully support.

### 3. No real-hardware validation, at all

**Issue**: Every result, including the Device Setup Wizard's "verified working" claims, traces back to replayed MOABB research data. Real EEG headsets differ from research-grade recording systems in impedance behavior, motion artifact, consumer-grade amplifier noise floors, and — most relevantly to this project's fixed 22-channel requirement — montage. The wizard's compatibility gate is real and honest, but it has never actually been exercised against a device that fails it, or one that passes it.

**Response**: Accurate and already stated as a limitation in the README, but worth restating here at review-level severity rather than as a one-line caveat: this project has built a well-tested simulation of a real-hardware pipeline, not a validated one. The distinction matters for anyone considering deploying this rather than reading it as a portfolio piece.

### 4. "80.4% of ceiling" compares against a full-data baseline evaluated differently than the few-shot methods

**Issue**: The Riemannian ceiling (72.3%) is a 5-fold cross-validation mean over all ~576 trials per subject. The fusion method's 58.2% is a mean over 10 random k=20 support-set draws, each evaluated against the remaining query trials. These are both legitimate accuracy estimates, but they are not computed identically — the "% of ceiling" framing implies a more direct comparison than the underlying methodologies fully support.

**Response**: A fair point about presentation, not correctness — both numbers are real, validated accuracies, just from different evaluation protocols. The "% of ceiling" language should be read as "how much of the fully-calibrated classical baseline's performance is recovered," not as a claim that the two numbers are computed via an identical procedure. Worth a clarifying footnote in the README rather than a methodology change, since re-running the Riemannian baseline as 10 draws of k=20 support/query splits wouldn't actually change its answer meaningfully — full-data 5-fold CV is the correct way to report the ceiling, it just isn't the same protocol as the few-shot arm by construction.

### 5. Vanilla NT-Xent was tuned along fairly conventional axes only

**Issue**: The three ruled-out interventions (dataset scale, LR schedule, augmentation strength) are all standard SSL hyperparameters. None of them tests whether a fundamentally different contrastive setup — larger batch/more negatives, a different temperature, a supervised-contrastive variant using the pretrain-subjects' available labels — might close more of the gap. The conclusion "vanilla contrastive pretraining has a real ceiling here" is stated more strongly than the evidence (three specific, conventional experiments) fully supports.

**Response**: Fair. The roadmap already lists a supervised-contrastive or prototypical-network objective as the most promising next lever, which implicitly concedes this point — but the README's "plateaued regardless of data scale, schedule, or augmentation tuning" phrasing should be read as "regardless of these three specific interventions," not as an exhaustive search of the SSL hyperparameter space.

## Overall assessment

The core finding — SSL+Riemannian fusion meaningfully reduces calibration burden relative to SSL alone or the classical baseline alone — is real and well-supported by the evidence collected. The main risk in how it's communicated is precision: statements should consistently read as "on these 2 holdout subjects per dataset, under this specific evaluation protocol" rather than implied population-level claims. None of the findings above suggest the project's central result is wrong; they suggest the confidence interval around it is wider than a first read of the README implies.