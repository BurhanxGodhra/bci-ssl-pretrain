# Project Narrative

The full build story, in order, including the parts that didn't work the first time.

## The question

Every EEG-BCI system — research or consumer — faces the same bottleneck: a new user typically needs 20-45 minutes of labeled calibration trials before the system produces anything usable, because EEG is so subject-specific that a model trained on other people's brains barely transfers. This project asks a narrow, testable version of that problem: can self-supervised pretraining across many subjects' unlabeled EEG reduce how many labeled trials a genuinely new, held-out user needs?

## Phase 1-2: Foundation

Environment setup surfaced the first real lesson before any modeling began — a `git init` accidentally run from the home directory rather than the project folder, caught before any commit landed. The repository structure, leakage-safe subject splits, and EEG-specific augmentations (channel dropout, temporal jitter, frequency-band masking, noise injection) were built and verified against real downloaded data (BNCI2014_001) before any model code was written.

## Phase 3: Pretraining works, mechanically

EEGNet encoder, NT-Xent loss, and a contrastive dataset with leakage-safe subject filtering baked into its constructor. The first synthetic verification matched theory almost exactly (loss at random init: 3.4305 vs. theoretical 3.4340) — a strong early signal the implementation was correct. A 3-epoch smoke test on real data showed genuine learning (loss dropping from ~4.1 to ~3.2), and a full multi-dataset run (7 subjects BNCI2014_001, 8 subjects PhysionetMI, 25 epochs) completed cleanly on Apple Silicon via MPS.

One real bug here: combining BNCI2014_001 and PhysionetMI produced a timepoint mismatch (1001 vs. 1002 samples) from resampling rounding differences — fixed by cropping to a shared minimum length as a hard invariant.

## Phase 4: The classical baseline, taken seriously

Rather than treating the Riemannian baseline as a formality, it was built to the field's actual standard (Ledoit-Wolf shrinkage covariance → tangent-space projection → regularized Logistic Regression) — because a negative comparison against a weak baseline would be worthless. Results: 72.3% on BNCI2014_001, 67.6% (with much higher subject variance) on PhysionetMI. This number became the fixed reference point for every subsequent claim.

## Phase 5: Few-shot evaluation — and the first surprise

Linear probing on the frozen pretrained encoder reached 44.8% at k=5 (20 total labeled trials) against a 25% chance floor — a real, if modest, signal. Scaling up pretraining (more subjects, more epochs) was the obvious next move — and it didn't help. K=20 accuracy went from 42.2% (small-scale) to 41.5% (full-scale), essentially flat to slightly worse.

This is where the project's actual discipline shows: rather than assuming the bigger run was simply better because the loss curve looked better, a diagnostic was pulled forward from Phase 6 to check what the encoder had actually learned.

## Phase 6 (pulled forward): The amplitude bug

A UMAP visualization plus quantitative silhouette scoring revealed the encoder's embedding space was overwhelmingly organized around *which dataset* a trial came from (+0.43 silhouette), not motor-imagery class (-0.08). Direct measurement confirmed the cause: BNCI2014_001 and PhysionetMI differ in raw signal amplitude by roughly 2.5x, giving the contrastive task a trivial shortcut that had nothing to do with real neural structure. Per-trial normalization fixed this (dataset silhouette dropped to +0.07), and re-running the full-scale pretrain plus few-shot sweep showed a genuine improvement — k=20 accuracy rose to 46.9%, with the gap over the pre-fix result *widening* with more labeled data, the signature of a real effect rather than noise.

A follow-up robustness check (UMAP at multiple `n_neighbors` settings) found a more precise, honest conclusion than either "the encoder learned nothing" or "the encoder learned subject identity": two specific subjects showed stable cross-scale clustering, most others didn't — consistent with a small number of subjects having genuinely distinctive signal characteristics, not a global identity-learning failure.

## The accuracy sprint: three failures, then a fusion

A hard look at the numbers (all methods clustering in a tight 40-47% band despite real effort) prompted a systematic sprint. Three independent, reasonable interventions were tried and ruled out in turn: dataset combination (neutral), LR schedule shape (flat LR produced lower pretraining loss but neutral-to-worse downstream accuracy — the clearest evidence yet that pretraining loss isn't a reliable proxy for what actually matters), and augmentation strength (a small real gain, insufficient alone). A mandatory random-weight-encoder control, run partway through this sprint, confirmed something important: a meaningful fraction of raw accuracy came from the EEGNet architecture's inductive bias itself, not pretraining — pretraining contributed a real but bounded increment on top of that floor.

The breakthrough came from changing strategy rather than continuing to tune: fusing the frozen SSL embedding with classical Riemannian tangent-space features at the classification step. This reached 58.2% at k=20 — an 11.3-point improvement over SSL alone, with the advantage growing with more labeled data. A regularization sweep confirmed this was a genuine ceiling, not a lucky hyperparameter (flat across a 20x range of C values).

## Building for real users

Once the accuracy story was honest and complete, the project pivoted to two applications, deliberately scoped to avoid a common failure mode: building something that looks real but silently does the wrong thing. The Device Setup Wizard uses one real LSL code path for both simulated and (eventually) real hardware, includes a signal-quality gate before calibration, and refuses to fabricate an accuracy number during calibration itself (only a separate, honestly-cued Live Test step reports real accuracy). The Benchmarking Lab evaluates multiple datasets independently rather than pretending class taxonomies merge, and validates uploaded data by exact channel identity, not just shape, before running anything.

Several real bugs surfaced during this phase too — a replay outlet that silently ran out of data mid-session, and a stale MNE data-path configuration that took two rounds of investigation to fully resolve (the first fix addressed one broken config key; a second, independently broken key required scanning the entire config rather than guessing).

## Where it stands

A defensible, honestly-reported result: SSL+Riemannian fusion reaches 80.4% of full-calibration accuracy using 86% less calibration time. Not a state-of-the-art SSL result — vanilla contrastive pretraining alone plateaued well short of the classical baseline regardless of scale or tuning — but a real, working system with the negative results kept in, not edited out.