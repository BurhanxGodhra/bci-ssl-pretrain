# Engineering Decision Log

Chronological record of the decisions that materially shaped this project's design and results — including several reversals, once evidence contradicted an earlier assumption. Numbered for cross-reference from `PROJECT_NARRATIVE.md` and `BENCHMARKS.md`.

---

**D-001 — Editable install (`pyproject.toml`) over path hacks**
*Phase 2.* `ModuleNotFoundError: No module named 'src'` when running test scripts from `tests/`. Rather than `sys.path` manipulation, added a minimal `pyproject.toml` and `pip install -e .`. This is the correct production pattern, not a workaround, and was missing from the initial environment setup.

**D-002 — Subject-level split manifest as a structural leakage firewall**
*Phase 2.* Rather than trusting callers to pass the correct subject list, `PretrainContrastiveDataset` filters through a committed JSON split manifest (`data/splits/*.json`) in its constructor. Pretraining on a holdout subject now requires actively bypassing the class, not just forgetting a parameter. Verified with an explicit assertion test before any real training run.

**D-003 — EEGNet-style encoder over a Transformer**
*Phase 3.* Chosen for two reasons: (1) parameter efficiency appropriate to a pretraining set of a few thousand trials, where a Transformer would be expected to overfit or need substantially more data; (2) the temporal→spatial convolution structure mirrors what the classical Riemannian baseline computes explicitly (spatial covariance), making the eventual SSL-vs-classical comparison more apples-to-apples.

**D-004 — NT-Xent temperature=0.5, embed_dim=128, batch_size=64 (initial)**
*Phase 3.* Standard SimCLR defaults, not tuned at this point — later revisited (D-010) once downstream accuracy plateaued.

**D-005 — Channel alignment by name intersection, not position index**
*Phase 3.* Combining BNCI2014_001 (22ch) and PhysionetMI (64ch) required a shared feature space. Chose common-channel-name intersection over a fixed subset list, so the same code generalizes to any future dataset pair.

**D-006 — Per-trial z-score normalization (the single highest-impact fix in the project)**
*Phase 6.* A UMAP + silhouette-score diagnostic revealed the encoder's embedding space was dominated by which dataset a trial came from (+0.43 silhouette), not motor-imagery class. Root cause: BNCI2014_001 (std≈7.9) and PhysionetMI (std≈19.9) differ in raw amplitude by ~2.5x, giving contrastive pretraining a trivial shortcut. Fixed by normalizing each trial independently (not a global per-dataset correction, which would leave residual trial-to-trial variance as an exploitable signal). Dataset silhouette dropped to +0.07 post-fix; downstream k=20 accuracy improved from 41.5% to 46.9%.

**D-007 — Explicit shape-assertion at the channel-name source, not downstream**
*Accuracy sprint.* `load_dataset()`'s `channel_names` were pulled from the raw MNE recording (includes non-EEG EOG/STI channels) while the actual data array had already been filtered to EEG-only by MOABB's paradigm. This silently self-corrected when combining two datasets (name intersection filtered out the spurious channels by luck) but crashed immediately on single-dataset use. Fixed by filtering to EEG-only at the source and adding an assertion that channel-name count matches the array's channel dimension — converts a class of latent bug into an immediate, loud failure.

**D-008 — Mandatory random-weight encoder control**
*Accuracy sprint.* Before trusting any pretrained-encoder accuracy number, ran the identical evaluation pipeline against a same-architecture, never-trained encoder. Result: 39.1% at k=20, confirming the EEGNet architecture's inductive bias carries real signal independent of pretraining. This reframed every subsequent result as "pretraining contributes +N pts over this floor," not "pretraining alone explains this accuracy."

**D-009 — Reported full fine-tune losing to linear probing, not just the winning method**
*Phase 5.* Full fine-tuning underperformed frozen linear probing at every k≥5, with the gap *widening* with more data (opposite of an overfitting-that-resolves-with-data hypothesis). Kept and reported this as a real finding rather than omitting the weaker method — the widening-gap pattern is itself informative about how the unfrozen encoder behaves.

**D-010 — Three negative-result experiments run and reported before the winning method**
*Accuracy sprint.* Systematically tested: (1) dataset combination vs. BNCI-only pretraining — neutral; (2) flat vs. cosine LR schedule — flat produced lower loss but neutral-to-worse accuracy, the first clear evidence loss isn't a reliable downstream proxy; (3) augmentation strength (gentle vs. strong) — small real gain (+1.2pt), insufficient alone. All three are documented in `BENCHMARKS.md` rather than discarded, since ruling out plausible levers is genuine information.

**D-011 — SSL + Riemannian feature fusion, not further SSL tuning**
*Accuracy sprint.* After three tuning attempts converged to the same ~45-47% ceiling, switched strategy: concatenate the frozen SSL embedding with Riemannian tangent-space features (fit on the k-shot support set only, to avoid leakage into the query set) and classify jointly. Result: 58.2% at k=20, an +11.3pt improvement over SSL-only linear probing, with the gap widening with k — a genuine structural improvement, not noise. Confirmed not a regularization artifact via a C-value sweep (flat across C=0.05-1.0).

**D-012 — Loop the simulated LSL replay outlet indefinitely**
*Device Setup Wizard.* The replay outlet streamed its pre-loaded holdout-subject data once, at real wall-clock rate (~80s for 20 trials), then went silent. Wizard sessions (countdown + cue + pull, across 20 trials) routinely exceeded that window, stalling calibration mid-session. Fixed by wrapping the stream loop in `while True`. Diagnosed correctly on the first pass but initially appeared unfixed on retest — root cause was Streamlit's hot-reload not re-importing already-loaded Python modules; required a full process restart, not just a file save, to take effect.

**D-013 — MNE_DATA config resolved by scanning the full config dict, not patching one key**
*Benchmarking Lab.* A stale `MNE_DATA` value from an unrelated local project caused `FileNotFoundError` on PhysionetMI downloads. First fix (resetting the generic `MNE_DATA` key) didn't resolve a second occurrence — a dataset-specific downloader key (tied to PhysionetMI's `EEGBCI` signature) was independently broken and bypassed the generic key's fallback-to-default logic. Resolved by scanning the entire MNE config dictionary for any value pointing at the broken path, rather than guessing keys one at a time.

**D-014 — Benchmarking Lab evaluates datasets independently, never merges class labels**
*Benchmarking Lab.* BNCI2014_001 (4 classes) and PhysionetMI (5 classes, different taxonomy) cannot be combined into one evaluation — a classifier can't meaningfully predict among mismatched label sets. Dataset "mixing" is only valid at the unsupervised pretraining level (already supported via `train_multi()`); the Lab's multi-dataset feature runs the same encoder through independent per-dataset evaluations and compares results side-by-side, rather than pretending a combined evaluation is possible.

**D-015 — Custom dataset upload requires exact channel identity, not just channel count**
*Benchmarking Lab.* The encoder's spatial convolution weights are tied to specific learned electrode positions (C3 carries different meaning than Pz). A dataset with the right *count* of channels but different or reordered identities would silently produce meaningless output. The upload validator checks against the exact required 22-channel list by normalized name, reorders to match, and reports precisely which channels are missing if validation fails — rather than accepting anything with the right shape.