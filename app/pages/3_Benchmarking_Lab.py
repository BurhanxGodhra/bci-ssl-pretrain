"""
BCI Engineer's Benchmarking Lab: test minimum calibration-trial
requirements against our validated SSL+Riemannian fusion pipeline,
across one or more datasets (built-in and/or uploaded) for comparison.

Note on scope: datasets are evaluated INDEPENDENTLY, never combined at
the evaluation/label level -- different datasets can have different,
non-comparable class taxonomies (e.g. BNCI2014_001's 4 classes vs.
PhysionetMI's 5). Combining is only valid at the encoder-PRETRAINING
level (unsupervised), which this project already supports via
scripts/pretrain.py's train_multi(). This page compares how one fixed,
already-pretrained encoder generalizes ACROSS datasets, not a
combined-label evaluation.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.data.custom_dataset import REQUIRED_CHANNELS, validate_and_adapt_upload
from src.data.loaders import DATASET_REGISTRY, load_dataset
from src.data.splits import get_or_create_split
from src.finetune.fusion_eval import kshot_fusion_eval
from src.finetune.linear_probe_eval import load_pretrained_encoder
from src.utils.device import get_device

st.set_page_config(page_title="Benchmarking Lab", layout="centered")

st.title("Benchmarking Lab")
st.markdown(
    "Estimate the minimum number of calibration trials needed to reach a "
    "target accuracy, using the SSL + Riemannian fusion method validated "
    "in this project. Compare results across multiple datasets against "
    "the same pretrained encoder."
)

CHECKPOINTS = {
    "BNCI2014_001-only (best validated)": "checkpoints/encoder_bnci2014_001_only_e25_flat_gentleaug.pt",
    "Combined BNCI2014_001 + PhysionetMI": "checkpoints/encoder_multi_full_e25.pt",
}

st.divider()
st.subheader("Configuration")

checkpoint_label = st.selectbox("Encoder checkpoint", list(CHECKPOINTS.keys()))
checkpoint_path = CHECKPOINTS[checkpoint_label]

if not Path(checkpoint_path).exists():
    st.error(f"Checkpoint not found at {checkpoint_path}. Train it first via scripts/pretrain.py.")
    st.stop()

target_accuracy = st.slider("Target accuracy", min_value=0.30, max_value=0.90, value=0.55, step=0.05)
k_candidates = st.multiselect(
    "k values to test (trials per class)",
    options=[1, 3, 5, 10, 15, 20],
    default=[1, 5, 10, 20],
)
n_draws = st.slider("Draws per k (higher = more reliable, slower)", 3, 15, 8)

st.divider()
st.subheader("Data Sources")
st.caption(
    "Select one or more built-in datasets, and/or upload your own. Each source "
    "is evaluated independently (different datasets have different, "
    "non-comparable class sets), then results are shown side-by-side."
)

built_in_keys = st.multiselect(
    "Built-in datasets",
    options=list(DATASET_REGISTRY.keys()),
    default=["bnci2014_001"],
)

use_custom = st.checkbox("Also include an uploaded dataset")
custom_epochs = None

if use_custom:
    st.caption(
        f"Requires all {len(REQUIRED_CHANNELS)} of these exact channels (any order): "
        f"{', '.join(REQUIRED_CHANNELS)}"
    )
    st.caption(
        "Upload a .npz file with keys 'X' (n_trials, n_channels, n_timepoints) "
        "and 'y' (n_trials,) integer class labels."
    )

    uploaded_file = st.file_uploader("Upload .npz file", type=["npz"])
    channel_names_input = st.text_input(
        "Channel names (comma-separated, matching X's channel axis order)",
        placeholder="C1,C2,C3,...",
    )
    sfreq_input = st.number_input("Sample rate (Hz)", min_value=1.0, value=250.0)
    class_names_input = st.text_input(
        "Class names (comma-separated, matching label integers 0,1,2,...)",
        placeholder="left_hand,right_hand,feet,tongue",
    )

    if uploaded_file is not None and channel_names_input and class_names_input:
        if st.button("Validate Upload"):
            data = np.load(uploaded_file)
            if "X" not in data or "y" not in data:
                st.error("File must contain both 'X' and 'y' arrays.")
            else:
                result = validate_and_adapt_upload(
                    X=data["X"],
                    y=data["y"],
                    channel_names=[c.strip() for c in channel_names_input.split(",")],
                    sfreq=sfreq_input,
                    class_names=[c.strip() for c in class_names_input.split(",")],
                )
                if result["ok"]:
                    st.session_state.custom_epochs = result["epochs"]
                    st.success(
                        f"Validated: {result['epochs'].X.shape[0]} trials, "
                        f"{len(result['epochs'].label_map)} classes, montage compatible."
                    )
                else:
                    st.error(result["error"])
                    if "missing_channels" in result:
                        st.write(f"Missing: {result['missing_channels']}")
                        st.write(f"Found: {result['found_channels']}")

    if "custom_epochs" in st.session_state:
        custom_epochs = st.session_state.custom_epochs

st.divider()

if st.button("Run Benchmark", type="primary"):
    if not k_candidates:
        st.warning("Select at least one k value.")
        st.stop()
    if not built_in_keys and custom_epochs is None:
        st.warning("Select at least one built-in dataset, or validate an upload.")
        st.stop()

    device = get_device()
    with st.spinner("Loading encoder..."):
        encoder = load_pretrained_encoder(checkpoint_path, device)

    # Build the list of independent sources to evaluate
    sources = []
    for dk in built_in_keys:
        sources.append({"label": dk, "kind": "built_in", "key": dk})
    if custom_epochs is not None:
        sources.append({"label": "Uploaded dataset", "kind": "custom", "epochs": custom_epochs})

    all_results = {}  # {source_label: {k: accuracy}}
    n_classes_by_source = {}

    overall_progress = st.progress(0.0)
    total_steps = len(sources) * len(k_candidates)
    step = 0

    for source in sources:
        label = source["label"]
        st.write(f"Evaluating: **{label}**")

        if source["kind"] == "built_in":
            dk = source["key"]
            split = get_or_create_split(dk, DATASET_REGISTRY[dk]["subjects"])
            holdout_subjects = split["holdout_subjects"]
            epochs = load_dataset(dk, subjects=holdout_subjects)
        else:
            epochs = source["epochs"]
            holdout_subjects = sorted(set(epochs.subject_ids.tolist()))

        n_classes_by_source[label] = len(epochs.label_map)
        source_results = {}

        for k in sorted(k_candidates):
            subj_means = []
            for subj in holdout_subjects:
                mask = epochs.subject_ids == subj
                X_subj, y_subj = epochs.X[mask], epochs.y[mask]
                if len(X_subj) == 0:
                    continue
                r = kshot_fusion_eval(encoder, X_subj, y_subj, k=k, n_draws=n_draws, device=device)
                subj_means.append(r["mean_accuracy"])
            source_results[k] = sum(subj_means) / len(subj_means) if subj_means else 0.0

            step += 1
            overall_progress.progress(step / total_steps)

        all_results[label] = source_results

    st.divider()
    st.subheader("Results")

    # Comparison table: rows = k, columns = sources
    table_data = {"k": sorted(k_candidates)}
    for label, res in all_results.items():
        table_data[label] = [f"{res[k]:.1%}" for k in sorted(k_candidates)]
    st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Minimum Trials to Reach Target")

    for label, res in all_results.items():
        meeting_k = [k for k in sorted(res) if res[k] >= target_accuracy]
        n_classes = n_classes_by_source[label]
        if meeting_k:
            min_k = min(meeting_k)
            st.markdown(
                f"**{label}**: minimum **{min_k} trials/class** "
                f"({min_k * n_classes} total) to reach {target_accuracy:.0%}."
            )
        else:
            best_k = max(res, key=lambda k: res[k])
            st.markdown(
                f"**{label}**: target not reached with tested k values "
                f"(best: {res[best_k]:.1%} at k={best_k})."
            )