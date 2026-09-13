"""
BCI Engineer's Benchmarking Lab: test minimum calibration-trial
requirements against our validated SSL+Riemannian fusion pipeline,
choosing which pretrained encoder checkpoint to evaluate.
"""
from pathlib import Path

import streamlit as st

from src.data.loaders import DATASET_REGISTRY, load_dataset
from src.data.splits import get_or_create_split
from src.finetune.fusion_eval import kshot_fusion_eval
from src.finetune.linear_probe_eval import load_pretrained_encoder
from src.utils.device import get_device

st.set_page_config(page_title="Benchmarking Lab", layout="centered")

st.title("Benchmarking Lab")
st.markdown(
    "Estimate the minimum number of calibration trials needed to reach a "
    "target accuracy on a chosen dataset, using the SSL + Riemannian fusion "
    "method validated in this project."
)

CHECKPOINTS = {
    "BNCI2014_001-only (best validated)": "checkpoints/encoder_bnci2014_001_only_e25_flat_gentleaug.pt",
    "Combined BNCI2014_001 + PhysionetMI": "checkpoints/encoder_multi_full_e25.pt",
}

st.divider()
st.subheader("Configuration")

col1, col2 = st.columns(2)
with col1:
    dataset_key = st.selectbox("Evaluation dataset", list(DATASET_REGISTRY.keys()))
with col2:
    checkpoint_label = st.selectbox("Encoder checkpoint", list(CHECKPOINTS.keys()))
    checkpoint_path = CHECKPOINTS[checkpoint_label]

st.caption(
    "Note: evaluation happens against one dataset's class taxonomy at a time, "
    "even when the selected encoder was pretrained across multiple datasets. "
    "Class labels are not comparable across datasets with different taxonomies."
)

target_accuracy = st.slider("Target accuracy", min_value=0.30, max_value=0.90, value=0.55, step=0.05)
k_candidates = st.multiselect(
    "k values to test (trials per class)",
    options=[1, 3, 5, 10, 15, 20],
    default=[1, 5, 10, 20],
)
n_draws = st.slider("Draws per k (higher = more reliable, slower)", 3, 15, 8)

if not Path(checkpoint_path).exists():
    st.error(f"Checkpoint not found at {checkpoint_path}. Train it first via scripts/pretrain.py.")
    st.stop()

if st.button("Run Benchmark", type="primary"):
    if not k_candidates:
        st.warning("Select at least one k value.")
        st.stop()

    device = get_device()
    with st.spinner("Loading encoder and dataset..."):
        encoder = load_pretrained_encoder(checkpoint_path, device)
        split = get_or_create_split(dataset_key, DATASET_REGISTRY[dataset_key]["subjects"])
        holdout_subjects = split["holdout_subjects"]
        epochs = load_dataset(dataset_key, subjects=holdout_subjects)

    st.caption(f"Evaluating on held-out subjects: {holdout_subjects} (never used in pretraining)")

    results = {}
    progress_bar = st.progress(0.0)
    for i, k in enumerate(sorted(k_candidates)):
        subj_means = []
        for subj in holdout_subjects:
            mask = epochs.subject_ids == subj
            X_subj, y_subj = epochs.X[mask], epochs.y[mask]
            if len(X_subj) == 0:
                continue
            r = kshot_fusion_eval(encoder, X_subj, y_subj, k=k, n_draws=n_draws, device=device)
            subj_means.append(r["mean_accuracy"])
        results[k] = sum(subj_means) / len(subj_means) if subj_means else 0.0
        progress_bar.progress((i + 1) / len(k_candidates))

    st.divider()
    st.subheader("Results")

    for k in sorted(results):
        acc = results[k]
        met = acc >= target_accuracy
        icon_color = "#4ade80" if met else "#f87171"
        st.markdown(
            f"<span style='color:{icon_color}; font-weight:600;'>{'Met' if met else 'Not met'}</span> "
            f"&nbsp;—&nbsp; k={k}: <b>{acc:.1%}</b> accuracy",
            unsafe_allow_html=True,
        )

    st.divider()
    meeting_k = [k for k in sorted(results) if results[k] >= target_accuracy]
    if meeting_k:
        min_k = min(meeting_k)
        n_classes = len(epochs.label_map)
        st.success(
            f"Minimum trials needed: **{min_k} per class** "
            f"({min_k * n_classes} total trials) to reach {target_accuracy:.0%} accuracy."
        )
    else:
        st.warning(
            f"None of the tested k values reached {target_accuracy:.0%} accuracy. "
            f"Try a higher k, a lower target, or a different encoder checkpoint."
        )