"""
Interactive demo: simulates the few-shot BCI calibration flow using
held-out subject data and our validated pretrained encoder + linear probe.
"""
import streamlit as st
import numpy as np
import torch

from src.data.loaders import load_dataset
from src.data.splits import load_split
from src.finetune.linear_probe_eval import (
    load_pretrained_encoder, embed_all, train_linear_probe, evaluate_probe,
)
from src.finetune.fewshot_sampler import sample_k_shot_split
from src.utils.device import get_device
from src.analysis.calibration_time import k_to_time, N_CLASSES

st.set_page_config(page_title="Few-Shot BCI Calibration Demo", layout="centered")

st.title("Few-Shot Motor Imagery BCI Calibration")
st.markdown(
    "This demo simulates a **new user** calibrating a motor-imagery BCI "
    "using only a handful of labeled trials, powered by a self-supervised "
    "pretrained encoder. Data is from a real held-out test subject "
    "(never seen during pretraining) — this is not synthetic."
)

device = get_device()

@st.cache_resource
def load_resources():
    encoder = load_pretrained_encoder("checkpoints/encoder_multi_full_e25.pt", device)
    split = load_split("bnci2014_001")
    holdout_subjects = split["holdout_subjects"]
    epochs = load_dataset("bnci2014_001", subjects=holdout_subjects)
    return encoder, epochs, holdout_subjects

with st.spinner("Loading pretrained encoder and holdout subject data..."):
    encoder, epochs_data, holdout_subjects = load_resources()

st.divider()

subj = st.selectbox("Simulated new user (held-out test subject)", holdout_subjects)
k = st.select_slider("Labeled trials per class (k)", options=[1, 5, 10, 20], value=5)

mask = epochs_data.subject_ids == subj
X_subj, y_subj = epochs_data.X[mask], epochs_data.y[mask]
n_classes = len(np.unique(y_subj))
class_names = list(epochs_data.label_map.values())

time_info = k_to_time(k, n_classes=n_classes)
st.info(
    f"**Calibration cost:** {time_info['total_trials']} trials "
    f"({k} per class × {n_classes} classes) ≈ **{time_info['total_minutes']:.1f} minutes** "
    f"of imagery time, vs. ~77 min for full classical calibration."
)

if st.button("Calibrate & Evaluate", type="primary"):
    with st.spinner(f"Sampling {k} trials/class, embedding, training linear probe..."):
        X_sup, y_sup, X_qry, y_qry = sample_k_shot_split(X_subj, y_subj, k=k, seed=42)

        emb_sup = embed_all(encoder, X_sup, device)
        emb_qry = embed_all(encoder, X_qry, device)

        probe = train_linear_probe(emb_sup, y_sup, n_classes=n_classes, device=device)
        accuracy = evaluate_probe(probe, emb_qry, y_qry, device)

    st.success(f"✅ Calibration complete — accuracy on {len(y_qry)} unseen trials: **{accuracy:.1%}**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Labeled trials used", time_info["total_trials"])
    col2.metric("Calibration time", f"{time_info['total_minutes']:.1f} min")
    col3.metric("Accuracy achieved", f"{accuracy:.1%}")

    st.caption(
        f"Classes: {', '.join(class_names)}. Encoder: multi-dataset SSL pretrained "
        f"(BNCI2014_001 + PhysionetMI), evaluated on subject {subj} who was fully "
        f"excluded from pretraining."
    )

st.divider()
st.caption(
    "⚠️ This demo uses real EEG data from a held-out test subject, not live hardware. "
    "It demonstrates the calibration mechanism, not a production-ready BCI system."
)