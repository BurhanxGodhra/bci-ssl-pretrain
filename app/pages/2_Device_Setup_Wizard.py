"""
Consumer-facing setup wizard: connect device -> verify signal -> calibrate -> live test.
"""
import random
import time

import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data.loaders import load_dataset
from src.data.splits import load_split
from src.finetune.fusion_eval import compute_tangent_features
from src.finetune.linear_probe_eval import embed_all, load_pretrained_encoder
from src.streaming.lsl_utils import (
    LSL_AVAILABLE,
    check_signal_quality,
    connect_to_stream,
    find_real_eeg_streams,
    start_replay_outlet,
)
from src.utils.device import get_device

st.set_page_config(page_title="Device Setup Wizard", layout="centered")

CLASSES = ["left_hand", "right_hand", "feet", "tongue"]
CLASS_LABELS = {
    "left_hand": "LEFT HAND",
    "right_hand": "RIGHT HAND",
    "feet": "FEET",
    "tongue": "TONGUE",
}
CLASS_SYMBOLS = {
    "left_hand": "\u2190",   # ←
    "right_hand": "\u2192",  # →
    "feet": "\u2191",        # ↑
    "tongue": "\u25CF",      # ●
}
EXPECTED_N_CHANNELS = 22
EXPECTED_SFREQ = 250.0
IMAGERY_DURATION = 4.0
ENCODER_CHECKPOINT = "checkpoints/encoder_bnci2014_001_only_e25_flat_gentleaug.pt"


def render_countdown(placeholder, seconds: int):
    for remaining in range(seconds, 0, -1):
        placeholder.markdown(
            f"""
            <div style="text-align:center; padding:32px; background-color:#111827;
                        border:1px solid #374151; border-radius:8px;">
                <div style="font-size:14px; letter-spacing:2px; color:#9ca3af;
                            text-transform:uppercase; margin-bottom:8px;">Prepare</div>
                <div style="font-size:48px; font-weight:600; color:#e5e7eb;">{remaining}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(1)


def render_cue(placeholder, cls: str):
    placeholder.markdown(
        f"""
        <div style="text-align:center; padding:40px; background-color:#0f2418;
                    border:2px solid #16a34a; border-radius:8px;">
            <div style="font-size:14px; letter-spacing:2px; color:#4ade80;
                        text-transform:uppercase; margin-bottom:12px;">Imagine Movement</div>
            <div style="font-size:56px; color:#4ade80; line-height:1;">{CLASS_SYMBOLS[cls]}</div>
            <div style="font-size:28px; font-weight:600; color:#e5e7eb; margin-top:12px;
                        letter-spacing:1px;">{CLASS_LABELS[cls]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.title("BCI Device Setup Wizard")
st.markdown("Connect an EEG device, verify signal quality, and calibrate a working classifier.")

if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = "connect"
if "inlet" not in st.session_state:
    st.session_state.inlet = None

st.progress(
    {"connect": 0.25, "signal_check": 0.50, "calibrate": 0.75, "live_test": 1.0}[st.session_state.wizard_step]
)

# ---------------- Step 1: Connect ----------------
if st.session_state.wizard_step == "connect":
    st.header("Step 1 — Connect Device")

    if not LSL_AVAILABLE:
        st.warning("LSL library not available on this machine. Only 'Try it now' mode will function.")

    st.session_state.k_shot = st.select_slider(
        "Trials per class for calibration", options=[1, 5, 10, 20], value=5
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Connect Real Device")
        st.caption("Scans your network for an active EEG headset streaming via LSL.")
        if st.button("Scan for Devices", disabled=not LSL_AVAILABLE):
            with st.spinner("Scanning for LSL EEG streams..."):
                streams = find_real_eeg_streams(timeout=2.0)
            if streams:
                st.success(f"Found {len(streams)} EEG stream(s).")
                inlet, sfreq, n_channels, buf = connect_to_stream(streams[0])
                st.session_state.inlet = inlet
                st.session_state.sfreq = sfreq
                st.session_state.n_channels = n_channels
                st.session_state.wizard_step = "signal_check"
                st.rerun()
            else:
                st.error("No EEG devices found. Confirm your headset's LSL streaming app is running, or use 'Try it now'.")

    with col2:
        st.subheader("Try It Now (No Hardware)")
        st.caption(
            "Streams pre-recorded EEG data over a real LSL connection, exercising the "
            "same code path a physical device would use."
        )
        if st.button("Start Simulated Device", disabled=not LSL_AVAILABLE):
            with st.spinner("Starting simulated LSL stream..."):
                split = load_split("bnci2014_001")
                subj = split["holdout_subjects"][0]
                epochs = load_dataset("bnci2014_001", subjects=[subj])
                st.session_state.replay_epochs = epochs

                start_replay_outlet(epochs.X[:20], epochs.channel_names, epochs.sfreq)
                time.sleep(1.0)
                streams = find_real_eeg_streams(timeout=2.0)

            if streams:
                inlet, sfreq, n_channels, buf = connect_to_stream(streams[0])
                st.session_state.inlet = inlet
                st.session_state.sfreq = sfreq
                st.session_state.n_channels = n_channels
                st.session_state.wizard_step = "signal_check"
                st.rerun()
            else:
                st.error("Simulated stream failed to start. Confirm pylsl / liblsl installed correctly.")

# ---------------- Step 2: Signal Quality ----------------
elif st.session_state.wizard_step == "signal_check":
    st.header("Step 2 — Signal Quality Check")
    st.caption("Verifying electrode contact before calibration begins.")

    inlet = st.session_state.inlet
    with st.spinner("Collecting a short sample..."):
        chunk, _ = inlet.pull_chunk(timeout=3.0, max_samples=int(2 * st.session_state.sfreq))
        chunk = np.array(chunk)

    if len(chunk) == 0:
        st.error("No data received from stream. Try reconnecting.")
    else:
        quality = check_signal_quality(chunk)
        if quality["ok"]:
            st.success("Signal quality is acceptable on all channels.")
            if st.button("Continue to Calibration", type="primary"):
                st.session_state.wizard_step = "calibrate"
                st.rerun()
        else:
            if quality["flat_channels"]:
                st.error(f"Flat signal detected on channel(s): {quality['flat_channels']}. Check electrode contact.")
            if quality["noisy_channels"]:
                st.warning(f"Excessive noise on channel(s): {quality['noisy_channels']}. Check for loose connections.")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Re-check Signal"):
                    st.rerun()
            with col_b:
                if st.button("Proceed Anyway"):
                    st.session_state.wizard_step = "calibrate"
                    st.rerun()

# ---------------- Step 3: Calibration ----------------
elif st.session_state.wizard_step == "calibrate":
    st.header("Step 3 — Calibration")

    if st.session_state.n_channels != EXPECTED_N_CHANNELS:
        st.error(
            f"This device streams {st.session_state.n_channels} channels, but the trained "
            f"model expects {EXPECTED_N_CHANNELS} channels in a specific motor-cortex montage. "
            f"This device is not yet compatible with this pretrained model."
        )
        st.stop()

    if "calibration_plan" not in st.session_state:
        k = st.session_state.get("k_shot", 5)
        plan = [(cls, i) for cls in CLASSES for i in range(k)]
        random.seed(42)
        random.shuffle(plan)
        st.session_state.calibration_plan = plan
        st.session_state.calibration_idx = 0
        st.session_state.collected_X = []
        st.session_state.collected_y = []

    plan = st.session_state.calibration_plan
    idx = st.session_state.calibration_idx

    if idx < len(plan):
        cls, _ = plan[idx]
        st.progress(idx / len(plan))
        st.subheader(f"Trial {idx + 1} of {len(plan)}")

        placeholder = st.empty()
        if st.button("Start This Trial", type="primary", key=f"trial_{idx}"):
            render_countdown(placeholder, 2)
            render_cue(placeholder, cls)

            n_samples = int(IMAGERY_DURATION * st.session_state.sfreq)
            chunk, _ = st.session_state.inlet.pull_chunk(
                timeout=IMAGERY_DURATION + 1, max_samples=n_samples
            )
            chunk = np.array(chunk)

            if len(chunk) < n_samples * 0.9:
                placeholder.warning("Insufficient samples received for this trial. Retrying.")
            else:
                trial = chunk[:n_samples].T
                st.session_state.collected_X.append(trial)
                st.session_state.collected_y.append(CLASSES.index(cls))
                st.session_state.calibration_idx += 1
                placeholder.empty()
                st.rerun()
    else:
        st.success(f"Collected {len(st.session_state.collected_X)} calibration trials.")

        if "trained_clf" not in st.session_state:
            if st.button("Train Classifier", type="primary"):
                with st.spinner("Fitting classifier on calibration data..."):
                    device = get_device()
                    encoder = load_pretrained_encoder(ENCODER_CHECKPOINT, device)
                    X_sup = np.stack(st.session_state.collected_X)
                    y_sup = np.array(st.session_state.collected_y)

                    emb_sup = embed_all(encoder, X_sup, device).numpy()
                    ts_sup, _ = compute_tangent_features(X_sup, X_sup[:1])

                    combined = np.concatenate([emb_sup, ts_sup], axis=1)
                    scaler = StandardScaler()
                    combined_scaled = scaler.fit_transform(combined)

                    clf = LogisticRegression(max_iter=1000, C=0.1)
                    clf.fit(combined_scaled, y_sup)

                    st.session_state.trained_clf = clf
                    st.session_state.trained_scaler = scaler
                    st.session_state.trained_encoder = encoder

                st.success("Calibration complete. Classifier is ready.")
                st.rerun()
        else:
            st.success("Classifier trained and ready.")
            if st.button("Continue to Live Test", type="primary"):
                st.session_state.wizard_step = "live_test"
                st.rerun()

# ---------------- Step 4: Live Test ----------------
elif st.session_state.wizard_step == "live_test":
    st.header("Step 4 — Live Test")
    st.caption(
        "A random class will be cued, and the calibrated classifier's prediction will be "
        "shown alongside it. In simulated mode, predictions are not expected to be "
        "meaningfully accurate, since the replayed signal has no relationship to the cue."
    )

    if "live_test_history" not in st.session_state:
        st.session_state.live_test_history = []

    col1, col2 = st.columns(2)
    with col1:
        cue_placeholder = st.empty()
        if st.button("Run Live Test Trial", type="primary"):
            true_cls = random.choice(CLASSES)
            render_countdown(cue_placeholder, 2)
            render_cue(cue_placeholder, true_cls)

            n_samples = int(IMAGERY_DURATION * st.session_state.sfreq)
            chunk, _ = st.session_state.inlet.pull_chunk(
                timeout=IMAGERY_DURATION + 1, max_samples=n_samples
            )
            chunk = np.array(chunk)
            cue_placeholder.empty()

            if len(chunk) >= n_samples * 0.9:
                trial = chunk[:n_samples].T[np.newaxis, :, :]

                emb = embed_all(st.session_state.trained_encoder, trial, get_device()).numpy()
                _, ts_pred = compute_tangent_features(
                    np.stack(st.session_state.collected_X), trial
                )
                combined = np.concatenate([emb, ts_pred], axis=1)
                combined_scaled = st.session_state.trained_scaler.transform(combined)

                pred_idx = st.session_state.trained_clf.predict(combined_scaled)[0]
                pred_cls = CLASSES[pred_idx]

                st.session_state.live_test_history.append({
                    "true": true_cls,
                    "predicted": pred_cls,
                    "correct": pred_cls == true_cls,
                })
                st.rerun()

    with col2:
        if st.button("Restart Wizard"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    if st.session_state.live_test_history:
        history = st.session_state.live_test_history
        n_correct = sum(h["correct"] for h in history)
        st.metric("Live Accuracy", f"{n_correct}/{len(history)} ({n_correct / len(history):.0%})")

        st.divider()
        for h in reversed(history):
            status = "Correct" if h["correct"] else "Incorrect"
            color = "#4ade80" if h["correct"] else "#f87171"
            st.markdown(
                f"<span style='color:{color}; font-weight:600;'>{status}</span> "
                f"&nbsp;—&nbsp; Cued: <b>{CLASS_LABELS[h['true']]}</b> "
                f"&nbsp;→&nbsp; Predicted: <b>{CLASS_LABELS[h['predicted']]}</b>",
                unsafe_allow_html=True,
            )