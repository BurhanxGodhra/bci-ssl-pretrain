"""
Consumer-facing setup wizard: connect device -> verify signal -> calibrate.
"""
import streamlit as st
import numpy as np
import time

from src.streaming.lsl_utils import (
    LSL_AVAILABLE, find_real_eeg_streams, start_replay_outlet,
    connect_to_stream, check_signal_quality,
)
from src.data.loaders import load_dataset
from src.data.splits import load_split

st.set_page_config(page_title="Device Setup Wizard", layout="centered")
st.title("⚙️ BCI Device Setup Wizard")
st.markdown("Get your motor-imagery BCI calibrated and ready to use in a few minutes.")

if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = "connect"
if "inlet" not in st.session_state:
    st.session_state.inlet = None

st.progress(
    {"connect": 0.33, "signal_check": 0.66, "calibrate": 1.0}[st.session_state.wizard_step]
)

# ---------------- Step 1: Connect ----------------
if st.session_state.wizard_step == "connect":
    st.header("Step 1: Connect Your Device")

    if not LSL_AVAILABLE:
        st.warning("LSL library not available on this machine -- only 'Try it now' mode will work.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(" Connect Real Device")
        st.caption("Scans your network for an active EEG headset streaming via LSL.")
        if st.button("Scan for devices", disabled=not LSL_AVAILABLE):
            with st.spinner("Scanning for LSL EEG streams (2s)..."):
                streams = find_real_eeg_streams(timeout=2.0)
            if streams:
                st.success(f"Found {len(streams)} EEG stream(s)!")
                inlet, sfreq, n_channels, buf = connect_to_stream(streams[0])
                st.session_state.inlet = inlet
                st.session_state.sfreq = sfreq
                st.session_state.n_channels = n_channels
                st.session_state.wizard_step = "signal_check"
                st.rerun()
            else:
                st.error("No EEG devices found. Make sure your headset's LSL streaming app is running, or use 'Try it now' instead.")

    with col2:
        st.subheader(" Try It Now (No Hardware)")
        st.caption(
            "Simulates a real headset by streaming pre-recorded EEG data "
            "over an actual LSL connection -- exercises the same code path "
            "a real device would use."
        )
        if st.button("Start simulated device", disabled=not LSL_AVAILABLE):
            with st.spinner("Starting simulated LSL stream..."):
                split = load_split("bnci2014_001")
                subj = split["holdout_subjects"][0]
                epochs = load_dataset("bnci2014_001", subjects=[subj])
                st.session_state.replay_epochs = epochs

                outlet = start_replay_outlet(
                    epochs.X[:20], epochs.channel_names, epochs.sfreq
                )
                time.sleep(1.0)  # let the outlet register before resolving
                streams = find_real_eeg_streams(timeout=2.0)

            if streams:
                inlet, sfreq, n_channels, buf = connect_to_stream(streams[0])
                st.session_state.inlet = inlet
                st.session_state.sfreq = sfreq
                st.session_state.n_channels = n_channels
                st.session_state.wizard_step = "signal_check"
                st.rerun()
            else:
                st.error("Simulated stream failed to start. Check that pylsl/liblsl installed correctly.")

# ---------------- Step 2: Signal Quality ----------------
elif st.session_state.wizard_step == "signal_check":
    st.header("Step 2: Signal Quality Check")
    st.caption("Checking electrode contact before calibration begins.")

    inlet = st.session_state.inlet
    with st.spinner("Collecting a short sample..."):
        chunk, _ = inlet.pull_chunk(timeout=3.0, max_samples=int(2 * st.session_state.sfreq))
        chunk = np.array(chunk)

    if len(chunk) == 0:
        st.error("No data received from stream. Try reconnecting.")
    else:
        quality = check_signal_quality(chunk)
        if quality["ok"]:
            st.success(" Signal quality looks good on all channels.")
            if st.button("Continue to Calibration →", type="primary"):
                st.session_state.wizard_step = "calibrate"
                st.rerun()
        else:
            if quality["flat_channels"]:
                st.error(f"⚠️ Flat signal detected on channel(s): {quality['flat_channels']}. Check electrode contact.")
            if quality["noisy_channels"]:
                st.warning(f"⚠️ Excessive noise on channel(s): {quality['noisy_channels']}. Check for loose connections.")
            if st.button("Re-check signal"):
                st.rerun()
            if st.button("Proceed anyway"):
                st.session_state.wizard_step = "calibrate"
                st.rerun()

# ---------------- Step 3: Calibration (placeholder for next step) ----------------
elif st.session_state.wizard_step == "calibrate":
    st.header("Step 3: Calibration")
    st.info(" Guided cue-by-cue calibration flow — building next.")