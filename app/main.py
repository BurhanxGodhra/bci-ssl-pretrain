"""
Entry point for the BCI SSL Pretraining project's Streamlit app.
"""
import streamlit as st

st.set_page_config(page_title="BCI SSL Pretraining", layout="centered")

st.title("Few-Shot Motor Imagery BCI")
st.markdown(
    "Self-supervised pretraining for cross-subject EEG generalization, "
    "with few-shot calibration for new users."
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Benchmark Demo")
    st.caption(
        "Validated results on held-out research-dataset subjects: accuracy "
        "vs. calibration trials, SSL vs. classical baseline."
    )
    st.page_link("pages/1_Benchmark_Demo.py", label="Open Benchmark Demo")

with col2:
    st.subheader("Device Setup Wizard")
    st.caption(
        "Connect a real EEG headset, or try a simulated one, and calibrate "
        "a working classifier."
    )
    st.page_link("pages/2_Device_Setup_Wizard.py", label="Open Setup Wizard")

st.divider()
st.caption(
    "Built on a multi-dataset self-supervised pretrained encoder "
    "(BNCI2014_001 + PhysionetMI), fused with Riemannian tangent-space "
    "features for few-shot classification."
)