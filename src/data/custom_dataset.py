"""
Validates and adapts user-uploaded EEG data to the fixed montage our
pretrained encoder requires. Exact channel identity (not just count)
must match, since the encoder's spatial weights are tied to specific
electrode positions learned during pretraining.
"""
import numpy as np

from src.data.loaders import EEGEpochs
from src.data.channel_utils import _normalize_name, normalize_trials

# Must match the montage used during pretraining (src/data/channel_utils.py
# find_common_channels output when BNCI2014_001 + PhysionetMI were aligned)
REQUIRED_CHANNELS = [
    "C1", "C2", "C3", "C4", "C5", "C6",
    "CP1", "CP2", "CP3", "CP4", "CPZ", "CZ",
    "FC1", "FC2", "FC3", "FC4", "FCZ", "FZ",
    "P1", "P2", "POZ", "PZ",
]


def validate_and_adapt_upload(
    X: np.ndarray,
    y: np.ndarray,
    channel_names: list,
    sfreq: float,
    class_names: list,
) -> dict:
    """
    Returns {"ok": True, "epochs": EEGEpochs} on success, or
    {"ok": False, "error": str, ...} on failure.
    """
    if X.ndim != 3:
        return {"ok": False, "error": f"X must be 3D (n_trials, n_channels, n_timepoints), got shape {X.shape}"}
    if X.shape[0] != len(y):
        return {"ok": False, "error": f"X has {X.shape[0]} trials but y has {len(y)} labels."}
    if X.shape[1] != len(channel_names):
        return {"ok": False, "error": f"X has {X.shape[1]} channels but {len(channel_names)} channel names provided."}
    if X.shape[2] < 64:
        return {"ok": False, "error": f"Trials have only {X.shape[2]} timepoints -- too short for the encoder's temporal convolution (needs at least 64)."}

    norm_to_idx = {_normalize_name(c): i for i, c in enumerate(channel_names)}
    missing = [c for c in REQUIRED_CHANNELS if c not in norm_to_idx]

    if missing:
        return {
            "ok": False,
            "error": (
                f"Missing {len(missing)} of {len(REQUIRED_CHANNELS)} required channels. "
                f"This dataset's montage is not compatible with the pretrained encoder."
            ),
            "missing_channels": missing,
            "found_channels": [c for c in REQUIRED_CHANNELS if c in norm_to_idx],
        }

    reorder_idx = [norm_to_idx[c] for c in REQUIRED_CHANNELS]
    X_reordered = X[:, reorder_idx, :]
    X_normalized = normalize_trials(X_reordered)

    label_map = {i: name for i, name in enumerate(class_names)}
    n_trials = X.shape[0]

    epochs = EEGEpochs(
        X=X_normalized.astype(np.float32),
        y=y.astype(np.int64),
        subject_ids=np.zeros(n_trials, dtype=np.int64),
        dataset_name="custom_upload",
        channel_names=REQUIRED_CHANNELS,
        sfreq=sfreq,
        label_map=label_map,
    )
    return {"ok": True, "epochs": epochs}