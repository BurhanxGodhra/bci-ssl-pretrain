"""
MOABB dataset loaders for motor imagery EEG.
Wraps multiple MOABB datasets into a consistent epoched numpy format:
  X: (n_trials, n_channels, n_timepoints)
  y: (n_trials,)  -- integer class labels
  subject_ids: (n_trials,) -- which subject each trial belongs to
  dataset_name: str -- source dataset identifier (for cross-dataset tracking)
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import mne

from moabb.datasets import BNCI2014_001, PhysionetMI
from moabb.paradigms import MotorImagery


@dataclass
class EEGEpochs:
    X: np.ndarray            # (n_trials, n_channels, n_timepoints)
    y: np.ndarray            # (n_trials,) int labels
    subject_ids: np.ndarray  # (n_trials,) int subject id
    dataset_name: str
    channel_names: List[str]
    sfreq: float
    label_map: dict          # {int_label: class_name}


# Registry of supported MOABB datasets for this pipeline.
# Each entry defines dataset class + which subjects to pull.
DATASET_REGISTRY = {
    "bnci2014_001": {
        "class": BNCI2014_001,
        "subjects": list(range(1, 10)),   # 9 subjects
    },
    "physionet_mi": {
        "class": PhysionetMI,
        "subjects": list(range(1, 11)),   # start small: 10 subjects (109 available)
    },
}


def load_dataset(
    dataset_key: str,
    subjects: Optional[List[int]] = None,
    fmin: float = 4.0,
    fmax: float = 38.0,
    tmin: float = 0.0,
    tmax: float = 4.0,
    resample_sfreq: float = 250.0,
) -> EEGEpochs:
    """
    Load and epoch a single MOABB dataset into a consistent format.

    Args:
        dataset_key: key into DATASET_REGISTRY
        subjects: subset of subjects to load; defaults to registry list
        fmin/fmax: bandpass filter range (Hz) -- 4-38Hz captures mu (8-13Hz)
                   and beta (13-30Hz) rhythms central to motor imagery
        tmin/tmax: epoch window relative to cue onset (seconds)
        resample_sfreq: target sampling rate -- unifies datasets with
                         different native sample rates (e.g. 250Hz vs 160Hz)

    Returns:
        EEGEpochs with unified array format
    """
    if dataset_key not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_key}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    entry = DATASET_REGISTRY[dataset_key]
    dataset = entry["class"]()
    subjects = subjects or entry["subjects"]

    paradigm = MotorImagery(
        fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax,
        resample=resample_sfreq,
    )

    print(f"[{dataset_key}] Loading subjects {subjects} ...")
    X, labels, metadata = paradigm.get_data(
        dataset=dataset, subjects=subjects
    )
    # X: (n_trials, n_channels, n_timepoints) already unified by MOABB paradigm

    unique_labels = sorted(set(labels))
    label_to_int = {lab: i for i, lab in enumerate(unique_labels)}
    y = np.array([label_to_int[lab] for lab in labels], dtype=np.int64)

    subject_ids = metadata["subject"].to_numpy()

    # Channel names: pull from one representative raw recording
    info = dataset.get_data(subjects=[subjects[0]])[subjects[0]]
    session_key = list(info.keys())[0]
    run_key = list(info[session_key].keys())[0]
    raw = info[session_key][run_key]
    channel_names = raw.info["ch_names"]

    print(
        f"[{dataset_key}] Done. X={X.shape}, "
        f"classes={unique_labels}, subjects={sorted(set(subject_ids))}"
    )

    return EEGEpochs(
        X=X.astype(np.float32),
        y=y,
        subject_ids=subject_ids,
        dataset_name=dataset_key,
        channel_names=channel_names,
        sfreq=resample_sfreq,
        label_map={v: k for k, v in label_to_int.items()},
    )


def load_multiple_datasets(dataset_keys: List[str], **kwargs) -> List[EEGEpochs]:
    """Load several MOABB datasets for multi-dataset pretraining."""
    return [load_dataset(k, **kwargs) for k in dataset_keys]
