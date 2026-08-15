"""
Channel alignment for combining datasets with different EEG montages
into a shared input space for one encoder.
"""
from typing import List
from src.data.loaders import EEGEpochs


def _normalize_name(ch: str) -> str:
    return ch.strip().upper().replace(".", "")


def find_common_channels(channel_name_lists: List[List[str]]) -> List[str]:
    normalized_sets = [{_normalize_name(c) for c in names} for names in channel_name_lists]
    common = set.intersection(*normalized_sets)
    if len(common) == 0:
        raise ValueError("No common channels found -- check montage naming across datasets.")
    return sorted(common)


def align_channels(epochs_list: List[EEGEpochs]) -> List[EEGEpochs]:
    """Subset every EEGEpochs down to the shared channel set, same fixed order."""
    common = find_common_channels([e.channel_names for e in epochs_list])
    print(f"[align_channels] Common channels ({len(common)}): {common}")

    aligned = []
    for e in epochs_list:
        norm_to_orig = {_normalize_name(c): i for i, c in enumerate(e.channel_names)}
        idx = [norm_to_orig[c] for c in common]
        X_sub = e.X[:, idx, :]
        aligned.append(EEGEpochs(
            X=X_sub, y=e.y, subject_ids=e.subject_ids,
            dataset_name=e.dataset_name, channel_names=common,
            sfreq=e.sfreq, label_map=e.label_map,
        ))
        print(f"  {e.dataset_name}: {e.X.shape[1]} -> {X_sub.shape[1]} channels")
    return aligned

def align_epochs(epochs_list: List[EEGEpochs]) -> List[EEGEpochs]:
    """
    Aligns a list of EEGEpochs to (a) a shared channel set and
    (b) a shared timepoint length -- both required before they can be
    combined into one contrastive pretraining pool for a single encoder.
    """
    # --- channel alignment (existing logic) ---
    common = find_common_channels([e.channel_names for e in epochs_list])
    print(f"[align_epochs] Common channels ({len(common)}): {common}")

    # --- timepoint alignment (new) ---
    min_timepoints = min(e.X.shape[2] for e in epochs_list)
    lengths = {e.dataset_name: e.X.shape[2] for e in epochs_list}
    print(f"[align_epochs] Timepoint lengths per dataset: {lengths} -> cropping all to {min_timepoints}")

    aligned = []
    for e in epochs_list:
        norm_to_orig = {_normalize_name(c): i for i, c in enumerate(e.channel_names)}
        idx = [norm_to_orig[c] for c in common]
        X_sub = e.X[:, idx, :min_timepoints]  # channel subset + time crop together
        aligned.append(EEGEpochs(
            X=X_sub, y=e.y, subject_ids=e.subject_ids,
            dataset_name=e.dataset_name, channel_names=common,
            sfreq=e.sfreq, label_map=e.label_map,
        ))
        print(f"  {e.dataset_name}: {e.X.shape[1]}ch/{e.X.shape[2]}t -> {X_sub.shape[1]}ch/{X_sub.shape[2]}t")
    return aligned
