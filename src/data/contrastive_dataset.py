"""
Contrastive pretraining dataset. Filters to pretrain-pool subjects only
(via committed split manifest) and returns two augmented views per trial.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.loaders import EEGEpochs
from src.data.splits import load_split
from src.augmentations.eeg_augment import build_default_pipeline


class PretrainContrastiveDataset(Dataset):
    def __init__(self, epochs: EEGEpochs, seed: int = 42):
        """
        epochs: full EEGEpochs object (may include ALL subjects loaded from disk)
        seed:   base seed; each trial gets a derived, distinct augmentation stream
        """
        split = load_split(epochs.dataset_name)
        pretrain_subjects = set(split["pretrain_subjects"])

        keep_mask = np.isin(epochs.subject_ids, list(pretrain_subjects))
        excluded = int((~keep_mask).sum())
        if excluded > 0:
            print(
                f"[PretrainContrastiveDataset] Filtered out {excluded} trials "
                f"from holdout subjects {sorted(set(epochs.subject_ids[~keep_mask]))}"
            )

        self.X = epochs.X[keep_mask]
        self.subject_ids = epochs.subject_ids[keep_mask]
        self.sfreq = epochs.sfreq
        self.seed = seed

        if len(self.X) == 0:
            raise ValueError(
                f"No pretraining trials remain after filtering to subjects "
                f"{pretrain_subjects}. Check split manifest vs. loaded subjects."
            )

        print(
            f"[PretrainContrastiveDataset] Ready: {len(self.X)} trials, "
            f"subjects={sorted(pretrain_subjects)}"
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        trial = self.X[idx]  # (n_channels, n_timepoints)

        # Distinct RNG stream per (trial, view) so view1 != view2 and
        # different epoch passes over the same trial still vary.
        seed1 = self.seed + idx * 2
        seed2 = self.seed + idx * 2 + 1

        aug1 = build_default_pipeline(sfreq=self.sfreq, seed=seed1)
        aug2 = build_default_pipeline(sfreq=self.sfreq, seed=seed2)

        view1 = aug1(trial)
        view2 = aug2(trial)

        return torch.from_numpy(view1.copy()), torch.from_numpy(view2.copy())
