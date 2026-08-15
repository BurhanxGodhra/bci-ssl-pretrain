"""
Subject-level split management. This is the leakage firewall between
pretraining and few-shot evaluation (Phase 5) -- holdout subjects must
NEVER appear in pretraining, augmentation-cache generation, or model
selection.
"""
import json
from pathlib import Path
from typing import Dict, List

SPLITS_DIR = Path(__file__).resolve().parents[2] / "data" / "splits"


def create_split(
    dataset_name: str,
    all_subjects: List[int],
    n_holdout: int = 2,
    seed: int = 42,
) -> Dict:
    """
    Deterministically split subjects into pretrain-pool vs. held-out-for-fewshot.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    subjects = sorted(all_subjects)
    shuffled = rng.permutation(subjects).tolist()

    holdout = sorted(shuffled[:n_holdout])
    pretrain_pool = sorted(shuffled[n_holdout:])

    split = {
        "dataset_name": dataset_name,
        "seed": seed,
        "pretrain_subjects": pretrain_pool,
        "holdout_subjects": holdout,
    }

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SPLITS_DIR / f"{dataset_name}_split.json"
    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)

    print(f"Saved split -> {out_path}")
    print(f"  Pretrain pool ({len(pretrain_pool)}): {pretrain_pool}")
    print(f"  Holdout ({len(holdout)}):            {holdout}")
    return split


def load_split(dataset_name: str) -> Dict:
    path = SPLITS_DIR / f"{dataset_name}_split.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No split found for '{dataset_name}' at {path}. "
            f"Run create_split() first."
        )
    with open(path) as f:
        return json.load(f)

def get_or_create_split(dataset_name, all_subjects, n_holdout: int = 2, seed: int = 42):
    try:
        return load_split(dataset_name)
    except FileNotFoundError:
        return create_split(dataset_name, all_subjects, n_holdout=n_holdout, seed=seed)
