"""
Phase 3 pretraining entrypoint.
Usage: python scripts/pretrain.py
"""
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from src.data.loaders import load_dataset
from src.data.contrastive_dataset import PretrainContrastiveDataset
from src.models.encoder import SSLPretrainModel
from src.losses.nt_xent import NTXentLoss
from src.utils.device import get_device

from torch.utils.data import ConcatDataset
from src.data.channel_utils import align_channels

from src.data.channel_utils import align_epochs


CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def build_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())
    return LambdaLR(optimizer, lr_lambda)


def train(
    dataset_key: str = "bnci2014_001",
    subjects: list = None,
    batch_size: int = 64,
    epochs: int = 30,
    lr: float = 3e-4,
    temperature: float = 0.5,
    embed_dim: int = 128,
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    # --- Data ---
    raw_epochs = load_dataset(dataset_key, subjects=subjects)
    train_ds = PretrainContrastiveDataset(raw_epochs, seed=seed)
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True,  # NT-Xent needs consistent batch size for pos/neg indexing
        num_workers=0,   # start at 0 on macOS; MPS + multiprocessing can be flaky
    )

    n_channels = train_ds.X.shape[1]
    n_timepoints = train_ds.X.shape[2]

    # --- Model ---
    model = SSLPretrainModel(n_channels, n_timepoints, embed_dim=embed_dim).to(device)
    loss_fn = NTXentLoss(temperature=temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_steps = epochs * len(loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    history = []

    print(f"\nStarting pretraining: {epochs} epochs, {len(loader)} steps/epoch, "
          f"{total_steps} total steps\n")

    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

        for step, (v1, v2) in enumerate(loader):
            v1, v2 = v1.to(device), v2.to(device)

            optimizer.zero_grad()
            z1 = model(v1)
            z2 = model(v2)
            loss = loss_fn(z1, z2)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            if step % 5 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  epoch {epoch+1}/{epochs} step {step}/{len(loader)} "
                      f"loss={loss.item():.4f} lr={current_lr:.2e}")

        if device.type == "mps":
            torch.mps.empty_cache()

        avg_loss = running_loss / len(loader)
        elapsed = time.time() - epoch_start
        history.append(avg_loss)
        print(f"[epoch {epoch+1}/{epochs}] avg_loss={avg_loss:.4f} ({elapsed:.1f}s)\n")

    # Save encoder only (discard projection head)
    ckpt_path = CHECKPOINT_DIR / f"encoder_{dataset_key}.pt"
    torch.save({
        "encoder_state_dict": model.encoder.state_dict(),
        "n_channels": n_channels,
        "n_timepoints": n_timepoints,
        "embed_dim": embed_dim,
        "dataset_key": dataset_key,
        "loss_history": history,
    }, ckpt_path)
    print(f"Saved encoder checkpoint -> {ckpt_path}")

    return history


if __name__ == "__main__":
    train()

def train_multi(
    dataset_keys=("bnci2014_001", "physionet_mi"),
    subjects_per_dataset=None,   # dict: dataset_key -> subject list, or None for registry default
    batch_size=64,
    epochs=8,
    lr=3e-4,
    temperature=0.5,
    embed_dim=128,
    seed=42,
):
    torch.manual_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    subjects_per_dataset = subjects_per_dataset or {}
    raw_epochs_list = [
        load_dataset(dk, subjects=subjects_per_dataset.get(dk), tmin=0.0, tmax=4.0, resample_sfreq=250.0)
        for dk in dataset_keys
    ]
    aligned = align_epochs(raw_epochs_list)

    per_dataset_ds = [PretrainContrastiveDataset(e, seed=seed) for e in aligned]
    combined_ds = ConcatDataset(per_dataset_ds)
    print(f"\nCombined pretraining set: {len(combined_ds)} trials across {len(dataset_keys)} datasets")

    loader = DataLoader(combined_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    n_channels = aligned[0].X.shape[1]
    n_timepoints = aligned[0].X.shape[2]
    model = SSLPretrainModel(n_channels, n_timepoints, embed_dim=embed_dim).to(device)
    loss_fn = NTXentLoss(temperature=temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_steps = epochs * len(loader)
    scheduler = build_warmup_cosine_scheduler(optimizer, int(0.1 * total_steps), total_steps)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    history = []
    print(f"\nStarting multi-dataset pretraining: {epochs} epochs, {len(loader)} steps/epoch\n")

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for step, (v1, v2) in enumerate(loader):
            v1, v2 = v1.to(device), v2.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(v1), model(v2))
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        if device.type == "mps":
            torch.mps.empty_cache()
        avg_loss = running_loss / len(loader)
        history.append(avg_loss)
        print(f"[epoch {epoch+1}/{epochs}] avg_loss={avg_loss:.4f} ({time.time()-t0:.1f}s)")

    ckpt_path = CHECKPOINT_DIR / "encoder_multi_bnci_physionet.pt"
    torch.save({
        "encoder_state_dict": model.encoder.state_dict(),
        "n_channels": n_channels, "n_timepoints": n_timepoints,
        "embed_dim": embed_dim, "dataset_keys": list(dataset_keys),
        "loss_history": history,
    }, ckpt_path)
    print(f"Saved -> {ckpt_path}")
    return history
