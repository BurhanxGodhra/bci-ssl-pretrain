"""
Sanity check for Apple Silicon MPS backend + MOABB environment paths.
Run standalone: python tests/test_mps_sanity.py
"""
import sys
import platform
import torch
import torch.nn as nn


def check_mps_availability():
    print("=" * 60)
    print("SYSTEM INFO")
    print("=" * 60)
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")

    print("\n" + "=" * 60)
    print("MPS BACKEND CHECK")
    print("=" * 60)
    built = torch.backends.mps.is_built()
    available = torch.backends.mps.is_available()
    print(f"MPS built into this torch install: {built}")
    print(f"MPS available on this hardware:    {available}")

    if not (built and available):
        print("\n[WARNING] MPS not available — falling back to CPU.")
        return torch.device("cpu")

    return torch.device("mps")


def run_conv_forward_backward(device: torch.device):
    """
    Test the actual op family EEGNet depends on: Conv2d (spatial + temporal
    convolutions), BatchNorm, and backward pass — not just tensor creation.
    """
    print("\n" + "=" * 60)
    print(f"CONV FORWARD/BACKWARD TEST on '{device}'")
    print("=" * 60)

    torch.manual_seed(42)

    # Simulate EEG-like input: (batch, channels=1, EEG_channels=22, time=750)
    x = torch.randn(8, 1, 22, 750, device=device)

    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=(1, 64), padding="same"),
        nn.BatchNorm2d(16),
        nn.ELU(),
        nn.Conv2d(16, 32, kernel_size=(22, 1), groups=1),
        nn.BatchNorm2d(32),
        nn.ELU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(32, 4),
    ).to(device)

    try:
        out = model(x)
        loss = out.sum()
        loss.backward()
        print(f"Forward pass output shape: {tuple(out.shape)}")
        print("Backward pass completed without error.")
        print(f"Sample grad norm (first conv layer): "
              f"{model[0].weight.grad.norm().item():.6f}")
        print("\n[PASS] Conv2d/BatchNorm/backward all supported on this device.")
        return True
    except Exception as e:
        print(f"\n[FAIL] Error during forward/backward on {device}: {e}")
        return False


def check_moabb_env():
    print("\n" + "=" * 60)
    print("MOABB ENVIRONMENT CHECK")
    print("=" * 60)
    try:
        import moabb
        from moabb.datasets import BNCI2014_001
        print(f"MOABB version: {moabb.__version__}")

        ds = BNCI2014_001()
        print(f"MOABB data cache path (mne config): "
              f"{ds.data_path is not None}")
        import mne
        print(f"MNE data home: {mne.get_config('MNE_DATA')}")
        print("[PASS] MOABB import and dataset class instantiation succeeded.")
    except ImportError as e:
        print(f"[FAIL] MOABB/MNE import error: {e}")
    except Exception as e:
        print(f"[WARNING] MOABB instantiated but config incomplete: {e}")


def check_reproducibility():
    print("\n" + "=" * 60)
    print("SEED REPRODUCIBILITY CHECK")
    print("=" * 60)
    torch.manual_seed(42)
    a = torch.randn(3, 3)
    torch.manual_seed(42)
    b = torch.randn(3, 3)
    match = torch.allclose(a, b)
    print(f"Identical tensors from same seed: {match}")
    assert match, "Seeding is not reproducible — investigate torch install."
    print("[PASS] Seeding is reproducible.")


if __name__ == "__main__":
    device = check_mps_availability()
    success = run_conv_forward_backward(device)
    check_moabb_env()
    check_reproducibility()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Device to use for pretraining: {device}")
    print(f"Conv op support: {'OK' if success else 'FAILED — see errors above'}")
