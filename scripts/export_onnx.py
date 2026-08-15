"""
Exports the pretrained EEGNetEncoder to ONNX format, with mandatory
numerical parity verification against the original PyTorch model.
"""
import numpy as np
import torch
import onnx
import onnxruntime as ort

from src.models.encoder import EEGNetEncoder


def export_encoder_to_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 18,
):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder = EEGNetEncoder(
        n_channels=ckpt["n_channels"],
        n_timepoints=ckpt["n_timepoints"],
        embed_dim=ckpt["embed_dim"],
    )
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()  # CRITICAL: BatchNorm must use running stats, not batch stats, during export

    dummy_input = torch.randn(1, ckpt["n_channels"], ckpt["n_timepoints"])

    torch.onnx.export(
        encoder,
        dummy_input,
        output_path,
        input_names=["eeg_trial"],
        output_names=["embedding"],
        dynamic_axes={
            "eeg_trial": {0: "batch_size"},   # allow variable batch size at inference
            "embedding": {0: "batch_size"},
        },
        opset_version=opset_version,
    )
    print(f"Exported ONNX model -> {output_path}")

    # Structural validation: is this a well-formed ONNX graph?
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("[PASS] ONNX model structure is valid.")

    return encoder, ckpt


def verify_numerical_parity(
    pytorch_encoder: EEGNetEncoder,
    onnx_path: str,
    n_channels: int,
    n_timepoints: int,
    n_test_samples: int = 10,
    atol: float = 1e-4,
    seed: int = 42,
):
    """
    The real test: do PyTorch and ONNX produce matching outputs on the
    SAME inputs? Without this, the export is an unverified claim.
    """
    torch.manual_seed(seed)
    session = ort.InferenceSession(onnx_path)

    max_abs_diff = 0.0
    pytorch_encoder.eval()

    with torch.no_grad():
        for i in range(n_test_samples):
            x = torch.randn(1, n_channels, n_timepoints)

            pt_out = pytorch_encoder(x).numpy()
            onnx_out = session.run(None, {"eeg_trial": x.numpy()})[0]

            diff = np.abs(pt_out - onnx_out).max()
            max_abs_diff = max(max_abs_diff, diff)

    print(f"\nParity check over {n_test_samples} random inputs:")
    print(f"  Max absolute difference: {max_abs_diff:.2e}")
    print(f"  Tolerance: {atol:.2e}")

    assert max_abs_diff < atol, (
        f"ONNX export FAILED parity check: max diff {max_abs_diff:.2e} "
        f"exceeds tolerance {atol:.2e}. Export is NOT numerically equivalent."
    )
    print(f"[PASS] ONNX export is numerically equivalent to PyTorch (within {atol:.0e}).")


if __name__ == "__main__":
    checkpoint_path = "checkpoints/encoder_multi_full_e25.pt"
    output_path = "checkpoints/encoder_multi_full_e25.onnx"

    encoder, ckpt = export_encoder_to_onnx(checkpoint_path, output_path)
    verify_numerical_parity(
        encoder, output_path,
        n_channels=ckpt["n_channels"], n_timepoints=ckpt["n_timepoints"],
    )