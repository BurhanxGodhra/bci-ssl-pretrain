import torch
from src.models.encoder import SSLPretrainModel
from src.losses.nt_xent import NTXentLoss
from src.utils.device import get_device  # we'll write this tiny helper below

# Synthetic batch: mimics BNCI2014_001 shape (22 channels, 1001 timepoints)
torch.manual_seed(42)
device = get_device()
print(f"Using device: {device}")

N, n_channels, n_timepoints = 16, 22, 1001
view1 = torch.randn(N, n_channels, n_timepoints, device=device)
view2 = torch.randn(N, n_channels, n_timepoints, device=device)

model = SSLPretrainModel(n_channels=n_channels, n_timepoints=n_timepoints).to(device)
loss_fn = NTXentLoss(temperature=0.5)

z1 = model(view1)
z2 = model(view2)
print(f"Projected embedding shape: {z1.shape}")  # expect (16, 64)

loss = loss_fn(z1, z2)
print(f"Initial NT-Xent loss (untrained, random weights): {loss.item():.4f}")

# Sanity bound: with N=16 (2N=32 candidates), random-chance loss ≈ ln(2N-1) = ln(31) ≈ 3.43
import math
expected_random = math.log(2 * N - 1)
print(f"Expected ~random-init loss: ln(2N-1) = {expected_random:.4f}")

loss.backward()
enc_grad_norm = model.encoder.temporal_conv[0].weight.grad.norm().item()
print(f"Encoder first-layer grad norm: {enc_grad_norm:.6f}")
assert enc_grad_norm > 0, "No gradient reaching encoder -- check graph connectivity"
print("\n[PASS] Forward + backward through encoder -> projection head -> NT-Xent works.")
