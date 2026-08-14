import torch

def get_device() -> torch.device:
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
