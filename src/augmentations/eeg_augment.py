"""
EEG-specific augmentations for contrastive pretraining.
All operate on a single trial: (n_channels, n_timepoints) numpy array.
Designed to preserve motor-imagery-relevant signal while perturbing
nuisance variation (electrode noise, timing jitter, individual spectral
peak variation).
"""
import numpy as np


class SpatialChannelDropout:
    """
    Randomly zero out a fraction of EEG channels.
    Simulates electrode dropout/bad contact; forces the model to rely on
    distributed spatial patterns across motor cortex rather than any
    single electrode (important since C3/C4/Cz are classic MI channels
    but real deployments have variable electrode quality).
    """
    def __init__(self, max_drop_fraction: float = 0.2, p: float = 0.5):
        self.max_drop_fraction = max_drop_fraction
        self.p = p

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() > self.p:
            return x
        x = x.copy()
        n_channels = x.shape[0]
        n_drop = rng.integers(1, max(2, int(n_channels * self.max_drop_fraction)))
        drop_idx = rng.choice(n_channels, size=n_drop, replace=False)
        x[drop_idx, :] = 0.0
        return x


class TemporalJitter:
    """
    Randomly crop a slightly shifted window from a padded trial.
    Simulates imprecise cue-onset timing / reaction time variability
    across subjects and trials.
    """
    def __init__(self, max_shift_samples: int = 25, p: float = 0.5):
        self.max_shift_samples = max_shift_samples
        self.p = p

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() > self.p:
            return x
        n_channels, n_timepoints = x.shape
        shift = rng.integers(-self.max_shift_samples, self.max_shift_samples + 1)
        if shift == 0:
            return x.copy()
        x_shifted = np.zeros_like(x)
        if shift > 0:
            x_shifted[:, shift:] = x[:, : n_timepoints - shift]
        else:
            x_shifted[:, : n_timepoints + shift] = x[:, -shift:]
        return x_shifted


class FrequencyBandMasking:
    """
    Zero out a random contiguous frequency band in the FFT magnitude
    spectrum, then reconstruct via inverse FFT. Analogous to SpecAugment.
    Prevents over-reliance on one narrow band (e.g. only mu-rhythm),
    encouraging robustness to individual variation in dominant MI frequency.
    """
    def __init__(self, sfreq: float = 250.0, max_mask_hz: float = 4.0, p: float = 0.5):
        self.sfreq = sfreq
        self.max_mask_hz = max_mask_hz
        self.p = p

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() > self.p:
            return x
        n_channels, n_timepoints = x.shape
        freqs = np.fft.rfftfreq(n_timepoints, d=1.0 / self.sfreq)
        fft_vals = np.fft.rfft(x, axis=-1)

        mask_width_hz = rng.uniform(1.0, self.max_mask_hz)
        f_center = rng.uniform(freqs[1], freqs[-1] - mask_width_hz)
        band_mask = (freqs >= f_center) & (freqs <= f_center + mask_width_hz)

        fft_vals = fft_vals.copy()
        fft_vals[:, band_mask] = 0.0
        x_masked = np.fft.irfft(fft_vals, n=n_timepoints, axis=-1)
        return x_masked.astype(x.dtype)


class GaussianNoiseInjection:
    """
    Additive Gaussian noise scaled relative to each channel's own signal
    std. Simulates realistic sensor/environmental noise floor without
    overwhelming true signal amplitude.
    """
    def __init__(self, noise_std_fraction: float = 0.05, p: float = 0.5):
        self.noise_std_fraction = noise_std_fraction
        self.p = p

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() > self.p:
            return x
        channel_std = x.std(axis=-1, keepdims=True) + 1e-8
        noise = rng.normal(
            loc=0.0, scale=self.noise_std_fraction * channel_std, size=x.shape
        )
        return (x + noise).astype(x.dtype)


class ComposeAugmentations:
    """Applies a sequence of augmentations with a single shared RNG stream."""
    def __init__(self, augmentations: list, seed: int = None):
        self.augmentations = augmentations
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for aug in self.augmentations:
            x = aug(x, self.rng)
        return x


def build_default_pipeline(sfreq: float = 250.0, seed: int = None) -> ComposeAugmentations:
    """Standard augmentation pipeline for contrastive view generation."""
    return ComposeAugmentations(
        [
            SpatialChannelDropout(max_drop_fraction=0.2, p=0.5),
            TemporalJitter(max_shift_samples=25, p=0.5),
            FrequencyBandMasking(sfreq=sfreq, max_mask_hz=4.0, p=0.5),
            GaussianNoiseInjection(noise_std_fraction=0.05, p=0.5),
        ],
        seed=seed,
    )
