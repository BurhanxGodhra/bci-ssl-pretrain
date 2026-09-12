"""
LSL (Lab Streaming Layer) utilities. Same reading code path handles both
real hardware (if an LSL outlet is detected on the network) and our own
simulated replay outlet -- the app doesn't need to know which one it's using.
"""
import time
import threading
import numpy as np

try:
    import pylsl
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False


def find_real_eeg_streams(timeout: float = 2.0):
    """Scans the network for active EEG LSL outlets (real hardware)."""
    if not LSL_AVAILABLE:
        return []
    streams = pylsl.resolve_byprop("type", "EEG", timeout=timeout)
    return streams


def start_replay_outlet(X_trial_sequence: np.ndarray, channel_names: list, sfreq: float, stream_name: str = "BCI_SSL_Replay"):
    """
    Publishes pre-recorded EEG (our holdout subject's trials, concatenated)
    as a REAL LSL outlet at the correct sample rate. Downstream code reads
    this exactly like it would read a real headset -- no separate code path.
    Runs in a background thread so it doesn't block the Streamlit UI.
    """
    if not LSL_AVAILABLE:
        raise RuntimeError("pylsl not available -- cannot start replay outlet.")

    n_channels = len(channel_names)
    info = pylsl.StreamInfo(stream_name, "EEG", n_channels, sfreq, "float32", "bci_ssl_replay_001")
    chns = info.desc().append_child("channels")
    for ch in channel_names:
        chns.append_child("channel").append_child_value("label", ch)

    outlet = pylsl.StreamOutlet(info)
    flat_signal = X_trial_sequence.transpose(1, 0, 2).reshape(n_channels, -1).T  # (n_samples, n_channels)

    def _stream_loop():
        interval = 1.0 / sfreq
        for sample in flat_signal:
            outlet.push_sample(sample.tolist())
            time.sleep(interval)

    thread = threading.Thread(target=_stream_loop, daemon=True)
    thread.start()
    return outlet


def connect_to_stream(stream_info, buffer_seconds: float = 4.0):
    """Connects an inlet to any resolved stream (real or replay) and
    returns a callable to pull the latest buffered chunk."""
    inlet = pylsl.StreamInlet(stream_info)
    sfreq = stream_info.nominal_srate()
    n_channels = stream_info.channel_count()
    buffer_size = int(buffer_seconds * sfreq)
    return inlet, sfreq, n_channels, buffer_size


def check_signal_quality(chunk: np.ndarray, flat_threshold: float = 1e-6, noise_threshold: float = 500.0) -> dict:
    """
    Basic signal-quality heuristics: flags channels that are flat (likely
    poor electrode contact) or excessively noisy (likely a loose/artifact
    -ridden connection). This is what a real end-user needs to see BEFORE
    calibration, not after it fails.
    """
    per_channel_std = chunk.std(axis=0)
    flat_channels = np.where(per_channel_std < flat_threshold)[0].tolist()
    noisy_channels = np.where(per_channel_std > noise_threshold)[0].tolist()

    ok = len(flat_channels) == 0 and len(noisy_channels) == 0
    return {
        "ok": ok,
        "flat_channels": flat_channels,
        "noisy_channels": noisy_channels,
        "per_channel_std": per_channel_std.tolist(),
    }