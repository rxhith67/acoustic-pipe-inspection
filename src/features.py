"""
features.py
===========
Signal processing and feature extraction for acoustic pipe inspection signals.

Two representation strategies are provided:

1. **STFT spectrogram** — 2-D time-frequency image fed to a 2-D CNN.
2. **Sliding-window statistics** — compact 1-D feature vector per window,
   useful for traditional ML baselines and interpretability.

Both are designed to expose the echo structure that encodes blockage
presence and position.
"""

import numpy as np
from scipy.signal import stft, find_peaks, correlate
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# STFT / Spectrogram
# ---------------------------------------------------------------------------

def compute_spectrogram(
    signal: np.ndarray,
    fs: int = 44_100,
    nperseg: int = 512,
    noverlap: int = 384,
    freq_max: float = 8_000.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a magnitude spectrogram (dB scale) from a 1-D signal.

    Parameters
    ----------
    signal   : 1-D array, the received waveform.
    fs       : sampling frequency in Hz.
    nperseg  : FFT window length in samples.
    noverlap : overlap between consecutive windows.
    freq_max : upper frequency limit to keep (Hz) — reduces image height.

    Returns
    -------
    f   : frequency axis (Hz), shape (F,)
    t   : time axis (s),       shape (T,)
    Sdb : magnitude spectrogram (dB), shape (F, T)
    """
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Keep only frequencies up to freq_max
    f_mask = f <= freq_max
    f, Zxx = f[f_mask], Zxx[f_mask]

    Smag = np.abs(Zxx).astype(np.float32)
    Sdb  = 20.0 * np.log10(Smag + 1e-8)
    return f, t, Sdb


def spectrogram_batch(
    signals: np.ndarray,
    fs: int = 44_100,
    nperseg: int = 512,
    noverlap: int = 384,
    freq_max: float = 8_000.0,
    target_time_bins: Optional[int] = None,
) -> np.ndarray:
    """
    Convert a batch of raw waveforms to normalised spectrogram tensors.

    Parameters
    ----------
    signals          : shape (B, N) — batch of raw waveforms.
    target_time_bins : if given, spectrograms are interpolated to this
                       width so all samples are the same shape.

    Returns
    -------
    specs : shape (B, 1, F, T) — channel-first tensor suitable for CNN.
    """
    sample_f, sample_t, sample_s = compute_spectrogram(
        signals[0], fs=fs, nperseg=nperseg, noverlap=noverlap, freq_max=freq_max
    )
    F = sample_s.shape[0]
    T = sample_s.shape[1] if target_time_bins is None else target_time_bins

    out = np.zeros((len(signals), 1, F, T), dtype=np.float32)

    for i, sig in enumerate(signals):
        _, _, Sdb = compute_spectrogram(sig, fs=fs, nperseg=nperseg,
                                        noverlap=noverlap, freq_max=freq_max)
        if target_time_bins is not None and Sdb.shape[1] != T:
            # Simple nearest-neighbour resize along time axis
            idx = np.round(np.linspace(0, Sdb.shape[1] - 1, T)).astype(int)
            Sdb = Sdb[:, idx]

        # Normalise each spectrogram to [0, 1] independently
        mn, mx = Sdb.min(), Sdb.max()
        Sdb = (Sdb - mn) / (mx - mn + 1e-8)
        out[i, 0] = Sdb

    return out


# ---------------------------------------------------------------------------
# Sliding-window feature extraction (envelope + statistics)
# ---------------------------------------------------------------------------

def extract_envelope_features(
    signal: np.ndarray,
    fs: int = 44_100,
    window_sec: float = 0.005,
    hop_sec: float = 0.0025,
) -> np.ndarray:
    """
    Extract a compact feature vector per sliding window using the signal
    envelope (Hilbert-derived RMS).

    Each window returns: [rms, peak, zero_crossing_rate, spectral_centroid]
    → 4 features × n_windows → suitable for anomaly scoring over time.

    Returns
    -------
    features : shape (n_windows, 4)
    """
    win   = int(window_sec * fs)
    hop   = int(hop_sec * fs)
    N     = len(signal)
    starts = range(0, N - win, hop)
    feats  = []

    for s in starts:
        frame = signal[s:s + win].astype(np.float64)

        # RMS energy
        rms = float(np.sqrt(np.mean(frame ** 2)))

        # Peak amplitude
        peak = float(np.max(np.abs(frame)))

        # Zero-crossing rate
        zcr = float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0)

        # Spectral centroid
        spectrum = np.abs(np.fft.rfft(frame))
        freqs    = np.fft.rfftfreq(len(frame), d=1.0 / fs)
        sc_denom = spectrum.sum() + 1e-12
        sc       = float(np.sum(freqs * spectrum) / sc_denom)

        feats.append([rms, peak, zcr, sc])

    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Pulse template
# ---------------------------------------------------------------------------

def make_pulse(
    fs: int = 44_100,
    pulse_freq: float = 2_000.0,
    pulse_duration: float = 0.003,
) -> np.ndarray:
    """
    Generate the Gaussian-modulated sine pulse used by the simulator.
    Returned as float32 array, normalised to unit peak amplitude.
    Used as the matched-filter template.
    """
    n = int(fs * pulse_duration)
    t = np.linspace(0.0, pulse_duration, n)
    centre = pulse_duration / 2.0
    sigma  = pulse_duration / 6.0
    envelope = np.exp(-0.5 * ((t - centre) / sigma) ** 2)
    pulse = envelope * np.sin(2.0 * np.pi * pulse_freq * t)
    return (pulse / (np.abs(pulse).max() + 1e-12)).astype(np.float32)


# ---------------------------------------------------------------------------
# Echo peak detection (classical DSP baseline)
# ---------------------------------------------------------------------------

def detect_echo_peaks(
    signal: np.ndarray,
    fs: int = 44_100,
    pulse_duration: float = 0.003,
    min_prominence: float = 0.05,
    min_separation_sec: float = 0.002,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect echo arrivals in a received signal using peak detection on the
    signal envelope.  This is the classical DSP approach — used as a
    baseline and for ground-truth comparison.

    Returns
    -------
    peak_times  : echo arrival times in seconds
    peak_ampls  : corresponding amplitudes
    """
    # Build rectified envelope via sliding-window RMS
    win  = int(pulse_duration * fs)
    hop  = max(1, win // 8)
    env  = np.array([
        np.sqrt(np.mean(signal[s:s + win] ** 2))
        for s in range(0, len(signal) - win, hop)
    ], dtype=np.float32)

    # Ignore the direct pulse (first ~pulse_duration seconds)
    ignore_samples = int(pulse_duration * fs / hop) + 2
    search_env = env.copy()
    search_env[:ignore_samples] = 0.0

    min_dist = max(1, int(min_separation_sec * fs / hop))
    peaks, props = find_peaks(
        search_env,
        prominence=min_prominence * search_env.max() + 1e-12,
        distance=min_dist,
    )

    # Convert envelope indices back to time
    peak_times = peaks * hop / fs
    peak_ampls = env[peaks]
    return peak_times, peak_ampls


# ---------------------------------------------------------------------------
# Matched filter (theoretically optimal echo detector)
# ---------------------------------------------------------------------------

def matched_filter_echo_peaks(
    signal: np.ndarray,
    pulse_template: np.ndarray,
    fs: int = 44_100,
    pulse_duration: float = 0.003,
    min_prominence: float = 0.1,
    min_separation_sec: float = 0.002,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect echo arrivals using matched filtering.

    Cross-correlates the received signal with the known transmitted pulse
    template. This is the theoretically optimal linear detector for a known
    signal in additive white Gaussian noise — it maximises SNR at the echo
    arrival time.

    Compared to envelope peak detection, matched filtering:
    - Is more sensitive at low SNR (coherent integration gain)
    - Gives sharper, better-localised peaks (resolution ~ 1/bandwidth)
    - Suppresses off-pulse noise more effectively

    Parameters
    ----------
    signal : 1-D float array, the received waveform.
    pulse_template : 1-D float array, the transmitted pulse shape (from make_pulse()).
    fs : sampling frequency in Hz.
    pulse_duration : duration of the pulse in seconds (used to set the
        ignore window around the direct transmission).
    min_prominence : minimum peak prominence as a fraction of the global max
        of the matched-filter output.
    min_separation_sec : minimum time between detected echoes (seconds).

    Returns
    -------
    peak_times : echo arrival times in seconds
    peak_ampls : matched-filter output amplitude at each peak
    """
    P = len(pulse_template)

    # Cross-correlate: output[k] = sum_i signal[i] * template[i - k]
    # scipy.signal.correlate with mode='full' returns length N+P-1.
    # Slicing [P-1:] aligns the output so index k = lag k samples.
    mf = correlate(signal.astype(np.float64),
                   pulse_template.astype(np.float64),
                   mode='full')[P - 1:]
    mf = np.abs(mf).astype(np.float32)

    # Blank out the direct pulse window (first pulse_duration seconds)
    ignore = int(pulse_duration * fs) + P
    search = mf.copy()
    search[:ignore] = 0.0

    if search.max() < 1e-12:
        return np.array([]), np.array([])

    search_norm = search / search.max()

    min_dist = max(1, int(min_separation_sec * fs))
    peaks, _ = find_peaks(
        search_norm,
        prominence=min_prominence,
        distance=min_dist,
    )

    peak_times = peaks / fs
    peak_ampls = mf[peaks]
    return peak_times, peak_ampls


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalise_signals(signals: np.ndarray) -> np.ndarray:
    """
    Z-score normalise each signal independently.
    Input shape: (B, N)  Output shape: (B, N)
    """
    mu  = signals.mean(axis=1, keepdims=True)
    std = signals.std(axis=1,  keepdims=True) + 1e-8
    return ((signals - mu) / std).astype(np.float32)
