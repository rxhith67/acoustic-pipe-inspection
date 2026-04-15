"""
simulation.py
=============
Physics-based acoustic pulse echo simulation for non-invasive pipe inspection.

The model mirrors real acoustic pipe inspection:
  1. A Gaussian-modulated sine pulse is transmitted from one end of the pipe.
  2. Each blockage partially reflects the pulse; the echo arrives at
     t_echo = 2 * blockage_position / speed_of_sound  (round-trip time).
  3. Reflected amplitude scales with blockage severity and distance attenuation.
  4. Additive white Gaussian noise is applied at a configurable SNR.

This lets us generate unlimited labelled training data without hardware.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PipeConfig:
    """Physical parameters of the pipe under inspection."""
    length: float           # metres
    speed_of_sound: float   # m/s  (343 = air; ~1480 = water; ~5000 = steel)
    attenuation_db_m: float # signal attenuation in dB per metre


@dataclass
class Blockage:
    """A single blockage (or foreign object) inside the pipe."""
    position: float   # metres from the transmitter end
    severity: float   # reflection coefficient in [0, 1]; 1 = full blockage


# ---------------------------------------------------------------------------
# Default pipe preset (realistic telecom conduit filled with air)
# ---------------------------------------------------------------------------

DEFAULT_PIPE = PipeConfig(
    length=30.0,
    speed_of_sound=343.0,
    attenuation_db_m=0.08,
)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class AcousticPipeSimulator:
    """
    Generates synthetic acoustic signals for a pipe with zero or more blockages.

    Parameters
    ----------
    fs : int
        Sampling frequency in Hz (default 44 100 Hz — standard audio rate).
    pulse_freq : float
        Centre frequency of the interrogation pulse in Hz.
    pulse_duration : float
        Duration of the Gaussian-windowed sine pulse in seconds.
    """

    def __init__(self, fs: int = 44_100, pulse_freq: float = 2_000.0,
                 pulse_duration: float = 0.003):
        self.fs = fs
        self.pulse_freq = pulse_freq
        self.pulse_duration = pulse_duration
        self._pulse = self._make_pulse()

    # ------------------------------------------------------------------
    # Pulse generation
    # ------------------------------------------------------------------

    def _make_pulse(self) -> np.ndarray:
        """
        Gaussian-modulated sine burst — approximates a real acoustic transducer
        output. The Gaussian envelope suppresses spectral leakage.
        """
        n = int(self.fs * self.pulse_duration)
        t = np.linspace(0.0, self.pulse_duration, n)
        centre = self.pulse_duration / 2.0
        sigma  = self.pulse_duration / 6.0          # 3-sigma fits inside window
        envelope = np.exp(-0.5 * ((t - centre) / sigma) ** 2)
        return envelope * np.sin(2.0 * np.pi * self.pulse_freq * t)

    # ------------------------------------------------------------------
    # Single signal simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        pipe: PipeConfig,
        blockages: List[Blockage],
        snr_db: float = 20.0,
        signal_duration: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Simulate one transmission event.

        Returns
        -------
        t : ndarray, shape (N,)
            Time axis in seconds.
        received : ndarray, shape (N,)
            Simulated received waveform (transmitted pulse + echoes + noise).
        echo_times : list of float
            Ground-truth echo arrival times for each blockage (seconds).
        """
        # Duration must be long enough to capture the farthest echo
        max_travel = 2.0 * pipe.length / pipe.speed_of_sound
        if signal_duration is None:
            signal_duration = max_travel + 0.05   # 50 ms margin

        N = int(self.fs * signal_duration)
        received = np.zeros(N, dtype=np.float32)
        pulse = self._pulse
        P = len(pulse)

        # --- Transmitted pulse at t = 0 (direct signal) ---
        received[:P] += pulse.astype(np.float32)

        echo_times: List[float] = []

        # --- Blockage echoes ---
        for b in blockages:
            if not (0 < b.position < pipe.length):
                continue                            # ignore out-of-range blockages

            t_echo  = 2.0 * b.position / pipe.speed_of_sound
            i_echo  = int(t_echo * self.fs)
            # Linear amplitude attenuation with distance
            atten   = 10 ** (-pipe.attenuation_db_m * b.position / 20.0)
            amp     = b.severity * atten

            end_idx = i_echo + P
            if end_idx <= N:
                received[i_echo:end_idx] += (pulse * amp).astype(np.float32)
            else:
                clip = N - i_echo
                received[i_echo:N] += (pulse[:clip] * amp).astype(np.float32)

            echo_times.append(t_echo)

        # --- Pipe-end reflection (weak, always present) ---
        t_end  = 2.0 * pipe.length / pipe.speed_of_sound
        i_end  = int(t_end * self.fs)
        atten_end = 10 ** (-pipe.attenuation_db_m * pipe.length / 20.0)
        if i_end + P <= N:
            received[i_end:i_end + P] += (pulse * 0.05 * atten_end).astype(np.float32)

        # --- Additive white Gaussian noise ---
        sig_power   = float(np.mean(received ** 2)) + 1e-12
        noise_power = sig_power / (10 ** (snr_db / 10.0))
        noise = np.random.normal(0.0, np.sqrt(noise_power), N).astype(np.float32)
        received += noise

        t_axis = np.linspace(0.0, signal_duration, N, dtype=np.float32)
        return t_axis, received, echo_times

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------

    def generate_dataset(
        self,
        n_samples: int = 2_000,
        pipe_length_range: Tuple[float, float] = (10.0, 50.0),
        max_blockages: int = 3,
        snr_range: Tuple[float, float] = (10.0, 30.0),
        fixed_duration: float = 0.35,
        seed: Optional[int] = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a labelled dataset of synthetic pipe signals.

        Returns
        -------
        signals : ndarray, shape (n_samples, N)
            Raw waveforms, all padded / truncated to fixed_duration.
        labels : ndarray, shape (n_samples,)
            Binary: 1 = at least one blockage present, 0 = clear pipe.
        positions : ndarray, shape (n_samples, max_blockages)
            Normalised blockage positions in [0, 1].
            Padded with -1.0 where no blockage exists.
        """
        if seed is not None:
            np.random.seed(seed)

        N = int(self.fs * fixed_duration)
        signals   = np.zeros((n_samples, N), dtype=np.float32)
        labels    = np.zeros(n_samples,      dtype=np.int64)
        positions = np.full((n_samples, max_blockages), -1.0, dtype=np.float32)

        for i in range(n_samples):
            pipe_len = float(np.random.uniform(*pipe_length_range))
            pipe = PipeConfig(
                length=pipe_len,
                speed_of_sound=343.0,
                attenuation_db_m=float(np.random.uniform(0.04, 0.12)),
            )

            n_blk = int(np.random.randint(0, max_blockages + 1))
            blockages: List[Blockage] = []
            if n_blk > 0:
                # Ensure blockages are at least 1 m from each end
                raw_pos = np.random.uniform(1.0, pipe_len - 1.0, n_blk)
                raw_pos.sort()
                severities = np.random.uniform(0.3, 1.0, n_blk)
                for p, s in zip(raw_pos, severities):
                    blockages.append(Blockage(position=float(p), severity=float(s)))

            snr = float(np.random.uniform(*snr_range))
            _, sig, _ = self.simulate(pipe, blockages, snr_db=snr,
                                      signal_duration=fixed_duration)

            # Truncate or zero-pad to fixed length
            actual_len = len(sig)
            if actual_len >= N:
                signals[i] = sig[:N]
            else:
                signals[i, :actual_len] = sig

            labels[i] = 1 if n_blk > 0 else 0
            for j, b in enumerate(blockages[:max_blockages]):
                positions[i, j] = b.position / pipe_len   # normalised 0–1

        return signals, labels, positions


# ---------------------------------------------------------------------------
# Utility: echo-time → position
# ---------------------------------------------------------------------------

def echo_time_to_position(echo_time: float, speed_of_sound: float = 343.0) -> float:
    """Convert a measured echo round-trip time to a blockage position in metres."""
    return echo_time * speed_of_sound / 2.0
