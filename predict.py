"""
predict.py
==========
Run the trained detector and localiser on a single pipe signal.

Usage
-----
  # Simulate a signal and classify it
  python predict.py --pos 12.5 --severity 0.7 --snr 15

  # Classify a saved .npy waveform
  python predict.py --signal path/to/signal.npy

Options
-------
  --signal     Path to a .npy file containing a 1-D waveform (float32).
  --pos        Blockage position in metres (used when simulating).
  --severity   Blockage severity 0-1 (used when simulating, default 0.8).
  --snr        Signal-to-noise ratio in dB (used when simulating, default 15).
  --pipe_len   Pipe length in metres (default 30).
  --detector   Path to detector .pth (default outputs/models/detector.pth).
  --localiser  Path to localiser .pth (default outputs/models/localiser.pth).
  --specs      Path to spectrograms .npy, used to infer CNN input shape.
"""

import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.simulation import AcousticPipeSimulator, PipeConfig, Blockage
from src.features   import spectrogram_batch, normalise_signals, compute_spectrogram
from src.models     import BlockageDetector, BlockageLocaliser


FS = 44_100


def parse_args():
    p = argparse.ArgumentParser(description='Acoustic pipe inspection — single-signal inference')
    p.add_argument('--signal',    type=str,   default=None)
    p.add_argument('--pos',       type=float, default=None,
                   help='Blockage position in metres (simulate mode only)')
    p.add_argument('--severity',  type=float, default=0.8)
    p.add_argument('--snr',       type=float, default=15.0)
    p.add_argument('--pipe_len',  type=float, default=30.0)
    p.add_argument('--detector',  type=str,   default='outputs/models/detector.pth')
    p.add_argument('--localiser', type=str,   default='outputs/models/localiser.pth')
    p.add_argument('--specs',     type=str,   default='outputs/features/spectrograms.npy')
    return p.parse_args()


def load_or_simulate(args) -> np.ndarray:
    """Return a 1-D float32 waveform."""
    if args.signal:
        sig = np.load(args.signal).astype(np.float32)
        if sig.ndim != 1:
            raise ValueError(f'Expected 1-D signal array, got shape {sig.shape}')
        print(f'Loaded signal from {args.signal}  (length={len(sig)})')
        return sig

    sim = AcousticPipeSimulator(fs=FS, pulse_freq=2_000, pulse_duration=0.003)
    pipe = PipeConfig(length=args.pipe_len, speed_of_sound=343.0, attenuation_db_m=0.08)

    blockages = []
    if args.pos is not None:
        blockages = [Blockage(position=args.pos, severity=args.severity)]
        print(f'Simulating: blockage at {args.pos:.1f} m  severity={args.severity}  SNR={args.snr} dB')
    else:
        print(f'Simulating: clear pipe  SNR={args.snr} dB')

    _, sig, _ = sim.simulate(pipe, blockages, snr_db=args.snr, signal_duration=0.35)
    return sig


def run_detector(sig: np.ndarray, spec_path: str, model_path: str, device: str) -> tuple:
    """Returns (prediction_label, confidence)."""
    # Infer F, T from existing spectrogram batch
    if os.path.exists(spec_path):
        _, _, F, T = np.load(spec_path, mmap_mode='r').shape
    else:
        # Fall back: compute from the signal itself
        _, _, sample_s = compute_spectrogram(sig, fs=FS)
        F, T = sample_s.shape[0], 64

    spec = spectrogram_batch(sig[np.newaxis, :], fs=FS, nperseg=512, noverlap=384,
                             freq_max=8_000, target_time_bins=T)  # (1,1,F,T)

    model = BlockageDetector(freq_bins=F, time_bins=T)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(spec))          # (1, 2)
        probs  = torch.softmax(logits, dim=1).numpy()[0]

    pred  = int(np.argmax(probs))
    conf  = float(probs[pred])
    return pred, conf, probs


def run_localiser(sig: np.ndarray, pipe_len: float, model_path: str, device: str) -> float:
    """Returns estimated blockage position in metres."""
    sig_norm = normalise_signals(sig[np.newaxis, :])[0]    # (L,)
    x = torch.tensor(sig_norm[np.newaxis, np.newaxis, :])  # (1,1,L)

    model = BlockageLocaliser()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        pos_norm = model(x).item()

    return pos_norm * pipe_len


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sig = load_or_simulate(args)

    # --- Detection ---
    if not os.path.exists(args.detector):
        print(f'Detector model not found at {args.detector}. Run run_pipeline.py first.')
        sys.exit(1)

    pred, conf, probs = run_detector(sig, args.specs, args.detector, device)
    label = 'BLOCKED' if pred == 1 else 'CLEAR'

    print()
    print(f'Detection result : {label}  (confidence {conf:.1%})')
    print(f'  P(clear)={probs[0]:.3f}  P(blocked)={probs[1]:.3f}')

    # --- Localisation (only if blocked) ---
    if pred == 1:
        if not os.path.exists(args.localiser):
            print(f'Localiser model not found at {args.localiser}.')
        else:
            pos_m = run_localiser(sig, args.pipe_len, args.localiser, device)
            print(f'Estimated blockage position : {pos_m:.2f} m from transmitter end')
            print(f'  (pipe length assumed = {args.pipe_len:.1f} m)')


if __name__ == '__main__':
    main()
