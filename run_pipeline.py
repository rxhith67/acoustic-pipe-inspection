"""
run_pipeline.py
===============
End-to-end script: generate data → extract features → train → evaluate.

Run from the project root:
    python run_pipeline.py

All outputs (models, plots, features) are saved to outputs/.
"""

import os, sys, argparse, time
import numpy as np

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, os.path.dirname(__file__))

from src.simulation import AcousticPipeSimulator, PipeConfig, Blockage, echo_time_to_position
from src.features   import (
    spectrogram_batch, normalise_signals,
    detect_echo_peaks, extract_envelope_features,
)
from src.models import train_detector, train_localiser, BlockageDetector, BlockageLocaliser


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Acoustic Pipe Inspection Pipeline')
    p.add_argument('--n_samples',  type=int,   default=2_000, help='Dataset size')
    p.add_argument('--epochs',     type=int,   default=30,    help='Training epochs')
    p.add_argument('--batch_size', type=int,   default=64)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--snr_min',    type=float, default=5.0)
    p.add_argument('--snr_max',    type=float, default=25.0)
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--skip_train', action='store_true',
                   help='Skip training, only generate data and features')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helper: pretty section header
# ---------------------------------------------------------------------------

def section(title: str):
    print(f'\n{"="*60}')
    print(f'  {title}')
    print(f'{"="*60}')


# ---------------------------------------------------------------------------
# Step 1: Generate dataset
# ---------------------------------------------------------------------------

def step_generate(args) -> tuple:
    section('STEP 1 — Generating Synthetic Dataset')
    os.makedirs('data',             exist_ok=True)
    os.makedirs('outputs/plots',    exist_ok=True)
    os.makedirs('outputs/features', exist_ok=True)
    os.makedirs('outputs/models',   exist_ok=True)

    sim = AcousticPipeSimulator(fs=44_100, pulse_freq=2_000, pulse_duration=0.003)
    t0 = time.time()

    signals, labels, positions = sim.generate_dataset(
        n_samples=args.n_samples,
        pipe_length_range=(10.0, 50.0),
        max_blockages=3,
        snr_range=(args.snr_min, args.snr_max),
        fixed_duration=0.35,
        seed=args.seed,
    )

    np.save('data/signals.npy',   signals)
    np.save('data/labels.npy',    labels)
    np.save('data/positions.npy', positions)

    elapsed = time.time() - t0
    print(f'Generated {args.n_samples} samples in {elapsed:.1f}s')
    print(f'  signals  : {signals.shape}')
    print(f'  labels   : clear={( labels==0).sum()}  blocked={(labels==1).sum()}')
    return signals, labels, positions


# ---------------------------------------------------------------------------
# Step 2: Feature extraction
# ---------------------------------------------------------------------------

def step_features(signals: np.ndarray) -> tuple:
    section('STEP 2 — Feature Extraction')
    t0 = time.time()

    print('Computing STFT spectrograms...')
    specs = spectrogram_batch(signals, fs=44_100, nperseg=512, noverlap=384,
                              freq_max=8_000, target_time_bins=64)
    np.save('outputs/features/spectrograms.npy', specs)
    print(f'  Spectrograms saved: {specs.shape}  ({specs.nbytes/1e6:.1f} MB)')

    print('Normalising raw signals...')
    sigs_norm = normalise_signals(signals)
    np.save('outputs/features/signals_norm.npy', sigs_norm)
    print(f'  Normalised signals saved: {sigs_norm.shape}')

    print(f'Feature extraction done in {time.time()-t0:.1f}s')
    return specs, sigs_norm


# ---------------------------------------------------------------------------
# Step 3: Train detector
# ---------------------------------------------------------------------------

def step_train_detector(specs: np.ndarray, labels: np.ndarray, args) -> dict:
    section('STEP 3 — Training Blockage Detector (2-D CNN)')
    model, history = train_detector(
        spectrograms=specs,
        labels=labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=0.2,
        save_path='outputs/models/detector.pth',
    )
    best_acc = max(history['val_acc'])
    print(f'\nBest validation accuracy : {best_acc:.4f}')
    return history


# ---------------------------------------------------------------------------
# Step 4: Train localiser
# ---------------------------------------------------------------------------

def step_train_localiser(sigs_norm: np.ndarray, positions: np.ndarray, labels: np.ndarray, args) -> dict:
    section('STEP 4 — Training Blockage Localiser (1-D CNN)')

    # Use all blocked samples, not just single-blockage ones.
    # Target = position of the first (nearest) blockage, which is positions[:, 0]
    # for all samples with at least one blockage.
    mask = labels == 1
    sigs_blk = sigs_norm[mask]
    pos_blk  = positions[mask, 0]   # first blockage position (sorted nearest-first)

    print(f'Blocked samples used for localiser: {mask.sum()}')
    model, history = train_localiser(
        signals=sigs_blk,
        positions=pos_blk,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=0.2,
        save_path='outputs/models/localiser.pth',
    )
    best_mae = min(history['val_mae'])
    print(f'\nBest val MAE (normalised) : {best_mae:.4f}')
    print(f'Best val MAE (30m pipe)   : {best_mae * 30:.2f} m')
    return history


# ---------------------------------------------------------------------------
# Step 5: DSP baseline evaluation
# ---------------------------------------------------------------------------

def step_dsp_baseline(signals: np.ndarray, labels: np.ndarray):
    section('STEP 5 — Classical DSP Baseline Evaluation')
    FS = 44_100
    preds = []
    for sig in signals:
        peak_t, _ = detect_echo_peaks(sig, fs=FS, min_prominence=0.04)
        preds.append(1 if len(peak_t) > 0 else 0)
    preds = np.array(preds)

    acc = (preds == labels).mean()
    tp  = ((preds == 1) & (labels == 1)).sum()
    fp  = ((preds == 1) & (labels == 0)).sum()
    fn  = ((preds == 0) & (labels == 1)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    print(f'Accuracy  : {acc:.4f}')
    print(f'Precision : {precision:.4f}')
    print(f'Recall    : {recall:.4f}')
    print(f'F1 Score  : {f1:.4f}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    print('\nAcoustic Pipe Inspection Pipeline')
    print(f'  n_samples={args.n_samples}  epochs={args.epochs}  seed={args.seed}')

    signals, labels, positions = step_generate(args)
    specs, sigs_norm           = step_features(signals)

    if not args.skip_train:
        det_history   = step_train_detector(specs, labels, args)
        loc_history   = step_train_localiser(sigs_norm, positions, labels, args)

    step_dsp_baseline(signals, labels)

    section('PIPELINE COMPLETE')
    print('Saved:')
    print('  data/signals.npy              — raw waveforms')
    print('  data/labels.npy               — binary labels')
    print('  data/positions.npy            — blockage positions')
    print('  outputs/features/spectrograms.npy')
    print('  outputs/features/signals_norm.npy')
    if not args.skip_train:
        print('  outputs/models/detector.pth   — trained 2-D CNN detector')
        print('  outputs/models/localiser.pth  — trained 1-D CNN localiser')
    print('\nRun the notebooks (01–04) for full visualisations and analysis.')


if __name__ == '__main__':
    main()
