# Acoustic Pipe Inspection

Physics-based simulation and CNN pipeline for detecting and localising blockages in pipes using acoustic signals. Validated on real industrial audio from the MIMII pump dataset.

## What it does

1. Simulates acoustic wave propagation through pipes with randomised blockage positions, severities, and SNR levels — including pink (1/f) noise and multipath echoes
2. Extracts STFT spectrograms and normalised waveforms as features
3. Trains a 2D CNN on spectrograms to classify signals as clear or blocked
4. Trains a 1D CNN on raw waveforms to estimate blockage position along the pipe
5. Validates the same architecture on real MIMII pump recordings (no simulation)

## Results

### Synthetic pipeline (physics simulation)

| Model | Metric | Value |
|-------|--------|-------|
| 2D CNN Detector | Validation accuracy | 100% |
| 1D CNN Localiser | Validation MAE | 2.34 m on 30 m pipe |
| DSP Baseline | F1 Score | 0.860 |

### Real data (MIMII pump dataset)

4205 recordings across 4 pump IDs, 8.2:1 normal/abnormal imbalance.
Class-weighted `CrossEntropyLoss` (abnormal gets 7.9x more weight).

| Metric | Value |
|--------|-------|
| AUC | 0.836 |
| Abnormal recall | 63% |
| Normal recall | 92% |
| Accuracy | 89.1% |

## Project structure

```
acoustic_pipe_inspection/
├── src/
│   ├── simulation.py      # physics-based signal generation (pink noise, multipath)
│   ├── features.py        # STFT spectrograms, envelope features, peak detection
│   └── models.py          # CNN architectures and training loops
├── outputs/
│   └── models/
│       ├── detector.pth   # synthetic-trained detector
│       └── localiser.pth  # blockage position regressor
├── 01_simulation.ipynb    # signal generation and physics walkthrough
├── 02_signal_analysis.ipynb  # DSP baseline, peak detection, spectrograms
├── 03_detection.ipynb     # 2D CNN detector training and evaluation
├── 04_localisation.ipynb  # 1D CNN localiser, SNR ablation study
├── 05_mimii_realdata.ipynb   # real-data validation on MIMII pump dataset
├── run_pipeline.py        # end-to-end training script
├── predict.py             # single-signal inference
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

For GPU support (recommended), install PyTorch with CUDA:

```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Usage

Run the full synthetic pipeline:

```bash
python run_pipeline.py
```

Run inference on a single signal:

```bash
# Simulate a blocked pipe signal and classify it
python predict.py --pos 12.5 --severity 0.7 --snr 15

# Classify a saved waveform
python predict.py --signal path/to/signal.npy
```

Step through notebooks 01-05 for visualisations, analysis, and real-data validation.

## Key design decisions

- **Pink noise over white noise**: real pipe environments have 1/f noise profiles from flow and machinery vibration
- **Multipath echoes**: blockage echoes reflect off pipe ends and return as secondary arrivals
- **Class-weighted loss**: handles imbalanced datasets (both 3:1 synthetic and 8:1 MIMII)
- **Separate detector and localiser**: detection gates localisation — position is only estimated when a blockage is found
