# Acoustic Pipe Inspection

Physics-based simulation and CNN pipeline for detecting and localising blockages in pipes using acoustic signals.

## What it does

1. Simulates acoustic wave propagation through a 30 m pipe with randomised blockage positions and severities
2. Extracts STFT spectrograms and normalised waveforms as features
3. Trains a 2D CNN to classify signals as clear or blocked
4. Trains a 1D CNN to estimate the blockage position along the pipe

## Results

| Model | Metric | Value |
|-------|--------|-------|
| 2D CNN Detector | Validation accuracy | 100% |
| 1D CNN Localiser | Validation MAE | 3.42 m on 30 m pipe |
| DSP Baseline | F1 Score | 0.860 |

## Project structure

```
acoustic_pipe_inspection/
├── src/
│   ├── simulation.py    # synthetic signal generation
│   ├── features.py      # STFT and normalisation
│   └── models.py        # CNN architectures and training loops
├── outputs/
│   └── models/
│       ├── detector.pth
│       └── localiser.pth
├── 01_simulation.ipynb
├── 02_signal_analysis.ipynb
├── 03_detection.ipynb
├── 04_localisation.ipynb
├── run_pipeline.py
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the full pipeline end to end:

```bash
python run_pipeline.py
```

Or step through the notebooks in order (01 to 04) for visualisations and analysis.
