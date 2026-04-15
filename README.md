# Acoustic Pipe Inspection: Physics-Based Simulation and ML Pipeline for Non-Invasive Blockage Detection

## Overview

Underground pipes and conduits are subject to blockages from soil ingress, root intrusion, and structural deformation. Physical inspection requires excavation, which is costly and disruptive. This project builds a non-invasive acoustic inspection pipeline: transmit a pulse from one end of the pipe, record the reflected echoes, and use signal processing and deep learning to detect whether a blockage is present and estimate its position.

The pipeline is built entirely from a physics-based simulator, requiring no external dataset to train. It is then validated on real industrial acoustic recordings from the MIMII pump dataset to measure the simulation-to-reality gap.

---

## How It Works

```
Acoustic pulse  ->  Echo recording  ->  STFT spectrogram  ->  2D CNN (detect)
                                    ->  Raw waveform      ->  1D CNN (localise)
```

1. **Physics simulator** (`src/simulation.py`): models acoustic pulse propagation through a pipe of variable length and material. Blockages partially reflect the pulse; echo arrival time follows $t = 2x/v$ where $x$ is blockage distance and $v$ is speed of sound. Generates unlimited labelled synthetic data including pink (1/f) noise and multipath echoes.

2. **Signal processing** (`src/features.py`): STFT spectrograms (2D time-frequency images), z-score normalised waveforms, sliding-window envelope features, and matched filtering for echo detection.

3. **Blockage Detector** (`src/models.py`): 2D CNN trained on spectrograms. Binary classification: clear pipe vs. blocked. Uses Focal Loss (alpha=0.25, gamma=2.0) with inverse-frequency class weighting to handle imbalanced datasets.

4. **Blockage Localiser** (`src/models.py`): 1D CNN trained on raw waveforms. Regresses the normalised position of the primary blockage along the pipe. Trained with a physics consistency loss (lambda=0.1) that penalises predictions inconsistent with DSP-derived echo timing: $d = vt/2$.

5. **Real-data validation** (`05_mimii_realdata.ipynb`): the same CNN architecture is trained and evaluated on the MIMII industrial pump dataset (4205 recordings, 8.2:1 class imbalance) to test generalisation beyond the simulator.

6. **Unsupervised baseline** (`06_autoencoder.ipynb`): convolutional autoencoder trained on normal signals only, using reconstruction error as an anomaly score.

---

## Results

### Synthetic pipeline

| Task | Method | Result |
|------|--------|--------|
| Blockage detection | 2D CNN + Focal Loss | AUC **1.000**, accuracy **100%** |
| Blockage localisation | 1D CNN + physics loss (lambda=0.1) | MAE **1.28 m** on 30 m pipe |
| DSP baseline (detection) | Peak detection | F1 0.860 |
| DSP baseline (localisation) | Matched filter echo timing | MAE 3.89 m |
| CNN vs DSP (localisation) | | **67% improvement** |

### Real-data validation (MIMII pump dataset)

| Metric | Value |
|--------|-------|
| AUC | **0.854** |
| Abnormal recall | **81%** |
| Normal recall | 66% |
| Accuracy | 67.8% |

8-channel microphone averaging + Focal Loss (alpha=0.891) targeting the minority abnormal class.

### SNR robustness (localiser)

| SNR | DSP MAE (matched filter) | CNN MAE |
|-----|--------------------------|---------|
| 5 dB | 12.85 m | 12.09 m |
| 15 dB | 8.58 m | 7.62 m |
| 25 dB | 0.08 m | 2.74 m |

CNN outperforms matched-filter DSP below 20 dB, which is the realistic operating regime for buried pipes with soil attenuation.

### Unsupervised anomaly detection (autoencoder, no labels)

| Method | AUC |
|--------|-----|
| Convolutional autoencoder (reconstruction error) | 0.525 |
| Supervised CNN (Focal Loss, all channels) | **0.854** |

The autoencoder is trained on normal signals only and uses reconstruction error as an anomaly score. The near-random AUC (0.525) confirms that pump anomalies in MIMII are spectrally subtle and require labelled supervision to detect reliably. The supervised CNN gap (0.329 AUC points) quantifies the value of labelled data for this task.

---

## Project Structure

```
acoustic_pipe_inspection/
├── src/
│   ├── simulation.py          # physics-based signal generation (pink noise, multipath)
│   ├── features.py            # STFT spectrograms, envelope features, matched filtering
│   └── models.py              # BlockageDetector, BlockageLocaliser, FocalLoss, training loops
├── outputs/
│   └── models/
│       ├── detector.pth           # trained 2D CNN detector (synthetic)
│       ├── localiser.pth          # trained 1D CNN localiser
│       ├── detector_mimii.pth     # 2D CNN trained on MIMII pump data
│       └── autoencoder_mimii.pth  # convolutional autoencoder (normal-only training)
├── 01_simulation.ipynb        # physics walkthrough, signal generation
├── 02_signal_analysis.ipynb   # DSP baseline, matched filtering vs envelope detection
├── 03_detection.ipynb         # 2D CNN detector: training, confusion matrix, ROC
├── 04_localisation.ipynb      # 1D CNN localiser, physics consistency loss, SNR ablation
├── 05_mimii_realdata.ipynb    # supervised real-data validation (MIMII pump dataset)
├── 06_autoencoder.ipynb       # unsupervised autoencoder anomaly detection (no labels)
├── run_pipeline.py            # end-to-end training script
├── predict.py                 # single-signal inference (detect + localise)
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

GPU support (recommended, tested on RTX 4060):

```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

---

## Usage

Run the full pipeline (generate data, extract features, train both models):

```bash
python run_pipeline.py
```

Run inference on a single signal:

```bash
# Simulate a blocked pipe at 12.5 m and classify
python predict.py --pos 12.5 --severity 0.7 --snr 15

# Classify a saved waveform
python predict.py --signal path/to/signal.npy
```

Open notebooks 01-06 in order for full visualisations, analysis, and real-data validation.

---

## Tech Stack

Python 3.11 · PyTorch 2.x · NumPy · SciPy · scikit-learn · Matplotlib

---

## Limitations and Future Work

- **Simulation-to-reality gap**: the synthetic simulator uses simplified physics (1D wave propagation, pink noise). Real pipes have bends, joints, and complex reverb. Fine-tuning on actual pipe recordings would improve real-world performance.
- **Fixed speed of sound**: the simulator uses 343 m/s (air). Conduits filled with water or with soil-coupled walls will have different propagation characteristics.
- **Localiser scope**: the localiser currently predicts the nearest blockage position. Extending to multi-blockage localisation is a direct next step.
- **No hardware integration**: the pipeline assumes pre-recorded signals. Integration with a transmitter/receiver hardware system would be required for field deployment.
