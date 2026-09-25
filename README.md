# Heartfulness EEG Burst and Aperiodic Analysis

Analysis code for phase-resolved EEG burst dynamics and aperiodic spectral organization during structured Heartfulness meditation.

## Overview

This repository contains the analysis pipeline used for EEG preprocessing, feature extraction, and statistical analysis in the study of structured Heartfulness meditation.

The pipeline is organized into three main stages:

1. **Continuous EEG preprocessing**
2. **EEG burst and aperiodic feature extraction**
3. **Statistical and reviewer-sensitivity analyses**

Participant-level EEG recordings and derived participant-level datasets are **not included** in this repository.

---

## Study design

The analysis uses EEG recorded across seven ordered protocol phases:

| Phase | Description |
|---|---|
| B | Baseline |
| M1 | Meditation 1 |
| M2 | Meditation 2 |
| T1 | Transmission 1 |
| T2 | Transmission 2 |
| P1 | Post 1 |
| P2 | Post 2 |

Three participant groups are represented:

- **CG** — Control Group
- **STM** — Short-Term Meditators
- **LTM** — Long-Term Meditators

The final complete-case analytical sample contains **29 participants**.

EEG features are evaluated across:

- **5 scalp regions:** Frontal, Central, Temporal, Parietal, Occipital
- **4 frequency bands:** delta (1–4 Hz), theta (4–8 Hz), alpha (8–12 Hz), beta (13–30 Hz)

---

## Analysis pipeline

### 1. Preprocessing

`01_preprocessing.py`

The preprocessing pipeline:

1. Loads continuous EGI `.mff` recordings.
2. Removes the non-EEG reference channel (`Vertex Reference`) when present.
3. Applies a 0.3–40 Hz band-pass filter.
4. Performs ICA-based ocular-artifact correction.
5. Identifies bad channels using RANSAC.
6. Interpolates bad channels using spherical interpolation.
7. Applies a common-average reference.
8. Saves the continuous cleaned EEG as `.fif`.

The continuous recording is retained throughout preprocessing.

Temporary 2-s epochs are used only for RANSAC channel-quality assessment. They are not used for subsequent feature extraction and are not concatenated.

No 50-Hz notch filter is applied because the preprocessing low-pass cutoff is 40 Hz.

### 2. Burst and aperiodic feature extraction

`02_feature_extraction.py`

The feature-extraction pipeline uses non-overlapping 3-s windows for artifact/QC screening.

- Peak-to-peak rejection threshold: **100 µV**
- Clean windows remain separate for burst detection.
- Clean windows are not concatenated for burst detection.
- Bursts are therefore detected independently within each clean window.
- Minimum burst duration: **80 ms**
- Minimum envelope-peak separation: **80 ms**
- Hilbert-transform envelope with Gaussian smoothing.
- Burst threshold: **median + 3 × MAD**
- Thresholds are estimated once per **participant × region × band**, using data pooled across all seven phases.
- Threshold sensitivity analyses use 2 × MAD, 3 × MAD, and 4 × MAD thresholds.

The script extracts burst, peak, power-spike, band-power, and aperiodic measures.

### 3. Aperiodic spectral analysis

Aperiodic parameters are estimated using **specparam**.

The analysis:

- Fits the 2–30 Hz frequency range.
- Uses a fixed aperiodic component.
- Uses Gaussian periodic peaks.
- Allows up to four peaks.
- Uses the specparam aperiodic exponent as the sole aperiodic estimate.
- Converts the exponent to the manuscript's slope convention:

```text
Slope_1f = -exponent
```

No log-log linear-regression fallback is used.

Aperiodic offset, exponent, slope, R², fitting error, and fit status are retained for quality control.

Clean windows are concatenated **only for Welch PSD estimation used by the aperiodic analysis**. They are not concatenated for burst detection.

### 4. Statistical analysis

`03_statistical_analysis.py`

The statistical pipeline includes:

- Primary Phase × Group linear mixed-effects models
- ML likelihood-ratio tests for fixed-effect comparisons
- REML estimation for final model coefficients
- Participant-level cluster bootstrap sensitivity analyses
- Phase × Group × Band heterogeneity analysis
- Band-specific Phase × Group follow-ups
- Phase × Group × Region heterogeneity analysis
- Region-specific Phase × Group follow-ups
- Corrected aperiodic-slope analysis without duplicated band-level slope observations
- Model-consistent estimated marginal mean group contrasts
- Participant-level slope bootstrap
- Secondary four-state aggregation analysis
- Dependency-aware burst–aperiodic coupling
- Phase-specific adjusted slope effects
- Penalized GAM analysis
- Participant-level GAM bootstrap
- Threshold-sensitivity audit
- EEG quality-control summaries
- Dataset and subject-completeness audits

The primary mixed model is:

```text
Y ~ Phase × Group + Band + Region + (1 | Subject)
```

Likelihood-ratio comparisons are performed using ML. Final coefficient estimates are obtained using REML.

---

## Repository structure

```text
heartfulness-eeg-burst-aperiodic-analysis/
│
├── README.md
├── requirements.txt
│
├── 01_preprocessing.py
├── 02_feature_extraction.py
├── 03_statistical_analysis.py
│

```

The `config/`, `docs/`, `figures/`, and `results/` directories are intended to contain configuration and reproducibility documentation rather than participant-level EEG data.

---

## Data availability

Raw EEG recordings are not included in this repository.

Participant-level feature tables and statistical output files are also not distributed here unless explicitly approved for release.

To reproduce the analysis, researchers must have access to the required EEG data and place the files in the local data locations specified by their configuration.

---

## Reproducibility

The recommended execution order is:

```text
01_preprocessing.py
        ↓
02_feature_extraction.py
        ↓
03_statistical_analysis.py
```

The preprocessing script generates continuous cleaned `.fif` recordings.

The feature-extraction script uses these recordings to generate the detailed feature dataset and associated QC logs.

The statistical-analysis script starts from `EEG_Features_Detailed.csv` and performs the mixed-effects, sensitivity, aperiodic, coupling, GAM, and QC analyses.

### Important

The scripts currently contain local path definitions from the analysis environment in which the study was conducted. Before redistribution, these paths should be moved to the repository configuration file and replaced with user-defined local paths.

---

## Software

The analysis uses Python and scientific-computing packages including:

- MNE-Python
- NumPy
- pandas
- SciPy
- statsmodels
- patsy
- autoreject
- specparam
- tqdm
- matplotlib
- pyGAM (for the GAM analysis)

See `requirements.txt` for the package environment.

---

## Important methodological notes

### Non-overlapping QC windows

The feature-extraction pipeline uses non-overlapping 3-s windows:

```text
window 1 | window 2 | window 3 | ...
```

Rejected windows are removed, while retained windows remain separate for burst detection.

### Phase-independent burst thresholds

Burst thresholds are calculated across all seven phases for each participant, region, and frequency band. The threshold is therefore not recalculated separately for each meditation phase.

### Aperiodic slope

The reported 1/f slope follows:

```text
Slope_1f = -specparam exponent
```

More negative values correspond to a steeper negative spectral slope.

### GAM interpretation

The GAM uses an ordered phase index from 1 to 7. This represents the ordered experimental protocol rather than continuous physiological time. GAM results are therefore interpreted as smooth patterns across the ordered protocol phases, not as evidence of continuous temporal dynamics or causal phase-to-phase changes.

---

## License

The code license will be specified separately in the repository.

