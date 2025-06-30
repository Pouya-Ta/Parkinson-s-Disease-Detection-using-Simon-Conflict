# Preprocessing Pipeline for Simon Conflict EEG Data

This repository provides a fully automated preprocessing pipeline for EEG data collected during the Simon Conflict task, organized in [BIDS](https://bids.neuroimaging.io/) format (dataset: `ds003509`). The pipeline leverages [MNE-Python](https://mne.tools/) and related tools to standardize, filter, and clean raw EEG recordings, producing ready-to-analyze data and quality-control figures.

---

## Usage

Run the pipeline over all subjects and sessions:

```bash
python Preprocessing_Pipeline.py
```

Each subject/session pair will be processed, generating:

- `sub-<ID>[_ses-<ID>]_task-SimonConflict_preproc_raw.fif`: cleaned raw data

---

## Pipeline Overview

1. **BIDS Loading**: Read raw data via `mne_bids.read_raw_bids`.
2. **Channel Cleanup**:
   - Drop accelerometer channels (`X, Y, Z`).
   - Set types: EEG vs. EOG.
3. **Montage**: Apply the standard 10–20 montage for electrode locations.
4. **Filtering**:
   - Notch filter at 60 Hz to remove line noise.
   - Bandpass filter 0.5–40 Hz to isolate neural signals.
5. **Re-referencing**: Compute the average reference over all EEG channels.
6. **ICA Decomposition**:
   - Fit ICA on 1 Hz high‑passed data to separate sources.
   - Automatically label components (`iclabel`).
   - Exclude artifactual components (e.g., muscle, eye blinks).
   - Apply ICA correction to raw data.
7. **Saving Outputs**: Write cleaned FIF files and QC figures.

---

## Theory & Mathematics

### 1. Notch Filtering

A notch filter attenuates a narrow band around the line frequency \$\omega\_0\$ (e.g., 60 Hz). The continuous-time transfer function can be expressed as:

$$
H(s) = \frac{s^2 + \omega_0^2}{s^2 + \frac{\omega_0}{Q} s + \omega_0^2},
$$

where \$Q\$ is the quality factor controlling the notch bandwidth.

### 2. Bandpass Filtering

To isolate frequencies between \$\omega\_l\$ and \$\omega\_h\$, an ideal bandpass filter has frequency response:

$$
H(\omega) =
\begin{cases}
1, & \omega_l \leq \omega \leq \omega_h, \\
0, & \text{otherwise.}
\end{cases}
$$

Practically, MNE uses FIR/IIR filters with finite impulse responses defined by windowing or Butterworth designs.

### 3. Average Reference

Each channel voltage \$V\_i\$ is re-referenced to the mean over \$N\$ EEG electrodes:

$$
V_i^{\text{(ref)}} = V_i - \frac{1}{N} \sum_{j=1}^{N} V_j.
$$

This reduces global noise and volume conduction effects.

### 4. Independent Component Analysis (ICA)

ICA models the observed data \$\mathbf{X}(t)\$ as a linear mixture of statistically independent sources \$\mathbf{S}(t)\$:

$$
\mathbf{X} = A \, \mathbf{S},
$$

where \$A\$ (mixing matrix) is unknown. ICA estimates an unmixing matrix \$W = A^{-1}\$ such that:

$$
\mathbf{S} = W \, \mathbf{X},
$$

maximizing non-Gaussianity of components to isolate artifacts (e.g., eye blinks, muscle activity). Automated classification (ICLabel) assigns each component a label; artifactual ones are excluded from reconstruction.

---

## Dependencies


```
mne
mne-bids
mne-icalabel
matplotlib
```


