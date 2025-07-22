# Parkinson’s Disease Detection using Simon Conflict

A hybrid deep learning framework that leverages EEG signals recorded during the Simon Conflict Task to distinguish Parkinson’s Disease (PD) patients from healthy controls with high accuracy.

## Table of Contents

* [Abstract](#abstract)
* [Motivation](#motivation)
* [Methodology](#methodology)

  * [Overview](#overview)
  * [Model Architecture](#model-architecture)
* [Data Description](#data-description)
* [Model Implementation](#model-implementation)

  * [Preprocessing](#preprocessing)
  * [Feature Extraction and Selection](#feature-extraction-and-selection)
  * [Classification](#classification)
* [Experimental Results](#experimental-results)
* [Usage Instructions](#usage-instructions)

  * [Installation](#installation)
  * [Running the Pipeline](#running-the-pipeline)
* [Visualizations](#visualizations)
* [Limitations and Future Directions](#limitations-and-future-directions)
* [Citation](#citation)
* [License](#license)

---

## Abstract

Parkinson’s Disease (PD) is a progressive neurodegenerative disorder characterized by motor and early cognitive control deficits. This project presents a hybrid deep learning framework that combines EEGNet for spatial filtering and bidirectional LSTM layers for temporal modeling of EEG signals recorded during the Simon Conflict Task. The proposed approach achieves up to 89.7% classification accuracy, demonstrating a significant improvement over traditional feature-based methods (p < 0.01).

## Motivation

Accurate and early detection of PD is crucial for intervention and management of disease progression. Cognitive control impairments often precede motor symptoms and can be detected via event-related potentials (ERPs). The Simon Conflict Task elicits robust mid-frontal theta and alpha responses associated with conflict monitoring, making it an ideal paradigm for PD detection. Automating the extraction of spatiotemporal EEG features with deep learning offers the potential to surpass handcrafted-feature approaches in sensitivity and specificity.

## Methodology

### Overview

1. **EEG Acquisition**: Fifty-six participants (28 PD patients and 28 healthy controls) performed 200 trials of the Simon Conflict Task. PD patients were recorded in both ON- and OFF-medication states.
2. **Signal Preprocessing**: Raw EEG data were band-pass filtered (0.1–40 Hz), common-average referenced, and cleaned via ICA with ICLabel for artifact rejection. One-second stimulus-locked epochs were extracted.
3. **Hybrid Feature Learning**: Spatial features were learned using EEGNet’s convolutional blocks, followed by three stacked bidirectional LSTM layers to capture temporal dependencies.
4. **Classification**: The spatiotemporal feature vectors were classified using support vector machines (SVM) and ensemble combinations with kNN and Naïve Bayes. Subject-wise cross-validation ensured no data leakage.

### Model Architecture

* **EEGNet**

  * **Temporal Convolution**: F1 = 8 kernels, batch normalization.
  * **Depthwise Convolution**: D = 2, ELU activation, average pooling, dropout (0.25).
  * **Separable Convolution**: F2 = 16, batch normalization, pooling, dropout, flatten.

* **Bidirectional LSTM**

  * **Layer 1**: 64 units per direction, batch normalization, dropout (0.4).
  * **Layer 2**: 32 units per direction, batch normalization, dropout (0.4).
  * **Layer 3**: 32 units per direction, batch normalization, dropout (0.4).
  * **Output**: Final hidden state (64-dimensional feature vector).

## Data Description

* **EEG System**: 64-channel, 10–20 montage.
* **Participants**: 28 PD patients (ON/OFF medication) and 28 age- and sex-matched controls.
* **Task**: Simon Conflict Task with color-based responses (yellow = left, blue = right) and spatial incongruence. Reward/punishment conditions varied probabilistically.

## Model Implementation

### Preprocessing

* Band-pass filter (0.1–40 Hz) using FIR design.
* Common-average referencing across all channels.
* ICA decomposition (15 components) with ICLabel for artifact removal.
* Epoch extraction: 1-second windows around stimulus onset.

### Feature Extraction and Selection

* **Spatial Filtering**: Convolutional layers of EEGNet learn frequency-specific spatial filters.
* **Temporal Modeling**: Bidirectional LSTMs capture time dependencies within each epoch.
* **Validation**: StratifiedGroupKFold (k = 5) ensures participant-level separation in cross-validation.

### Classification

| Classifier                    | Accuracy | Sensitivity | Specificity |
| ----------------------------- | -------: | ----------: | ----------: |
| EEGNet–LSTM + SVM             |    89.7% |       91.8% |       85.0% |
| EEGNet–LSTM + kNN+SVM         |   88.75% |      87.75% |      79.26% |
| EEGNet–LSTM + SVM+Naïve Bayes |   82.28% |      82.28% |      69.19% |

The SVM classifier on deep features significantly outperformed prior handcrafted-feature methods by 5–10% (p < 0.01).

## Experimental Results

Performance metrics indicate the hybrid framework’s capacity to detect PD-related cognitive control impairments with high reliability. Detailed statistical analyses and ROC curves are provided in the `results/` directory.

## Usage Instructions

### Installation

```bash
git clone https://github.com/Pouya-Ta/Parkinson-s-Disease-Detection-using-Simon-Conflict.git
cd Parkinson-s-Disease-Detection-using-Simon-Conflict
pip install -r requirements.txt
```

### Running the Pipeline

1. **Preprocess EEG Data**

   ```bash
   python DataPreprocessing/preprocess.py \
     --input data/raw/ \
     --output data/processed/
   ```
2. **Extract Features**

   ```bash
   python FeatureExtraction/extract_features.py \
     --input data/processed/ \
     --output features/epochs_features.npz
   ```
3. **Train and Evaluate Models**

   ```bash
   python MachineLearningModels/train_models.py \
     --features features/epochs_features.npz \
     --model_dir models/ \
     --results_dir results/
   ```

## Visualizations

The following figures are available in the `assets/` directory:

* **Preprocessing Results**: Raw vs. cleaned EEG traces.
* **Confusion Matrix**: SVM classifier.
* **ROC Curve**: Comparison across classifiers.

## Limitations and Future Directions

* **Sample Size**: Limited to 56 participants; larger cohorts are needed for generalization.
* **Medication Effects**: ON-/OFF-medication sessions could be modeled separately.
* **Task Diversity**: Additional cognitive paradigms and resting-state EEG should be explored.
* **Multimodal Integration**: Combining EEG with kinematic or imaging data may enhance detection accuracy.
* **Real-World Deployment**: Developing wearable EEG solutions for continuous monitoring.

## Citation

If you use this work, please cite:

```bibtex
@article{Taghipour2025,
  author = {Taghipour Langrodi, Pouya and Ahadzadeh, Mohammad and Khodadadi, Amirsadra and Rostami, Mostafa and Modaresi, Ali Sadat and Madadi, Sadegh},
  title = {Parkinson’s Disease Classification Using EEG and a Hybrid EEGNet–LSTM Architecture},
  year = {2025},
  institution = {Amirkabir University of Technology, BioMechatronic Lab},
  url = {https://github.com/Pouya-Ta/Parkinson-s-Disease-Detection-using-Simon-Conflict}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
