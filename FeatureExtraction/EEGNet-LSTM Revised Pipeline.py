import os
import mne
import mne_bids
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

class Config:
    # Data stuff
    BIDS_ROOT = Path("ds003509")
    TASK_NAME = "SimonConflict"
    TMIN, TMAX = -0.2, 0.8
    BASELINE = (-0.2, 0)
    N_SPLITS = 5 # CV folds

    # EEGNet frontend
    KERN_LENGTH = 64
    F1 = 8
    D = 2
    F2 = 16
    EEG_DROPOUT = 0.25

    # LSTM backend
    LSTM_HIDDEN_1 = 128
    LSTM_HIDDEN_2 = 64
    LSTM_OUTPUT_DIM = 64 # final feature size
    LSTM_DROPOUT = 0.4

    # Training params
    LEARNING_RATE = 0.0005
    EPOCHS = 50
    PATIENCE = 15
    BATCH_SIZE = 16

    # Output files
    FINAL_FEATURE_FILENAME = "e2e_model_features.npy"
    FINAL_LABEL_FILENAME = "e2e_model_labels.npy"
    FINAL_GROUPS_FILENAME = "e2e_model_groups.npy"


class EEGNetLSTM(nn.Module):
    # Combined EEGNet-LSTM model.
    def __init__(self, n_classes, chans, samples, config):
        super().__init__()
        self.config = config
        self.Chans = chans
        self.Samples = samples

        # EEGNet part
        self.conv1 = nn.Conv2d(1, config.F1, (1, config.KERN_LENGTH), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(config.F1)
        self.depthwise_conv = nn.Conv2d(config.F1, config.F1 * config.D, (self.Chans, 1), groups=config.F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(config.F1 * config.D)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(config.EEG_DROPOUT)
        self.separable_conv = nn.Conv2d(config.F1 * config.D, config.F2, (1, 16), padding='same', bias=False)
        self.batchnorm3 = nn.BatchNorm2d(config.F2)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(config.EEG_DROPOUT)
        self.flatten = nn.Flatten()

        # Figure out the flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.Chans, self.Samples)
            eegnet_feature_size = self._get_eegnet_output_size(dummy_input)

        # LSTM part
        self.lstm1 = nn.LSTM(eegnet_feature_size, config.LSTM_HIDDEN_1 // 2, batch_first=True, bidirectional=True)
        self.norm1 = nn.LayerNorm(config.LSTM_HIDDEN_1)
        self.drop1 = nn.Dropout(config.LSTM_DROPOUT)
        self.lstm2 = nn.LSTM(config.LSTM_HIDDEN_1, config.LSTM_HIDDEN_2 // 2, batch_first=True, bidirectional=True)
        self.norm2 = nn.LayerNorm(config.LSTM_HIDDEN_2)
        self.drop2 = nn.Dropout(config.LSTM_DROPOUT)
        self.lstm3 = nn.LSTM(config.LSTM_HIDDEN_2, config.LSTM_OUTPUT_DIM // 2, batch_first=True, bidirectional=True)
        self.norm3 = nn.LayerNorm(config.LSTM_OUTPUT_DIM)
        self.drop3 = nn.Dropout(config.LSTM_DROPOUT)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.LSTM_OUTPUT_DIM, 32),
            nn.ReLU(),
            nn.Dropout(config.LSTM_DROPOUT),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def _get_eegnet_output_size(self, x):
        # helper to get the EEGNet output size
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise_conv(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        x = self.separable_conv(x)
        x = self.batchnorm3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        return x.shape[1]

    def extract_features(self, x):
        # Forward pass, but stops before the classifier to get features.
        # EEGNet part
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise_conv(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        x = self.separable_conv(x)
        x = self.batchnorm3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)

        # reshape for LSTM (batch, seq_len, features)
        x = x.unsqueeze(1)

        # LSTM part
        x, _ = self.lstm1(x)
        x = self.drop1(self.norm1(x))
        x, _ = self.lstm2(x)
        x = self.drop2(self.norm2(x))
        x, _ = self.lstm3(x)
        x = self.norm3(x)
        x = self.drop3(x) # dropout on features too

        # return last time step output
        return x[:, -1, :]

    def forward(self, x):
        # full pass for training
        features = self.extract_features(x)
        output = self.classifier(features)
        return output

def load_and_prepare_eeg_data(config):
    # Loads BIDS data and makes epochs.
    print("--- Loading and Preparing EEG Data ---")
    participants_path = config.BIDS_ROOT / "participants.tsv"
    if not participants_path.exists():
        raise FileNotFoundError(f"Can't find participants file: {participants_path}")
    participants_df = pd.read_csv(participants_path, sep='\t')

    search_pattern = str(config.BIDS_ROOT / "sub-*" / "ses-*" / "eeg" / f"sub-*_ses-*_task-{config.TASK_NAME}_eeg.fif")
    eeg_files = glob(search_pattern)
    if not eeg_files:
        raise FileNotFoundError(f"No EEG files found: {search_pattern}")

    all_X, all_y, all_groups = [], [], []
    for fpath in tqdm(eeg_files, desc="Processing Subjects"):
        try:
            bids_path = mne_bids.get_bids_path_from_fname(fpath)
            subject_id = bids_path.subject
            diagnosis = participants_df[participants_df['participant_id'] == f'sub-{subject_id}']['Group'].iloc[0]
            label = 1 if diagnosis == 'PD' else 0

            raw = mne.io.read_raw_fif(fpath, preload=True, verbose=False)
            events, _ = mne.events_from_annotations(raw, verbose=False)
            epochs = mne.Epochs(raw, events, event_id={'stimulus': 1},
                                tmin=config.TMIN, tmax=config.TMAX,
                                proj=True, picks='eeg', baseline=config.BASELINE,
                                preload=True, verbose=False)
            if len(epochs) == 0:
                print(f"\n[Warning] No epochs for {os.path.basename(fpath)}. Skipping.")
                continue

            all_X.append(epochs.get_data(picks='eeg'))
            all_y.extend([label] * len(epochs))
            all_groups.extend([int(subject_id)] * len(epochs))
        except Exception as e:
            print(f"\n[Error] Failed processing {os.path.basename(fpath)}: {e}")

    if not all_X:
         raise ValueError("Processing failed for all files. No data.")

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.array(all_y, dtype=np.int64)
    groups = np.array(all_groups)
    X = np.expand_dims(X, axis=1) # need this for Conv2D

    print(f"\nData ready. Shapes: X={X.shape}, y={y.shape}, groups={groups.shape}")
    return X, y, groups

def run_end_to_end_cross_validation(config, device):
    # Runs the whole CV pipeline.
    print("\n========== Starting End-to-End Cross-Validation Pipeline ==========")
    X, y, groups = load_and_prepare_eeg_data(config)

    sgkf = StratifiedGroupKFold(n_splits=config.N_SPLITS)
    n_chans, n_samples = X.shape[2], X.shape[3]
    n_classes = len(np.unique(y))

    # placeholders for final results
    final_features_oos = np.zeros((X.shape[0], config.LSTM_OUTPUT_DIM))
    final_labels_oos = np.zeros_like(y)
    final_groups_oos = np.zeros_like(groups)

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        print(f"\n--- Starting Fold {fold + 1}/{config.N_SPLITS} ---")

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

        # deal with imbalanced classes
        class_sample_counts = np.bincount(y_train)
        class_weights = 1. / class_sample_counts
        sample_weights = np.array([class_weights[t] for t in y_train])
        sampler = WeightedRandomSampler(torch.from_numpy(sample_weights).double(), len(sample_weights), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        model = EEGNetLSTM(n_classes=n_classes, chans=n_chans, samples=n_samples, config=config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        criterion = nn.BCELoss()

        best_val_metric, patience_counter, best_model_state = 0, 0, None
        for epoch in range(config.EPOCHS):
            model.train()
            total_train_loss = 0
            train_preds, train_labels = [], []
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device).float().unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                train_preds.extend(outputs.detach().cpu().round().numpy())
                train_labels.extend(batch_y.cpu().numpy())

            avg_train_loss = total_train_loss / len(train_loader)
            train_balanced_acc = balanced_accuracy_score(train_labels, train_preds)

            model.eval()
            total_val_loss = 0
            val_preds, val_labels = [], []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_y_float = batch_y.to(device).float().unsqueeze(1)
                    outputs = model(batch_X.to(device))
                    loss = criterion(outputs, batch_y_float)
                    total_val_loss += loss.item()
                    val_preds.extend(outputs.cpu().round().numpy())
                    val_labels.extend(batch_y.numpy())

            avg_val_loss = total_val_loss / len(val_loader)
            val_balanced_acc = balanced_accuracy_score(val_labels, val_preds)

            print(f"  Epoch {epoch+1:02d}/{config.EPOCHS} | "
                  f"Train Loss: {avg_train_loss:.4f}, Train Bal Acc: {train_balanced_acc:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f}, Val Bal Acc: {val_balanced_acc:.4f}")

            # early stopping
            if val_balanced_acc > best_val_metric:
                best_val_metric = val_balanced_acc
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= config.PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}. Best Val Balanced Acc: {best_val_metric:.4f}")
                    break

        # Extract features using the best model for this fold
        print("  Extracting out-of-sample features from test set...")
        model.load_state_dict(best_model_state if best_model_state else model.state_dict())
        model.eval()
        with torch.no_grad():
            final_features_fold = model.extract_features(torch.from_numpy(X_test).to(device)).cpu().numpy()

        # Store results
        final_features_oos[test_idx] = final_features_fold
        final_labels_oos[test_idx] = y_test
        final_groups_oos[test_idx] = groups[test_idx]

    print("\n--- End-to-End Cross-Validation Complete ---")
    return final_features_oos, final_labels_oos, final_groups_oos

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    final_features, final_labels, final_groups = run_end_to_end_cross_validation(config, device)

    print("\n========== Saving Final Results ==========")
    np.save(config.FINAL_FEATURE_FILENAME, final_features)
    np.save(config.FINAL_LABEL_FILENAME, final_labels)
    np.save(config.FINAL_GROUPS_FILENAME, final_groups)

    print(f"Saved features: {final_features.shape} to '{config.FINAL_FEATURE_FILENAME}'")
    print(f"Saved labels: {final_labels.shape} to '{config.FINAL_LABEL_FILENAME}'")
    print(f"Saved groups: {final_groups.shape} to '{config.FINAL_GROUPS_FILENAME}'")
    print("\n--- Done ---")


if __name__ == "__main__":
    main()
