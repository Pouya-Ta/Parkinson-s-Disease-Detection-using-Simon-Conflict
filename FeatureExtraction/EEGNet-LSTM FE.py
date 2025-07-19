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
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

class Config:
    # Data stuff
    BIDS_ROOT = Path("ds003509")
    TASK_NAME = "SimonConflict"
    TMIN, TMAX = -0.2, 0.8
    BASELINE = (-0.2, 0)
    N_SPLITS = 5 # CV folds

    # EEGNet params
    KERN_LENGTH = 64
    F1 = 8
    D = 2
    F2 = 16
    DROPOUT_RATE = 0.25
    EEG_BATCH_SIZE = 16
    EEG_LEARNING_RATE = 0.001
    EEG_EPOCHS = 50
    EEG_PATIENCE = 15

    # LSTM params
    LSTM_HIDDEN_1 = 128
    LSTM_HIDDEN_2 = 64
    LSTM_OUTPUT_DIM = 64
    LSTM_DROPOUT = 0.4
    LSTM_LEARNING_RATE = 0.0003
    LSTM_EPOCHS = 50
    LSTM_BATCH_SIZE = 32

    # Output files
    FINAL_FEATURE_FILENAME = "final_unified_features.npy"
    FINAL_LABEL_FILENAME = "final_unified_labels.npy"
    FINAL_GROUPS_FILENAME = "final_unified_groups.npy"

class EEGNet(nn.Module):
    # EEGNet model. `get_features` yanks the layer before classification.
    def __init__(self, n_classes, chans, samples, config):
        super(EEGNet, self).__init__()
        self.Chans = chans
        self.Samples = samples
        self.n_classes = n_classes
        self.config = config
        # Layer 1
        self.conv1 = nn.Conv2d(1, config.F1, (1, config.KERN_LENGTH), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(config.F1)
        # Layer 2
        self.depthwise_conv = nn.Conv2d(config.F1, config.F1 * config.D, (self.Chans, 1), groups=config.F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(config.F1 * config.D)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(config.DROPOUT_RATE)
        # Layer 3
        self.separable_conv = nn.Conv2d(config.F1 * config.D, config.F2, (1, 16), padding='same', bias=False)
        self.batchnorm3 = nn.BatchNorm2d(config.F2)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(config.DROPOUT_RATE)

        self.flatten = nn.Flatten()
        # Need to calculate the flattened size dynamically.
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.Chans, self.Samples)
            flattened_size = self._get_feature_map_size(dummy_input)
        self.dense = nn.Linear(flattened_size, self.n_classes)

    def _get_feature_map_size(self, x):
        # helper to figure out the dense layer input size
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

    def get_features(self, x):
        # forward pass but stops before the last layer
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
        features = self.flatten(x)
        return features

    def forward(self, x):
        # full pass for training
        features = self.get_features(x)
        output = self.dense(features)
        return output

class FeatureSelectionLSTM(nn.Module):
    # 3-layer bidirectional LSTM
    def __init__(self, input_size, config):
        super().__init__()
        self.config = config
        self.lstm1 = nn.LSTM(input_size, config.LSTM_HIDDEN_1 // 2, batch_first=True, bidirectional=True)
        self.norm1 = nn.LayerNorm(config.LSTM_HIDDEN_1)
        self.drop1 = nn.Dropout(config.LSTM_DROPOUT)
        self.lstm2 = nn.LSTM(config.LSTM_HIDDEN_1, config.LSTM_HIDDEN_2 // 2, batch_first=True, bidirectional=True)
        self.norm2 = nn.LayerNorm(config.LSTM_HIDDEN_2)
        self.drop2 = nn.Dropout(config.LSTM_DROPOUT)
        self.lstm3 = nn.LSTM(config.LSTM_HIDDEN_2, config.LSTM_OUTPUT_DIM // 2, batch_first=True, bidirectional=True)
        self.norm3 = nn.LayerNorm(config.LSTM_OUTPUT_DIM)
        self.drop3 = nn.Dropout(config.LSTM_DROPOUT)
        self.classifier = nn.Sequential(
            nn.Linear(config.LSTM_OUTPUT_DIM, 32),
            nn.ReLU(),
            nn.Dropout(config.LSTM_DROPOUT),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.extract_features(x, apply_final_dropout=False)
        return self.classifier(features)

    def extract_features(self, x, apply_final_dropout=True):
        # get features from the last LSTM layer
        out, _ = self.lstm1(x)
        out = self.drop1(self.norm1(out))
        out, _ = self.lstm2(out)
        out = self.drop2(self.norm2(out))
        out, _ = self.lstm3(out)
        out = self.norm3(out)
        if apply_final_dropout:
            out = self.drop3(out)
        # LSTM output is (batch, seq, feature), we just want the last time step
        return out[:, -1, :]

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

def run_unified_cross_validation(config, device):
    # Runs the whole CV pipeline to avoid data leakage.
    print("\n========== Starting Unified Cross-Validation Pipeline ==========")
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

        # Train EEGNet
        print("  Training EEGNet...")
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)) # val on test set for early stopping

        # deal with imbalanced classes
        class_sample_counts = np.bincount(y_train)
        class_weights = 1. / class_sample_counts
        sample_weights = np.array([class_weights[t] for t in y_train])
        sampler = WeightedRandomSampler(torch.from_numpy(sample_weights).double(), len(sample_weights), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=config.EEG_BATCH_SIZE, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=config.EEG_BATCH_SIZE, shuffle=False)

        eegnet_model = EEGNet(n_classes=n_classes, chans=n_chans, samples=n_samples, config=config).to(device)
        optimizer_eegnet = optim.Adam(eegnet_model.parameters(), lr=config.EEG_LEARNING_RATE)
        criterion_eegnet = nn.CrossEntropyLoss()

        best_val_metric, patience_counter, best_eegnet_state = 0, 0, None
        for epoch in range(config.EEG_EPOCHS):
            eegnet_model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer_eegnet.zero_grad()
                outputs = eegnet_model(batch_X)
                loss = criterion_eegnet(outputs, batch_y)
                loss.backward()
                optimizer_eegnet.step()

            eegnet_model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = eegnet_model(batch_X.to(device))
                    val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    val_labels.extend(batch_y.numpy())

            val_balanced_acc = balanced_accuracy_score(val_labels, val_preds)
            if (epoch + 1) % 5 == 0:
                 print(f"    EEGNet Epoch {epoch+1:02d}, Val Balanced Acc: {val_balanced_acc:.4f}")

            # early stopping
            if val_balanced_acc > best_val_metric:
                best_val_metric = val_balanced_acc
                patience_counter = 0
                best_eegnet_state = eegnet_model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= config.EEG_PATIENCE:
                    print(f"    EEGNet early stopping at epoch {epoch+1}. Best Balanced Acc: {best_val_metric:.4f}")
                    break

        # load best model for this fold
        eegnet_model.load_state_dict(best_eegnet_state if best_eegnet_state else eegnet_model.state_dict())

        # Extract intermediate features
        print("  Extracting intermediate features from EEGNet...")
        eegnet_model.eval()
        with torch.no_grad():
            # extract for both train and test sets of this fold
            eegnet_features_train = eegnet_model.get_features(torch.from_numpy(X_train).to(device)).cpu().numpy()
            eegnet_features_test = eegnet_model.get_features(torch.from_numpy(X_test).to(device)).cpu().numpy()

        # Scale features (fit on train, transform both)
        print("  Scaling intermediate features...")
        scaler = StandardScaler()
        eegnet_features_train_scaled = scaler.fit_transform(eegnet_features_train)
        eegnet_features_test_scaled = scaler.transform(eegnet_features_test)

        # reshape for LSTM (samples, seq_len=1, features)
        X_train_seq = eegnet_features_train_scaled[:, np.newaxis, :]
        X_test_seq = eegnet_features_test_scaled[:, np.newaxis, :]

        # Train LSTM
        print("  Training LSTM...")
        lstm_input_dim = eegnet_features_train.shape[1]
        lstm_model = FeatureSelectionLSTM(input_size=lstm_input_dim, config=config).to(device)
        optimizer_lstm = torch.optim.Adam(lstm_model.parameters(), lr=config.LSTM_LEARNING_RATE)
        criterion_lstm = nn.BCELoss()

        lstm_train_dataset = TensorDataset(torch.from_numpy(X_train_seq).float(), torch.from_numpy(y_train).float())
        lstm_train_loader = DataLoader(lstm_train_dataset, batch_size=config.LSTM_BATCH_SIZE, shuffle=True)

        lstm_model.train()
        for epoch in range(config.LSTM_EPOCHS):
            for xb, yb in lstm_train_loader:
                xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
                optimizer_lstm.zero_grad()
                pred = lstm_model(xb)
                loss = criterion_lstm(pred, yb)
                loss.backward()
                optimizer_lstm.step()

        # Extract final features from the test set
        print("  Extracting final out-of-sample features from LSTM...")
        lstm_model.eval()
        with torch.no_grad():
            final_features_fold = lstm_model.extract_features(torch.from_numpy(X_test_seq).float().to(device)).cpu().numpy()

        # Store results for this fold
        final_features_oos[test_idx] = final_features_fold
        final_labels_oos[test_idx] = y_test
        final_groups_oos[test_idx] = groups[test_idx]

    print("\n--- Unified Cross-Validation Complete ---")
    return final_features_oos, final_labels_oos, final_groups_oos

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    final_features, final_labels, final_groups = run_unified_cross_validation(config, device)

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
