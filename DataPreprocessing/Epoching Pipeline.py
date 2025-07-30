import mne
import mne_bids
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path
from scipy.stats import zscore

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
BIDS_ROOT = SCRIPT_DIR / "ds003509"
TASK_NAME = "SimonConflict"
PARTICIPANTS_FPATH = BIDS_ROOT / 'participants.tsv'

# Epoching parameters
TMIN = -0.2
TMAX = 0.8
BASELINE = (-0.2, 0)
REJECT_CRITERIA = dict(eeg=150e-6) # 150 µV

# Find all subject pairs in the BIDS directory
def get_subject_session_pairs(bids_root):
    subjects = glob(os.path.join(bids_root, "sub-*"))
    pairs = []
    for sub_path in subjects:
        subject_id = os.path.basename(sub_path).split("-")[1]
        sessions = glob(os.path.join(sub_path, "ses-*"))
        if sessions:
            for ses_path in sessions:
                session_id = os.path.basename(ses_path).split("-")[1]
                pairs.append((subject_id, session_id))
        else:
            pairs.append((subject_id, None))
    return pairs

# Get label for subject based on participants.tsv info
def get_label_for_session(subject_id, session_id, participants_df):
    participant_data = participants_df[participants_df['participant_id'] == f'sub-{subject_id}']
    if participant_data.empty:
        raise ValueError(f"Participant sub-{subject_id} not found in participants.tsv")
    participant_data = participant_data.iloc[0]
    group = participant_data['Group']

    if group == 'CTL':
        return 0, "CTL"
    elif group == 'PD':
        med_status = None
        if session_id == '01':
            med_status = participant_data['sess1_Med']
        elif session_id == '02':
            med_status = participant_data['sess2_Med']
        else:
            raise ValueError(f"Session '{session_id}' is not handled. Check participants.tsv columns.")

        if med_status == 'ON':
            return 1, "PD_ON"
        elif med_status == 'OFF':
            return 2, "PD_OFF"
        else:
            raise ValueError(f"Unknown medication status '{med_status}' for sub-{subject_id}, ses-{session_id}")
    else:
        raise ValueError(f"Unknown group '{group}' for sub-{subject_id}")

# Load data, create epochs, label, normalize, and save
def create_epochs(subject_id, session_id, participants_df):
    print(f"\n=== Epoching sub-{subject_id}, ses-{session_id or 'N/A'} ===")

    bids_path = mne_bids.BIDSPath(subject=subject_id,
                                  session=session_id,
                                  task=TASK_NAME,
                                  suffix='eeg',
                                  datatype='eeg',
                                  root=BIDS_ROOT)
    
    preprocessed_fpath = Path(bids_path.directory) / f"{bids_path.basename}.fif"

    if not preprocessed_fpath.exists():
        print(f"  -> Preprocessed file not found. Skipping. Expected at: {preprocessed_fpath}")
        return

    raw = mne.io.read_raw_fif(preprocessed_fpath, preload=True, verbose=False)
    print(f"  -> Input file successfully read: {preprocessed_fpath}")

    # Get label for this subject
    try:
        label, label_str = get_label_for_session(subject_id, session_id, participants_df)
        print(f"  -> Subject condition: {label_str}, Label: {label}")
    except ValueError as e:
        print(f"  -> ⚠️  Could not determine label. Skipping. Reason: {e}")
        return

    # Extract events and select only 'Test Stim' events
    print("  -> Epoching data...")
    events, event_id_map = mne.events_from_annotations(raw, verbose=False)
    stim_event_keys = {key for key in event_id_map.keys() if 'Test Stim' in key}
    if not stim_event_keys:
        print("  -> ⚠️ No 'Test Stim' events found. Skipping epoching for this subject.")
        return
    stim_event_id = {key: event_id_map[key] for key in stim_event_keys}

    # Create epochs for selected events
    epochs = mne.Epochs(raw, events, event_id=stim_event_id,
                          tmin=TMIN, tmax=TMAX,
                          proj=True, picks='eeg', baseline=BASELINE,
                          preload=True, verbose=True,
                          reject=REJECT_CRITERIA)
    
    # Store label in epochs metadata
    epochs.metadata = pd.DataFrame({'label': [label] * len(epochs)})

    # Z-score normalization across time for each epoch
    print("  -> Applying Z-score normalization...")
    epochs_data = epochs.get_data()
    zscored_data = zscore(epochs_data, axis=2)
    
    epochs_zscored = mne.EpochsArray(zscored_data, epochs.info, events=epochs.events, 
                                     tmin=epochs.tmin, event_id=epochs.event_id,
                                     metadata=epochs.metadata)

    # Save processed epochs
    output_dir = Path(bids_path.directory)
    output_fname = output_dir / f"{bids_path.basename}-epo.fif"
    print(f"  -> Saving epoched data to: {output_fname}")
    epochs_zscored.save(output_fname, overwrite=True)

# Run epoching for all subjectss
if __name__ == "__main__":
    if not PARTICIPANTS_FPATH.exists():
        raise FileNotFoundError(f"Participants file not found at: {PARTICIPANTS_FPATH}")
    
    participants_df = pd.read_csv(PARTICIPANTS_FPATH, sep='\t', engine='python')
    subject_session_pairs = get_subject_session_pairs(BIDS_ROOT)
    
    for subject_id, session_id in subject_session_pairs:
        try:
            create_epochs(subject_id, session_id, participants_df)
        except Exception as e:
            print(f"⚠️ Skipped sub-{subject_id}, ses-{session_id or 'N/A'} due to a critical error:\n{e}")
