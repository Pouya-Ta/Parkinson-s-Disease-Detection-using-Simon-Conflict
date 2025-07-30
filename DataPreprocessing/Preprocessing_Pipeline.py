import mne
import mne_bids
import matplotlib.pyplot as plt
import os
from glob import glob
from pathlib import Path

from mne_icalabel import label_components
from mne.preprocessing import ICA

# Configuration
SCRIPT_DIR = Path(__file__).parent
BIDS_ROOT = SCRIPT_DIR / "ds003509"
TASK_NAME = "SimonConflict"
N_JOBS = 8
DOWNSAMPLE_FREQ = 250

# Find all subject/session pairs in BIDS directory
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

# Main preprocessing function for one subject/session
def preprocess_continuous(subject_id, session_id):
    print(f"\n=== Preprocessing sub-{subject_id}, ses-{session_id or 'N/A'} ===")

    bids_path = mne_bids.BIDSPath(subject=subject_id,
                                  session=session_id,
                                  task=TASK_NAME,
                                  suffix='eeg',
                                  datatype='eeg',
                                  root=BIDS_ROOT)

    try:
        raw = mne_bids.read_raw_bids(bids_path=bids_path, verbose=False)
    except FileNotFoundError:
        print(f"  -> Original BIDS file not found. Skipping.")
        return
        
    raw.load_data()
    print(f"  -> Input file successfully read: {bids_path.fpath}")

    # Drop unused or problematic channels
    channels_to_exclude = ['FT9', 'FT10', 'TP9', 'TP10']
    raw.drop_channels(channels_to_exclude, on_missing='warn')
    print(f"  -> Attempted to drop channels: {channels_to_exclude}")

    accel_channels = ['X', 'Y', 'Z']
    channels_to_drop_accel = [ch for ch in accel_channels if ch in raw.ch_names]
    if channels_to_drop_accel:
        raw.drop_channels(channels_to_drop_accel)
        print(f"  -> Dropped accelerometer channels: {channels_to_drop_accel}")

    # Set channel types for EOG and EEG
    channel_types = {ch: ('eog' if 'VEOG' in ch else 'eeg') for ch in raw.ch_names}
    raw.set_channel_types(channel_types)

    # Apply standard electrode montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='raise')

    # Notch and bandpass filtering
    raw.notch_filter(freqs=60, picks=['eeg', 'eog'], n_jobs=N_JOBS)
    raw.filter(l_freq=0.5, h_freq=40.0, picks=['eeg', 'eog'], n_jobs=N_JOBS)

    # Set average reference
    raw.set_eeg_reference('average', projection=True)

    # ICA for artifact removal
    print("  -> Running ICA with InfoMax and ICLabel...")
    ica_raw = raw.copy().filter(l_freq=1.0, h_freq=None, n_jobs=N_JOBS)
    
    ica = ICA(n_components=0.99, method='infomax', max_iter='auto', random_state=42)
    ica.fit(ica_raw, picks='eeg')

    ic_labels = label_components(raw, ica, method='iclabel')
    labels = ic_labels['labels']
    
    labels_to_exclude = ['muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise']
    exclude_idx = [idx for idx, label in enumerate(labels) if label in labels_to_exclude]

    if exclude_idx:
        print(f"  -> Excluding {len(exclude_idx)} components identified by ICLabel.")
        ica.exclude = exclude_idx
    else:
        print("  -> ICLabel did not flag any components for exclusion.")

    ica.apply(raw)

    # Downsample to target frequency
    print(f"  -> Downsampling data to {DOWNSAMPLE_FREQ}Hz...")
    raw.resample(DOWNSAMPLE_FREQ, n_jobs=N_JOBS)

    # Save preprocessed data
    output_dir = Path(bids_path.directory)
    output_fname = output_dir / f"{bids_path.basename}.fif"
    print(f"  -> Saving preprocessed continuous data to: {output_fname}")
    raw.save(output_fname, overwrite=True)

# Run preprocessing for all subjects/sessions found
if __name__ == "__main__":
    subject_session_pairs = get_subject_session_pairs(BIDS_ROOT)
    
    for subject_id, session_id in subject_session_pairs:
        try:
            preprocess_continuous(subject_id, session_id)
        except Exception as e:
            print(f"⚠️ Skipped sub-{subject_id}, ses-{session_id or 'N/A'} due to a critical error:\n{e}")
