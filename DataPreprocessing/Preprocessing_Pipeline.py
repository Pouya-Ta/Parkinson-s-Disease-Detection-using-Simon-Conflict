import mne
import mne_bids
import matplotlib.pyplot as plt
import os
from glob import glob
# Required for the artifact detection step
from mne_icalabel import label_components

bids_root = "ds003509_05"
task_name = "SimonConflict"

def get_subject_session_pairs(bids_root):
    """Find all subject/session pairs in the BIDS directory."""
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
            # Handle subjects without a session folder
            pairs.append((subject_id, None))
    return pairs

def preprocess(subject_id, session_id):
    """Run the preprocessing pipeline for a single subject/session."""
    print(f"\n=== Processing sub-{subject_id}, ses-{session_id or 'N/A'} ===")

    bids_path = mne_bids.BIDSPath(subject=subject_id,
                                  session=session_id,
                                  task=task_name,
                                  root=bids_root)

    raw = mne_bids.read_raw_bids(bids_path=bids_path, verbose=False)
    raw.load_data()

    # Drop accelerometer channels if present
    accel_channels = ['X', 'Y', 'Z']
    channels_to_drop = [ch for ch in accel_channels if ch in raw.ch_names]
    if channels_to_drop:
        raw.drop_channels(channels_to_drop)
        print(f"Dropped accelerometer channels: {channels_to_drop}")

    # Set channel types (EEG vs. EOG)
    channel_types = {ch: ('eog' if 'VEOG' in ch else 'eeg') for ch in raw.ch_names}
    raw.set_channel_types(channel_types)

    # Use a standard montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='warn')

    # Apply filters
    raw.notch_filter(freqs=60, picks=['eeg', 'eog'])
    raw.filter(l_freq=0.5, h_freq=40.0, picks=['eeg', 'eog'])

    # Re-reference to average
    raw.set_eeg_reference('average', projection=True)

    # Fit ICA on high-pass filtered data for better performance
    ica_raw = raw.copy().filter(l_freq=1.0, h_freq=None)
    ica = mne.preprocessing.ICA(n_components=15, max_iter='auto', random_state=715)
    ica.fit(ica_raw)
    
    # Automatically label components (brain, muscle, eye, etc.)
    print("Automatically labeling ICA components...")
    ic_labels = label_components(raw, ica, method='iclabel')
    labels = ic_labels['labels']
    
    # Find artifactual components to exclude
    labels_to_exclude = [
        'muscle artifact', 'eye blink', 'heart beat', 
        'line noise', 'channel noise'
    ]
    exclude_idx = [idx for idx, label in enumerate(labels) if label in labels_to_exclude]

    print(f"Found {len(exclude_idx)} components to exclude.")
    if exclude_idx:
        print(f"Excluding components with indices: {exclude_idx}")
        excluded_labels = [labels[i] for i in exclude_idx]
        print(f"Labels of excluded components: {excluded_labels}")
        ica.exclude = exclude_idx
    else:
        print("No components were flagged for exclusion.")

    # Apply ICA to remove artifacts
    ica.apply(raw)

    # Construct filename and save preprocessed data
    output_fname = f"sub-{subject_id}"
    if session_id:
        output_fname += f"_ses-{session_id}"
    output_fname += f"_task-{task_name}_preproc_raw.fif"

    output_path = os.path.join(bids_path.directory, output_fname)
    raw.save(output_path, overwrite=True)
    print(f"Saved preprocessed file: {output_path}")

    # Save a quality-check plot of the cleaned data
    fig_raw = raw.plot(duration=10, title=f"Cleaned Data sub-{subject_id} ses-{session_id or 'N/A'}", show=False)
    plot_path_raw = output_path.replace('.fif', '_plot.png')
    fig_raw.savefig(plot_path_raw)
    plt.close(fig_raw)
    print(f"Saved cleaned data plot: {plot_path_raw}")
    
    # Save a plot of the ICA component labels
    try:
        fig_labels = ic_labels.plot()
        plot_path_labels = output_path.replace('.fif', '_iclabels.png')
        fig_labels.savefig(plot_path_labels)
        plt.close(fig_labels)
        print(f"Saved ICA labels plot: {plot_path_labels}")
    except Exception as e:
        print(f"Could not save ICA labels plot: {e}")


if __name__ == "__main__":
    subject_session_pairs = get_subject_session_pairs(bids_root)
    for subject_id, session_id in subject_session_pairs:
        try:
            preprocess(subject_id, session_id)
        except Exception as e:
            print(f"⚠️ Skipped sub-{subject_id}, ses-{session_id or 'N/A'} due to error:\n{e}")
