function assign_channel_locations(set_file, channels_file, electrodes_file, output_file)
    % ---------------------------------------------------------------------
    % ASSIGN EEG CHANNEL LOCATIONS IN EEGLAB
    % 
    % Inputs:
    %   - set_file: Path to the EEG .set file
    %   - channels_file: Path to the channels.tsv file
    %   - electrodes_file: Path to the electrodes.tsv file
    %   - output_file: Name of the saved EEG file
    %
    % Outputs:
    %   - Updated EEG file with correct channel locations
    %
    % ---------------------------------------------------------------------
    clc; clearvars -except set_file channels_file electrodes_file output_file;
    
    % LOAD EEG DATASET
    fprintf('Loading EEG dataset: %s\n', set_file);
    EEG = pop_loadset('filename', set_file);
    EEG = eeg_checkset(EEG);

    % LOAD CHANNEL LABELS
    fprintf('Loading channel labels from: %s\n', channels_file);
    channels = readtable(channels_file, 'FileType', 'text');
    
    % Assign channel labels
    for i = 1:height(channels)
        EEG.chanlocs(i).labels = channels.name{i};
    end
    fprintf('Channel labels assigned.\n');

    % LOAD ELECTRODE COORDINATES
    fprintf('Loading electrode coordinates from: %s\n', electrodes_file);
    electrodes = readtable(electrodes_file, 'FileType', 'text');

    % Assign X, Y, Z coordinates
    for i = 1:height(electrodes)
        EEG.chanlocs(i).X = electrodes.x(i);
        EEG.chanlocs(i).Y = electrodes.y(i);
        EEG.chanlocs(i).Z = electrodes.z(i);
    end
    fprintf('Electrode coordinates assigned.\n');

