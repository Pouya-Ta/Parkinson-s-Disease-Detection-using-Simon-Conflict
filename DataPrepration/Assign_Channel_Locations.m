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
    
