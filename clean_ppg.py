import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, detrend

# Define a function to clean the signal
def clean_ppg_signal(signal, fs=125, lowcut=0.5, highcut=5.0):
    # Remove the DC component by detrending
    signal_detrended = detrend(signal)
    
    # Design the Butterworth band-pass filter
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype='band')
    
    # Apply the filter to the detrended signal
    signal_filtered = filtfilt(b, a, signal_detrended)
    
    return signal_filtered

# Function to remove the initial anomaly
def remove_initial_anomaly(signal, window_size=100):
    # Compute the median of the initial window
    initial_median = np.median(signal[:window_size])
    
    # Correct the signal by subtracting the initial median
    corrected_signal = signal - initial_median
    
    # Optional: Discard the first few samples if they are still noisy
    corrected_signal = corrected_signal[window_size:]
    
    return corrected_signal

# Function to trim or pad the signal to 60 seconds
def adjust_signal_length(signal, desired_length=7500):
    current_length = len(signal)
    
    if current_length > desired_length:
        # Trim the signal
        signal_adjusted = signal[:desired_length]
    else:
        # Pad the signal with zeros
        signal_adjusted = np.pad(signal, (0, desired_length - current_length), 'constant')
    
    return signal_adjusted

# Base directory containing all subject folders
base_dir = '/mnt/d/My_dataset2/'

# Find all subfolders
subject_folders = [f.path for f in os.scandir(base_dir) if f.is_dir()]

# Process each subject folder
for folder in subject_folders:
    # Path to the ground truth file
    ground_truth_path = os.path.join(folder, 'ground_truth.csv')
    
    # Check if the ground_truth.csv file exists
    if os.path.isfile(ground_truth_path):
        # Load the data
        data = pd.read_csv(ground_truth_path)
        green_channel = data.iloc[:, 1]
        
        # Clean the green channel
        green_cleaned = clean_ppg_signal(green_channel)
        
        # Remove the initial anomaly
        green_corrected = remove_initial_anomaly(green_cleaned)
        
        # Normalize the final PPG signal
        green_normalized = (green_corrected - np.min(green_corrected)) / (np.max(green_corrected) - np.min(green_corrected))
        
        # Adjust the signal length to 60 seconds (7500 samples)
        green_adjusted = adjust_signal_length(green_normalized, desired_length=7500)
        
        # Save the final output as a .txt file (one value per line, no delimiter)
        output_path = os.path.join(folder, 'ground_truth_cleaned.txt')
        np.savetxt(output_path, green_adjusted, fmt='%f')

print("Processing complete.")
