import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

def normalize_signal(signal):
    """Normalize the signal to have a maximum amplitude of 1."""
    return signal / np.max(np.abs(signal))

def plot_ppg_signal(file_path1, file_path2=None, label_sampling_rate=125, prediction_sampling_rate=None):
    """
    Reads one or two PPG signals from txt files, normalizes them, and plots them on the same plot if both are provided.
    The x-axis is in seconds.

    Args:
    file_path1 (str): The path to the first .txt file containing the PPG signal data.
    file_path2 (str, optional): The path to the second .txt file containing the PPG signal data. Default is None.
    label_sampling_rate (int, optional): The sampling rate of the label PPG sensor in Hz. Default is 125 Hz.
    prediction_sampling_rate (int, optional): The sampling rate of the prediction PPG sensor in Hz. If None, assumed to be the same as label_sampling_rate.
    """
    # Read and process the label PPG data from the first file
    ppg_data1 = np.loadtxt(file_path1, delimiter=',')
    ppg_data1 = np.diff(ppg_data1)  # First derivative
    ppg_data1 = np.abs(ppg_data1)   # Absolute value
    ppg_data1 = np.diff(ppg_data1)  # Second derivative
    ppg_data1 = normalize_signal(ppg_data1)  # Normalize

    # Generate the time axis in seconds for the label data
    time_axis1 = np.arange(len(ppg_data1)) / label_sampling_rate

    plt.figure(figsize=(10, 4))
    plt.plot(time_axis1, ppg_data1, label='Label', color='red')

    # If a second file path is provided, read and process the prediction PPG data
    if file_path2:
        ppg_data2 = np.loadtxt(file_path2, delimiter=',')
        ppg_data2 = np.diff(ppg_data2)  # First derivative
        ppg_data2 = np.abs(ppg_data2)   # Absolute value
        ppg_data2 = np.diff(ppg_data2)  # Second derivative

        if prediction_sampling_rate is None:
            prediction_sampling_rate = label_sampling_rate
        
        # Resample the prediction data to match the length of the label data
        ppg_data2 = resample(ppg_data2, len(ppg_data1))
        ppg_data2 = normalize_signal(ppg_data2)  # Normalize

        # Generate the time axis in seconds for the prediction data
        time_axis2 = np.arange(len(ppg_data2)) / label_sampling_rate
        
        plt.plot(time_axis2, ppg_data2, label='Prediction', color='blue')

    # Configure the plot
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized Amplitude')
    plt.title('Photoplethysmogram (PPG) Signal')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == '__main__':
    label = "/mnt/d/My_dataset2/subject_7/ground_truth_cleaned.txt"  # Replace with the path to your first PPG file
    predictions = '/home/guourg5/rPPG-Toolbox/predictions_physnet/saved_test_outputs/subj_subject_7_ppg.txt'  # Replace with the path to your second PPG file (optional)
    plot_ppg_signal(label, predictions, label_sampling_rate=125, prediction_sampling_rate=None)  # Adjust prediction_sampling_rate if needed
