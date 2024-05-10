import numpy as np
import matplotlib.pyplot as plt

def plot_ppg_signal(file_path):
    """
    Reads a PPG signal from a txt file and plots it.

    Args:
    file_path (str): The path to the .txt file containing the PPG signal data.
    """
    # Read the PPG data from the file
    ppg_data = np.loadtxt(file_path, delimiter=',')

    # Create a plot of the PPG data
    plt.figure(figsize=(10, 4))
    plt.plot(ppg_data, label='PPG Signal', color='red')
    plt.xlabel('Time (arbitrary units)')
    plt.ylabel('Amplitude')
    plt.title('Photoplethysmogram (PPG) Signal')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == '__main__':
    file_path = '/home/guourg5/rPPG-Toolbox/runs/exp/VUB-rPPG_SizeW72_SizeH72_ClipLength180_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceFalse_BackendHC_Large_boxFalse_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse/saved_test_outputs/subj_subject_13_ppg.txt'  # Replace with the path to your PPG file
    plot_ppg_signal(file_path)
