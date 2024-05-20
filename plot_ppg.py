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
    file_path = '/home/guourg5/rPPG-Toolbox/predictions/subject_7.txt'  # Replace with the path to your PPG file
    plot_ppg_signal(file_path)
