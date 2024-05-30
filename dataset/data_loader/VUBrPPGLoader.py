import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
import pandas as pd
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
import imgaug.augmenters as iaa


class VUBrPPGLoader(BaseLoader):
    """The data loader for the VUB-rPPG dataset."""

    def __init__(self, name, data_path, config_data, frames_save_path="/home/jerba/rPPG-Toolbox/ubfc/PreprocessedData/frames/"):
        """Initializes an UBFC-rPPG dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- subject_1/
                     |       |-- vid.mp4
                     |       |-- ground_truth.xlsx
                     |   |-- subject_2/
                     |       |-- vid.mp4
                     |       |-- ground_truth.xlsx
                     |...
                     |   |-- subject_n/
                     |       |-- vid.mp4
                     |       |-- ground_truth.xlsx
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)
        self.data_path = data_path


    def get_raw_data(self, data_path):
        """Returns data directories under the path(For UBFC-rPPG dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "subject_*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = [{"index": re.search(
            'subject_(\d+)*', data_dir).group(0), "path": data_dir} for data_dir in data_dirs]
        return dirs
    def is_original_folder_name(self,folder_name):
        # Split folder name by '_'
        parts = folder_name.split('_')
        # Check if the folder name has '_n_1' pattern
        if len(parts) > 2:
            return False  # Augmented folder name
        else:
            return True  # Original folder name
        
    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new
    

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process."""
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        # Read Frames
        if 'None' in config_preprocess.DATA_AUG:
            # Utilize dataset-specific function to read video
            frames = self.read_video(
                os.path.join(data_dirs[i]['path'],"segmented_vid.mp4"))
        elif 'Motion' in config_preprocess.DATA_AUG:
            # Utilize general function to read video in .npy format
            frames = self.read_npy_video(
                glob.glob(os.path.join(data_dirs[i]['path'],'*.npy')))
        else:
            raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')
        if len(frames) == 0:
            raise ValueError(f"Failed to read frames from {data_dirs[i]['path']}/segmented_vid.mp4")
        
        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            # check if the folder name contains 'fake'
            if 'fake' in filename:
                print(f"Generating null psuedo labels for {filename}")
                bvps = self.generate_null_psuedo_labels(frames, fs=self.config_data.FS)
            else:
                if self.is_original_folder_name(filename):
                    print(f"Generating psuedo labels for {filename} using the same folder")
                    bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
                else:
                    # get the original folder name
                    original_folder_name = '_'.join(filename.split('_')[:-1])
                    #check if still need to remove a part. example: subject_19_inc_lum
                    if not self.is_original_folder_name(original_folder_name):
                        original_folder_name = '_'.join(original_folder_name.split('_')[:-1])

                    # get the data path
                    data_path = os.path.split(data_dirs[i]['path'])[0] 
                    # print(f"Data path: {data_path}")               
                    # get the original folder path
                    original_folder_path = os.path.join(data_path, original_folder_name)
                    # print(f"Original folder path: {original_folder_path}")
                    # get the original folder frames
                    original_frames = self.read_video(os.path.join(original_folder_path, "segmented_vid.mp4"))
                    # generate psuedo labels
                    bvps = self.generate_pos_psuedo_labels(original_frames, fs=self.config_data.FS)
                    print(f"Augmented folder: {filename} using original folder: {original_folder_name}")
        else:
            if 'fake' in filename:
                # generate null labels
                print(f"Generating null labels for {filename}")
                bvps = self.generate_null_psuedo_labels(frames, fs=self.config_data.FS)
            else:
                if self.is_original_folder_name(filename):
                    subject_number = '_'.join(filename.split('_')[1:])
                    print(f"Reading labels from subject_{subject_number}")
                    bvps = self.read_wave(
                        os.path.join(data_dirs[i]['path'],f"ground_truth_preprocessed.txt"))
                else:
                    # get the original folder name
                    original_folder_name = '_'.join(filename.split('_')[:-1])
                    #check if still need to remove a part. example: subject_19_inc_lum
                    if not self.is_original_folder_name(original_folder_name):
                        original_folder_name = '_'.join(original_folder_name.split('_')[:-1])

                    # get the data path
                    data_path = os.path.split(data_dirs[i]['path'])[0] 
                    # get the original folder path
                    original_folder_path = os.path.join(data_path, original_folder_name)
                    # get labels
                    subject_number = '_'.join(original_folder_name.split('_')[1:])
                    bvps = self.read_wave(os.path.join(original_folder_path,f"ground_truth_preprocessed.txt"))
                    print(f"Augmented folder: {filename} using original folder: {subject_number}")
        
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, filename)
        file_list_dict[i] = input_name_list
        return file_list_dict


    @staticmethod
    def read_video(video_file): 
        """Reads a video file, returns frames(T, H, W, 3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            # frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            # frame = frame[:, frame.shape[1]//2:] # remove the pulse oximeter from the video
            frames.append(frame)
            success, frame = VidObj.read()
        return np.asarray(frames)
    
    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        bvp = np.loadtxt(bvp_file)
        return np.asarray(bvp)
        # # Read the xlsx file
        # df = pd.read_excel(bvp_file)

        # # Remove the first row (headers)
        # df = df.iloc[1:]

        # # Take the third column
        # bvp = df.iloc[:, 2].values
        # return np.asarray(bvp)
    
        
    
    
