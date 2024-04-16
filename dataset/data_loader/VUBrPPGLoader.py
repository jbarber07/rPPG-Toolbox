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

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For UBFC-rPPG dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "subject_*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = [{"index": re.search(
            'subject_(\d+)', data_dir).group(0), "path": data_dir} for data_dir in data_dirs]
        return dirs

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
    
    def augment_frames(self, frames):
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # Horizontally flip half of the frames
            iaa.Affine(
                rotate=(-10, 10)  # Mild rotation
            ),
            # iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)})  # Mild scaling
        ], random_order=True)  # Apply augmenters in random order

        if isinstance(frames, np.ndarray):
            frames_list = [frames[i] for i in range(frames.shape[0])]
        else:
            frames_list = frames

        augmented_frames = seq(images=frames_list)

        if not all(frame.ndim == 3 and frame.shape[2] == 3 for frame in augmented_frames):
            raise ValueError("Augmentation has produced frames with incorrect dimensions.")

        if isinstance(augmented_frames, list):
            augmented_frames = np.array(augmented_frames, dtype=np.uint8)

        return augmented_frames


    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        # Read and validate original frames
        original_frames = self.read_video(os.path.join(data_dirs[i]['path'], "vid.mp4"))
        if original_frames is None or len(original_frames) == 0:
            raise ValueError(f"Failed to read frames from {data_dirs[i]['path']}/vid.mp4")

        if not isinstance(original_frames, np.ndarray) or original_frames.ndim != 4 or original_frames.shape[-1] != 3:
            raise ValueError("Original frames must be a 4D numpy array with shape (num_frames, H, W, 3)")

        frames_to_process = [original_frames]  # Start with original frames

        # Apply augmentation if enabled and add to processing list
        if 'AUGMENT' in config_preprocess.DATA_AUG:
            augmented_frames = self.augment_frames(original_frames.copy())
            if any(frame.ndim != 3 or frame.shape[-1] != 3 for frame in augmented_frames):
                raise ValueError("Augmented frames have incorrect dimensions")
            frames_to_process.append(augmented_frames)  # Include augmented frames

        all_input_names = []

        # Read labels once, apply to both original and augmented frames
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(original_frames, fs=self.config_data.FS)
        else:
            bvps = self.read_wave(os.path.join(data_dirs[i]['path'], "ground_truth.xlsx"))

        # Process each set of frames (original and possibly augmented)
        for frames in frames_to_process:
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
            input_name_list, _ = self.save_multi_process(frames_clips, bvps_clips, saved_filename)

            all_input_names.extend(input_name_list)

        # Save processed input data path names to dictionary
        file_list_dict[i] = all_input_names

        return file_list_dict



    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, init_frame = VidObj.read()
        frames = list()
        while success:
            init_frame = init_frame[:, init_frame.shape[1]//2:] # remove the pulse oximeter from the video
            #converting from gbr to hsv color space
            img_HSV = cv2.cvtColor(init_frame, cv2.COLOR_BGR2HSV)
            #skin color range for hsv color space 
            HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
            HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

            #converting from gbr to YCbCr color space
            img_YCrCb = cv2.cvtColor(init_frame, cv2.COLOR_BGR2YCrCb)
            #skin color range for hsv color space 
            YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
            YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

            #merge skin detection (YCbCr and hsv)
            global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
            global_mask=cv2.medianBlur(global_mask,3)
            global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))

            masked_img = cv2.bitwise_and(init_frame, init_frame, mask=global_mask)
            masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
            frames.append(masked_img)
            success, init_frame = VidObj.read()
        return np.asarray(frames)
    
    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""

        # Read the xlsx file
        df = pd.read_excel(bvp_file)

        # Remove the first row (headers)
        df = df.iloc[1:]

        # Take the third column
        bvp = df.iloc[:, 2].values
        return np.asarray(bvp)
    
        
    
    
