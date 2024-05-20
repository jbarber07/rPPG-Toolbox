import os
import cv2

def create_video_from_frame(frame, output_path, duration=60, fps=30):
    try:
        height, width, layers = frame.shape
        size = (width, height)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, size)
        
        total_frames = duration * fps
        
        for _ in range(total_frames):
            out.write(frame)
        
        out.release()
        print(f"Video created successfully at {output_path}")
    except Exception as e:
        print(f"Error in creating video: {e}")

def capture_frame_from_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video file {video_path}")
            return None
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Error: Unable to capture frame from {video_path}")
            return None
        return frame
    except Exception as e:
        print(f"Error in capturing frame from video: {e}")
        return None

def add_fake_suffix(folder_name):
    parts = folder_name.split('_')
    if 'fake' not in parts:
        for i, part in enumerate(parts):
            if part.isdigit():
                parts.insert(i + 1, 'fake')
                break
    return '_'.join(parts)

def process_subject_folders(base_path, output_base_path):
    try:
        for folder in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith(".mp4"):
                        video_path = os.path.join(folder_path, file)
                        print(f"Processing video: {video_path}")
                        frame = capture_frame_from_video(video_path)
                        if frame is not None:
                            fake_folder_name = add_fake_suffix(folder)
                            output_folder_path = os.path.join(output_base_path, fake_folder_name)
                            os.makedirs(output_folder_path, exist_ok=True)
                            output_path = os.path.join(output_folder_path, "vid.mp4")
                            create_video_from_frame(frame, output_path)
    except Exception as e:
        print(f"Error in processing subject folders: {e}")

# Usage example
base_path = '/mnt/d/Backhand_last'  # Replace with the base path to your subject folders
output_base_path = '/mnt/d/fake_from_real'  # Replace with the path to your output folder
process_subject_folders(base_path, output_base_path)
