import cv2
import numpy as np
import os

def adjust_luminosity(video_path, factors, output_folder):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return
    
    # Retrieve video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #create folder if not exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Process each luminosity factor and save a new video
    for factor in factors:
        output_path = f"{output_folder}/video_luminosity_{factor}.mp4"
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), True)
        
        # Reset the capture to the beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Adjust luminosity frame by frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Adjust luminosity by multiplying the frame by the factor
            # Clip the values to ensure they remain within [0, 255]
            adjusted_frame = np.clip(frame * factor, 0, 255).astype(np.uint8)
            
            # Write the adjusted frame to the output video
            out.write(adjusted_frame)
        
        out.release()
    print("Luminosity adjustment complete in this folder: ", output_folder)
    cap.release()

# Uncomment the function call and adjust the parameters as needed for your specific video and output needs.
adjust_luminosity(r"/mnt/d/Backhand_last/subject_4/vid.mp4", [0.90,0.95, 1.00, 1.05, 1.10], r"/mnt/d/output_lumin")

# 4 and 6