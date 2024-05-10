import cv2
import numpy as np

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
adjust_luminosity(r"/home/guourg5/rPPG-Toolbox/My_dataset/Backhand/subject_6/vid.mp4", [0.90,0.95, 1.00, 1.05, 1.10], r"output_lumin")

