import os
import subprocess
from moviepy.editor import VideoFileClip

def resize_video(input_path, output_path, duration=60):
    """Resize the video to the specified duration in seconds."""
    clip = VideoFileClip(input_path)
    if clip.duration > duration:
        clip = clip.subclip(0, duration)
        clip.write_videofile(output_path, codec='libx264')
    else:
        # If the video is already shorter than or equal to the duration, just copy it
        if input_path != output_path:
            subprocess.run(['ffmpeg', '-i', input_path, '-c', 'copy', output_path])
    clip.close()

def check_and_resize_videos(root_dir, duration=60):
    """Check all subfolders for 'vid.mp4' and 'segmented_vid.mp4' and resize them if necessary."""
    for subdir, _, files in os.walk(root_dir):
        vid_path = os.path.join(subdir, "vid.mp4")
        segmented_vid_path = os.path.join(subdir, "segmented_vid.mp4")

        if os.path.exists(vid_path):
            resize_video(vid_path, vid_path, duration)

        if os.path.exists(segmented_vid_path):
            resize_video(segmented_vid_path, segmented_vid_path, duration)

if __name__ == "__main__":
    root_directory = "/mnt/d/My_dataset2"  # Change this to the root directory you want to check
    check_and_resize_videos(root_directory)
