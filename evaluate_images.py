import argparse
import logging
import os
import pathlib

import cv2
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from semantic_segmentation import models
from semantic_segmentation import load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Root directory containing subfolders with videos')
    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument('--model-type', type=str, choices=models, required=True, help='Type of model to use')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for segmentation')
    parser.add_argument('--save', action='store_true', help='Flag to save the output video')
    parser.add_argument('--display', action='store_true', help='Flag to display the video during processing')

    return parser.parse_args()


def _load_frame(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_width = (image.shape[1] // 32) * 32
    image_height = (image.shape[0] // 32) * 32
    image = image[:image_height, :image_width]
    return image


def process_video(video_path, model, fn_image_transform, device, args):
    logging.info(f'Evaluating video from {video_path}')
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logging.error(f'Error opening video file {video_path}')
        return

    # Set up video writer
    if args.save:
        output_path = video_path.parent / f'segmented_{video_path.name}'
        if output_path.exists():
            logging.info(f'Segmented video already exists: {output_path}')
            return
        logging.info(f'Saving segmented video to {output_path}')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use a widely supported codec
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    peak_memory_usage_before = torch.cuda.max_memory_allocated()  # Capture peak memory before inference starts
    index = 0

    # Check if display is available
    display_available = "DISPLAY" in os.environ

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        original_frame = frame.copy()  # Keep a copy of the original frame
        image = fn_image_transform(frame)

        with torch.no_grad():
            image = image.to(device).unsqueeze(0)
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()

            results = model(image)['out']
            results = torch.sigmoid(results)
            results = results > args.threshold

            end_time.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            inference_time = start_time.elapsed_time(end_time)  # Time in milliseconds

        # Create a mask
        mask = results.squeeze().cpu().numpy().astype('uint8') * 255  # Convert to uint8 format and scale to [0, 255]

        # Apply mask to original frame
        mask = cv2.resize(mask, (frame_width, frame_height))  # Ensure mask size matches frame size
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convert mask to BGR format
        masked_frame = cv2.bitwise_and(original_frame, mask)

        if args.save:
            out.write(masked_frame)

        if args.display and display_available:
            cv2.imshow('Segmented Mask', masked_frame)
            if cv2.waitKey(1) == ord('q'):
                logging.info('Exiting...')
                break

        index += 1

    cap.release()
    if args.save:
        out.release()
    if args.display and display_available:
        cv2.destroyAllWindows()

    logging.info(f'Finished processing video from {video_path}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Running inference on {device}')

    assert args.display or args.save

    logging.info(f'Loading {args.model_type} from {args.model}')
    model = torch.load(args.model, map_location=device)
    model = load_model(models[args.model_type], model)
    model.to(device).eval()

    fn_image_transform = transforms.Compose(
        [
            transforms.Lambda(lambda frame: _load_frame(frame)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    root_dir = pathlib.Path(args.root)
    video_files = list(root_dir.rglob('vid.mp4'))
    if not video_files:
        logging.info('No videos found to process.')

    for video_path in video_files:
        output_path = video_path.parent / f'segmented_{video_path.name}'
        if output_path.exists():
            logging.info(f'Segmented video already exists: {output_path}')
            continue
        logging.info(f'Found video: {video_path}')
        process_video(video_path, model, fn_image_transform, device, args)

    logging.info('Done.')
