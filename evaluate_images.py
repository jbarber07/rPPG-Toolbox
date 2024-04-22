import argparse
import logging
import os
import pathlib
import functools

import cv2
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from semantic_segmentation import models
from semantic_segmentation import load_model
from semantic_segmentation import draw_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)

    parser.add_argument('--model-type', type=str, choices=models, required=True)

    parser.add_argument('--threshold', type=float, default=0.5)

    parser.add_argument('--save', action='store_true')
    parser.add_argument('--display', action='store_true')

    return parser.parse_args()


def find_files(dir_path: pathlib.Path, file_exts):
    assert dir_path.exists()
    assert dir_path.is_dir()

    for file_ext in file_exts:
        yield from dir_path.rglob(f'*{file_ext}')


def _load_image(image_path: pathlib.Path):
    image = cv2.imread(str(image_path))
    assert image is not None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_width = (image.shape[1] // 32) * 32
    image_height = (image.shape[0] // 32) * 32

    image = image[:image_height, :image_width]
    return image


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'running inference on {device}')

    assert args.display or args.save

    logging.info(f'loading {args.model_type} from {args.model}')
    model = torch.load(args.model, map_location=device)
    model = load_model(models[args.model_type], model)
    model.to(device).eval()

    logging.info(f'evaluating images from {args.images}')
    image_dir = pathlib.Path(args.images)

    fn_image_transform = transforms.Compose(
        [
            transforms.Lambda(lambda image_path: _load_image(image_path)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    # TensorBoard setup
    if not os.path.exists('./logs/inference_metrics/SEGMENTATION'):
        os.makedirs('./logs/inference_metrics/SEGMENTATION')

    writer = SummaryWriter(log_dir='./logs/inference_metrics/SEGMENTATION', filename_suffix='SEGMENTATION')

    # Measure model size
    model_size = os.path.getsize(args.model) / (1024 * 1024)  # Size in Megabytes
    print(f"Model size: {model_size:.2f} MB")

    # Add model size to writer
    writer.add_scalar('Model Size (MB)', model_size)

    peak_memory_usage_before = torch.cuda.max_memory_allocated()  # Capture peak memory before inference starts
    index = 0
    for image_file in find_files(image_dir, ['.png', '.jpg', '.jpeg']):
        logging.info(f'segmenting {image_file} with threshold of {args.threshold}')

        image = fn_image_transform(image_file)

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
        # Logging to TensorBoard
        writer.add_scalar('Inference Time (ms)', inference_time, index)
        peak_memory_usage_after = torch.cuda.max_memory_allocated()  # Capture peak memory after inference
        peak_memory_usage = (peak_memory_usage_after - peak_memory_usage_before) / (1024 ** 2)  # Convert to MB
        writer.add_scalar('Peak Memory Usage (MB)', peak_memory_usage, index)
        index += 1
        for category, category_image, mask_image in draw_results(image[0], results[0], categories=model.categories):
            if args.save:
                output_dir = pathlib.Path('segmentation_results')
                output_dir.mkdir(exist_ok=True)
                output_name = output_dir / f'results_{category}_{image_file.name}'
                logging.info(f'writing output to {output_name}')
                # cv2.imwrite(str(output_name), category_image)
                mask_output_name = output_dir / f'mask_{category}_{image_file.name}'
                # cv2.imwrite(str(mask_output_name), mask_image)

            if args.display:
                cv2.imshow(category, category_image)
                cv2.imshow(f'mask_{category}', mask_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                

        if args.display:
            if cv2.waitKey(0) == ord('q'):
                logging.info('exiting...')
                exit()
    writer.close()
    print('Evaluation complete.')
