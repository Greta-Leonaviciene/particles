#! /usr/bin/env python3
import time, os, json
import torch
from torch import amp
import cv2
from util.tile_generation import generate_tiles
from util.boxes import decode_output
from util.reconstruction import reconstruct_data
from util.detect_summary import process_json
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=r"`torch.cuda.amp.autocast\(args\.\.\.\)` is deprecated")


millis = lambda: int(round(time.time() * 1000))
parser = argparse.ArgumentParser()

OUTPUT_DIR = 'detect_output'

# Class labels
CLASS_LABELS = ["Defective", "Speckled", "Beaded","Clear", "Unknown"]

# Class colors
CLASS_COLORS = [
    (56, 56, 255),   # Red (Class 0)
    (151, 157, 255), # Light Red (Class 1)
    (31, 112, 255),  # Orange (Class 2)
    (29, 178, 255),  # Yellow (Class 3)
    (100, 205, 50)   # Green (Class 4)
]

# !python3 detect.py --weights {YOLO_WEIGHTS} --img {IMAGE_PATH}
if "__main__" in __name__:
    # Get command line arguments
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--img_folder", type=str, required=True)
    args = parser.parse_args()

    # Delete output directory if it exists
    if os.path.exists(OUTPUT_DIR):
        os.system(f'rm -r {OUTPUT_DIR}')

    # Create output directory if it does not exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    ## Setup the YOLO model from saved weights
    model = torch.hub.load("ultralytics/yolov5", "custom", args.weights)

    # Start timing
    inference_time_start = millis()

    ## Split the image into tiles
    tile_collection = generate_tiles(args.img_folder, window_size=(320, 320), stride=220)

    # Run the model on the images
    for base_name, data in tile_collection.items():

        # Get the image
        tile = data["tile"]

        # Convert image to tensor, normalize and add batch dimension
        image_norm = torch.from_numpy(tile).cuda()
        image_norm = image_norm.float() / 255.0
        image_norm = image_norm.permute(2, 0, 1).unsqueeze(0)

        # Use autocast for mixed precision
        with amp.autocast('cuda'):
            output = model(image_norm)

        # Decode the output to get bounding boxes
        boxes = decode_output(output, conf = 0.8)
        
        # Add predcitions to the dictionary
        tile_collection[base_name]["predictions"] = boxes

    # Calculate the inference time
    inference_time_end = millis()
    print('Inference time per tile: %.1f ms'%((inference_time_end - inference_time_start) / len(tile_collection.keys())))
    print('Total inference time for all images: %.3f s'%( (inference_time_end - inference_time_start)/1000))

    # Reconstruct the images and labels
    reconstruct_data(tile_collection, image_size=(1080, 1920, 3), stride = 220, class_colors=CLASS_COLORS, class_labels=CLASS_LABELS, output_dir=OUTPUT_DIR)

