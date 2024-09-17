import os
import cv2
import numpy as np

def tiles_labels_to_dictionary(tile_dir=None, label_dir=None, image_ext=".png"):
    tiles_with_labels = {}
    for tile_name in os.listdir(tile_dir):
        if not tile_name.endswith(".png") and not tile_name.endswith(".jpg"):
            continue

        # Get the tile base name
        base_name = os.path.splitext(tile_name)[0]

        # Create the label file name
        label_name = base_name + ".txt"

        # Check if the labl file exists
        if not os.path.exists(os.path.join(label_dir, label_name)):
            continue

        # Create an empty dictionary for the image and labels
        tiles_with_labels[base_name] = dict()

        # Load the tile file and add to the dictionary
        tile = cv2.imread(os.path.join(tile_dir, tile_name))
        tiles_with_labels[base_name]["tile"] = tile

        # Load the labels and convert to list of lists
        with open(os.path.join(label_dir, label_name), "r") as f:
            label = f.read().strip()
            tiles_with_labels[base_name]["labels"] = [[float(x) for x in box.split()] for box in label.split('\n')]

    return tiles_with_labels
