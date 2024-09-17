import torch
import torchvision
import cv2
import numpy as np
import os
import json

##### Functions required to apply for the whole image reconstruction from tiles

# Function responsible to converting tile coordinates to image coordinates
# In this case, tile id is from 0 to 44 (45 tiles per image)
def tile_to_image_coordinates(tile_id, tile_shape=(320, 320), image_size=(1080, 1920, 3), stride=220):
    # get the number of rows and columns
    num_rows = image_size[0] // stride
    num_cols = image_size[1] // stride

    # Check if there are partial rows and columns
    rows_partial = image_size[0] % stride > 0
    cols_partial = image_size[1] % stride > 0

    if rows_partial:
        num_rows += 1
    if cols_partial:
        num_cols += 1

    # Determine row and column based on tile_id
    row = int(tile_id) // num_cols
    col = int(tile_id) % num_cols

    # Calculate the position of the tile
    v_start = row * stride
    h_start = col * stride

    # Check if the tile is at the edge
    if row == num_rows - 1 and rows_partial:
        v_start = image_size[0] - tile_shape[0]
    if col == num_cols - 1 and cols_partial:
        h_start =  image_size[1] - tile_shape[1]

    # Calculate the end coorinates
    v_end = v_start + tile_shape[0]
    h_end = h_start + tile_shape[1]

    return (v_start, h_start), (v_end, h_end)

# Function for image reconstruction from tiles (after updating coordinates)
def reconstruct_image_from_tiles(tiles_dict, image_size=(1080, 1920, 3), stride=220):
    # Create an empty image to reconstruct the original image
    reconstructed_image = np.zeros(image_size)

    for tile_id, data in tiles_dict.items():
        tile = data["tile"]

        # Get the shape
        tile_h, tile_w = tile.shape[:2]

        # get tile coordinates within image
        (v_start,h_start), (v_end,h_end) = tile_to_image_coordinates(tile_id, (tile_h, tile_w), image_size, stride)
        
        # Add the tile to the reconstructed image
        reconstructed_image[v_start:v_end, h_start:h_end] = tile
    
    return reconstructed_image

# Function for label/prediction reconstruction
def reconstruct_image_boxes_from_tiles(tiles_dict, image_size=(1080, 1920, 3), stride=220, predictions=True):
    # Create an empty list to store the reconstructed labels
    reconstructed_labels = []

    # Get the number of rows and columns
    num_rows = image_size[0] // stride
    num_cols = image_size[1] // stride

    # Check if there are partial rows and columns
    rows_partial = image_size[0] % stride > 0
    cols_partial = image_size[1] % stride > 0

    if rows_partial:
        num_rows += 1
    if cols_partial:
        num_cols += 1

    for tile_id, data in tiles_dict.items():
        tile = data["tile"]

        # Get the shape
        tile_h, tile_w = tile.shape[:2]

        # Get tile coordinates within image
        (v_start,h_start), (v_end,h_end) = tile_to_image_coordinates(tile_id, (tile_h, tile_w), image_size, stride)

        # Get the labels for the tile
        if predictions:
            tile_labels = data["predictions"]

            # Update the labels with the new position
            for label in tile_labels:
                # Get the class and coordinates
                x1, y1, x2, y2, conf, cls = label

                # Transform and add coordinates to the list
                reconstructed_labels.append([
                    h_start + x1,
                    v_start + y1,
                    h_start + x2,
                    v_start + y2,
                    conf,
                    cls
                ])

        else:
            tile_labels = data["labels"]

            # Update the labels with the new position
            for label in tile_labels:
                # Get the class and coordinates
                cls, x, y, w, h = label

                # Transform and add coordinates to the list
                reconstructed_labels.append([
                    h_start + (x - w/2) * tile_w,
                    v_start + (y - h/2) * tile_h,
                    h_start + (x + w/2) * tile_w,
                    v_start + (y + h/2) * tile_h,
                    1,
                    cls
                ])

    return reconstructed_labels

# Helper function to plot one bounding box
# def plot_one_box(reconstructed_label, reconstructed_image, color=None, label=None, line_thickness=2):
#     # Coordinates of the bounding box
#     x1, y1, x2, y2 = [int(i) for i in reconstructed_label]
#     # Draw rectangle
#     cv2.rectangle(reconstructed_image, (x1, y1), (x2, y2), color, thickness=line_thickness)
#     # Draw label if provided
#     if label:
#         font_scale = 0.7
#         font_thickness = 2
#         cv2.putText(reconstructed_image, label, (x1, y1 - 2), 0, font_scale, color, font_thickness, lineType=cv2.LINE_AA)

# def plot_one_box(reconstructed_label, reconstructed_image, color=None, label=None, line_thickness=1):
#     # Coordinates of the bounding box
#     x1, y1, x2, y2 = [int(i) for i in reconstructed_label]
    
#     # Draw rectangle (bounding box)
#     cv2.rectangle(reconstructed_image, (x1, y1), (x2, y2), color, thickness=line_thickness)
    
#     # Draw label with background if provided
#     if label:
#         font_scale = 0.6
#         font_thickness = 1
        
#         # Get the size of the label text
#         (text_width, text_height), baseline = cv2.getTextSize(label, fontFace=0, fontScale=font_scale, thickness=font_thickness)
#         baseline += 3  # Adjust baseline
        
#         # Set coordinates for label background
#         label_background_top_left = (x1, y1 - text_height - baseline)
#         label_background_bottom_right = (x1 + text_width, y1)
        
#         # Draw label background (same color as the bounding box)
#         cv2.rectangle(reconstructed_image, label_background_top_left, label_background_bottom_right, color, thickness=-1)  # Thickness -1 fills the rectangle
        
#         # Draw label text (in white)
#         cv2.putText(reconstructed_image, label, (x1, y1 - baseline), 0, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

def plot_one_box(reconstructed_label, reconstructed_image, color=None, label=None, line_thickness=1):
    # Coordinates of the bounding box
    x1, y1, x2, y2 = [int(i) for i in reconstructed_label]
    
    # Draw rectangle (bounding box)
    cv2.rectangle(reconstructed_image, (x1, y1), (x2, y2), color, thickness=line_thickness)
    
    # Draw label with background if provided
    if label:
        font_scale = 0.5
        font_thickness = 1
        
        # Get the size of the label text
        (text_width, text_height), baseline = cv2.getTextSize(label, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=font_thickness)
        baseline += 1  # Adjust baseline
        
        # Set some padding for the label box to make it look cleaner
        padding = 5
        label_background_top_left = (x1, y1 - text_height - baseline - padding)
        label_background_bottom_right = (x1 + text_width + 2 * padding, y1)
        
        # Draw label background (same color as the bounding box)
        cv2.rectangle(reconstructed_image, label_background_top_left, label_background_bottom_right, color, thickness=-1)
        
        # Draw label text (in white), with padding adjustments for positioning
        cv2.putText(reconstructed_image, label, (x1 + padding, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)


##### Main function to reconstruct the images and labels (if available)
def reconstruct_data(tiles_with_labels=None, image_size=(1080, 1920, 3), stride=220, class_colors = [], class_labels = [], output_dir = 'output'):
    # Group the tiles by image_id
    images = {}
    for base_name, data in tiles_with_labels.items():
        image_id = int(base_name.split('_')[1])
        tile_id = int(base_name.split('_')[3])
        if image_id not in images:
            images[image_id] = dict()
        images[image_id][tile_id] = data

        # Dictionary to store predictions and labels
        results = {}

    # Reconstruct the images from the tiles
    for image_id, tiles in images.items():
        reconstructed_image = reconstruct_image_from_tiles(tiles, image_size=(1080, 1920, 3), stride=220)
        reconstructed_preds = reconstruct_image_boxes_from_tiles(tiles, image_size=(1080, 1920, 3), stride=220, predictions=True)
        # print(reconstructed_preds)
        
        # Load the labels if available
        if "labels" in tiles[0]:
            reconstructed_labels = reconstruct_image_boxes_from_tiles(tiles, image_size=(1080, 1920, 3), stride=220, predictions=False)
            reconstructed_labels = sorted(reconstructed_labels, key=lambda x: x[4], reverse=True)
            reconstructed_labels = torch.tensor(reconstructed_labels, device='cuda')
            mask = torchvision.ops.nms(reconstructed_labels[:, :4], reconstructed_labels[:, 4], 0.5)  # NMS
            reconstructed_labels = reconstructed_labels[mask]  # limit detections


        # Perform non-maximum suppression on the predictions
        # print("Before NMS: ", len(reconstructed_preds))
        reconstructed_preds = sorted(reconstructed_preds, key=lambda x: x[4], reverse=True)
        reconstructed_preds = torch.tensor(reconstructed_preds, device='cuda')
        mask = torchvision.ops.nms(reconstructed_preds[:, :4], reconstructed_preds[:, 4], 0.5)  # NMS

        reconstructed_preds = reconstructed_preds[mask]  # limit detections
        # print("Detected objects after nms: ", len(reconstructed_preds))

        # Save predictions and labels to dictionary
        if "labels" in tiles[0]:
            results[image_id] = {
                'predictions': reconstructed_preds.tolist(),
                'labels': reconstructed_labels.tolist()
            }
        else:
            results[image_id] = {
                'predictions': reconstructed_preds.tolist(),
            }
            
        # Draw the prediction bounding boxes
        img_tmp = reconstructed_image.copy()
        for box in reconstructed_preds:
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            plot_one_box([x1, y1, x2, y2], img_tmp, color=class_colors[int(cls)], label=class_labels[int(cls)])
        cv2.imwrite(f'{output_dir}/{image_id}_pred.png', img_tmp)

        # Draw the ground truth bounding boxes
        if "labels" in tiles[0]:
            img_tmp = reconstructed_image.copy()
            for box in reconstructed_labels:
                x1, y1, x2, y2, conf, cls = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                plot_one_box([x1, y1, x2, y2], img_tmp, color=class_colors[int(cls)], label=class_labels[int(cls)])
            cv2.imwrite(f'{output_dir}/{image_id}_gt.png', img_tmp)

    # Save results (predictions and labels) to a JSON file
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved in {os.path.join(output_dir, 'results.json')}")

        
