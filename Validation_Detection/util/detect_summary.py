from collections import defaultdict
import os
import json
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display

# Process the JSON data to calculate class counts
def process_json(json_data, class_labels, print_info = True):
    # Output dict to populate with statistics
    stats = dict()
    stats["image_class_counts"] = dict()
    stats["prediction_class_counts"] = dict()

    # Class counts for predictions
    prediction_class_count = defaultdict(int)
    
    for image_id, data in json_data.items():
        predictions = data['predictions']
        
        # Class count for each image
        image_class_count = defaultdict(int)
        
        # Count total classes in predictions
        for pred in predictions:
            pred_class = pred[5]
            pred_class_name = class_labels[int(pred_class)]  # Map to class name
            image_class_count[pred_class_name] += 1
            prediction_class_count[pred_class_name] += 1
        
        # Add class counts for this image to stats
        stats["image_class_counts"][image_id] = {cls: image_class_count.get(cls, 0) for cls in class_labels}
        
        # Class counts for this image
        if print_info:
            print(f"Class counts for image {image_id}:")
            for cls in class_labels:
                print(f"{cls:<10} Predictions: {image_class_count.get(cls, 0):<5}")
    
    # Add the total count of classes in predictions across all images
    stats["prediction_class_counts"] = {cls: prediction_class_count.get(cls, 0) for cls in class_labels}

    # Print the total count of classes in predictions across all images
    if print_info:
        print("\nTotal Class counts in Predictions across all images:")
        print(f"{'Class':<10}{'Predictions':<10}")
        for cls in class_labels:
            print(f"{cls:<10}{prediction_class_count.get(cls, 0):<10}")
    
    return stats

# Function to combine image with statistics
def process_images_with_stats(output_dir, json_file, class_labels, font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", print_info=False):
    """
    Processes images by adding a white edge and overlaying statistics from a JSON file.

    Parameters:
    output_dir (str): The directory containing the images and results.json file.
    json_file (str): The path to the JSON file containing the image statistics.
    class_labels (list): The class labels used to process the JSON data.
    font_path (str): The path to the font used for the text overlay on images. Default is DejaVuSans-Bold.
    print_info (bool): Whether to print additional information during processing. Default is False.
    """
    # Load JSON data from the file
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    # Apply function that uses data in the JSON file to calculate validation metrics
    image_stats = process_json(json_data, class_labels, print_info=print_info)

    # Iterate over all image files in the directory
    for img_name in os.listdir(output_dir):
        if not img_name.endswith(".png") and not img_name.endswith(".jpg"):
            continue

        # Get the corresponding prediction data from the saved results
        image_id = int(img_name.split('_')[0])

        # If predictions are available, extend the image with a white edge and add statistics
        if "image_class_counts" in image_stats and str(image_id) in image_stats["image_class_counts"]:
            predictions = image_stats["image_class_counts"][str(image_id)]

            # Open the image
            image = Image.open(os.path.join(output_dir, img_name))

            # Add a white edge to the image
            width, height = image.size
            new_image = Image.new("RGB", (width + 200, height), "white")
            new_image.paste(image, (0, 0))

            # Add statistics to the image
            draw = ImageDraw.Draw(new_image)
            text = f"Image ID: {image_id}\n\n"
            text += f"Class statistics:\n"
            text += f"{'-' * 20}\n"
            for label, count in predictions.items():
                text += f"{label}: {count}\n"
            text += f"\n{'-' * 20}\n"
            text += f"Total: {sum(predictions.values())}\n"

            # Set the font size for the text overlay
            font = ImageFont.truetype(font_path, 20)

            # Draw the text on the image
            draw.text((width + 10, 10), text, fill="black", font=font)

            # Display the image inline (if in a notebook environment)
            display(new_image)



