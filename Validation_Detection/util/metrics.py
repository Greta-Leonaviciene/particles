import numpy as np
from collections import defaultdict

# Functions provided in this file use data saved in json format

# Helper function to calculate Intersection over Union (IoU)
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Compute the area of intersection
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # Compute the area of both the prediction and label boxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # Compute the IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Function to compute true positives, false positives, and false negatives
def compute_metrics(predictions, labels, iou_threshold=0.8):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    matched_labels = set()

    for pred in predictions:
        pred_box = pred[:4]
        pred_class = pred[5]  # Class is the last element in the prediction

        best_iou = 0
        best_label_idx = -1
        for i, label in enumerate(labels):
            label_box = label[:4]
            label_class = label[5]  # Class is the last element in the label
            
            # Compute IoU between predicted box and label box
            iou = calculate_iou(pred_box, label_box)
            
            # Check if IoU and class match
            if iou > best_iou and pred_class == label_class:
                best_iou = iou
                best_label_idx = i
        
        # Evaluate prediction based on IoU and class match
        if best_iou >= iou_threshold:
            true_positives += 1
            matched_labels.add(best_label_idx)
        else:
            false_positives += 1

    # False negatives are labels that are not matched by any prediction
    false_negatives = len(labels) - len(matched_labels)
    
    return true_positives, false_positives, false_negatives

# Processing all images with class names
def process_json(json_data, class_labels):
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    
    # Aggregate class counts for all images
    overall_label_class_count = defaultdict(int)
    overall_prediction_class_count = defaultdict(int)

    for image_id, data in json_data.items():
        predictions = data['predictions']
        labels = data['labels']
        
        # Reset class counts for each image
        label_class_count = defaultdict(int)
        prediction_class_count = defaultdict(int)

        # Count total classes in labels and predictions for the current image
        for label in labels:
            label_class = label[5]
            label_class_name = class_labels[int(label_class)]  # Map to class name
            label_class_count[label_class_name] += 1
        for pred in predictions:
            pred_class = pred[5]
            pred_class_name = class_labels[int(pred_class)]  # Map to class name
            prediction_class_count[pred_class_name] += 1

        # Compute true positives, false positives, and false negatives for the current image
        tp, fp, fn = compute_metrics(predictions, labels)
        
        # Print precision and recall for the current image
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"Image {image_id} - Precision: {precision:.4f}, Recall: {recall:.4f}")

        # Class counts for this image
        print(f"Class counts for image {image_id}:")
        for cls in class_labels:
            print(f"{cls:<10} Labels: {label_class_count.get(cls, 0):<5} Predictions: {prediction_class_count.get(cls, 0):<5}")
        separator = f"{'-' * 20}\n"
        print(separator)

    #     # Aggregate class counts across all images
    #     for cls in label_class_count:
    #         overall_label_class_count[cls] += label_class_count[cls]
    #     for cls in prediction_class_count:
    #         overall_prediction_class_count[cls] += prediction_class_count[cls]

    #     # Aggregate results for overall metrics
    #     total_true_positives += tp
    #     total_false_positives += fp
    #     total_false_negatives += fn

    # # Combined precision and recall across all images
    # combined_precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    # combined_recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0

    # print(f"\nCombined Precision: {combined_precision:.4f}, Combined Recall: {combined_recall:.4f}")
    
    # # Print the total count of classes in labels and predictions across all images
    # print("\nTotal Class counts in Labels and Predictions:")
    # print(f"{'Class':<10}{'Labels':<10}{'Predictions':<10}")
    # for cls in class_labels:
    #     print(f"{cls:<10}{overall_label_class_count.get(cls, 0):<10}{overall_prediction_class_count.get(cls, 0):<10}")