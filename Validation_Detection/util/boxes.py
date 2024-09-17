import torch

def decode_output(output, conf=0.5):
    # Assuming output is a tensor of shape [batch_size, num_predictions, num_features]
    # where the last dimension contains [x_center, y_center, width, height, obj_conf, class_logits]

    # Remove batch dimension (assuming batch_size=1)
    output = output[0] # Shape: [num_predictions, num_features]

    # Filter boxes based on objectness confidence
    boxes = output[output[:, 4] > conf]  # Shape: [num_filtered_boxes, num_features]

    # Separate class logits
    class_logits = boxes[:, 5:]  # Shape: [num_filtered_boxes, num_classes]

    # Get the class index with the highest probability for each box
    class_indices = class_logits.argmax(1)  # Shape: [num_filtered_boxes]

    # Get the class confidence using softmax, and extract the the highest probability
    class_confidence = class_logits.softmax(1).max(1).values  # Shape:[num_filtered_boxes]

    # Compute the final confidence by multiplying objectness confidence with class confidence
    final_confidence = boxes[:, 4] * class_confidence

    # Replace objectness confidence with the final confidence
    boxes[:, 4] = final_confidence # Shape: [num_filtered_boxes, num_features]

    # Add class indices (as the predicted class) to the boxes
    boxes[:, 5] = class_indices  # Shape: [num_filtered_boxes, num_features]

    # Truncate the boxes to retain only [x_center, y_center, width, height, confidence, class_index]
    boxes = boxes[:, :6] # Shape: [num_filtered_boxes, 6]

    # Convert from [x_center, y_center, width, height] to [x_min, y_min, x_max, y_max]
    boxes[:, :2] -= boxes[:, 2:4] / 2  
    boxes[:, 2:4] += boxes[:, :2]      

    # Return the boxes with shape [num_filtered_boxes, 6]
    # Each row represents [x_min, y_min, x_max, y_max, confidence, class_index]
    return boxes.cpu().numpy()



