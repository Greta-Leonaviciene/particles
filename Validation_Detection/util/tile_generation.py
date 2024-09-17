import os
import cv2

# Function that splits the image into tiles using a sliding window approach and returns a dictionary of tile information
def generate_tiles(folder_path, window_size=(320, 320), stride=220):

    # Supported image extensions
    supported_extensions = (".jpg", ".jpeg", ".png", ".tiff")

    # Dictionary to store tile information
    tile_info_dict = {}

    # Iterate through the images in the folder
    image_id = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_extensions):
            file_path = os.path.join(folder_path, filename)
            image = cv2.imread(file_path)

            # Get the shape of the original image
            height, width, _ = image.shape

            # Calculate the number of rows and columns for the sliding window
            nrows = (height - window_size[0]) // stride + 1
            ncols = (width - window_size[1]) // stride + 1

            # Adjust the strides to make sure the last tiles fit exactly within the image dimensions
            last_row_stride = height - window_size[0] - (nrows - 1) * stride if nrows > 1 else 0
            last_col_stride = width - window_size[1] - (ncols - 1) * stride if ncols > 1 else 0

            # Slide the window across the image
            tile_id = 0
            for i in range(nrows + 1):  # +1 to include the last partial tile
                for j in range(ncols + 1):  # +1 to include the last partial tile
                    if i == nrows:  # Adjust the last row
                        v_start = height - window_size[0]
                    else:
                        v_start = i * stride

                    if j == ncols:  # Adjust the last column
                        h_start = width - window_size[1]
                    else:
                        h_start = j * stride

                    h_end = h_start + window_size[1]
                    v_end = v_start + window_size[0]

                    # Crop the image using the calculated indices
                    cropped = image[v_start:v_end, h_start:h_end]

                    # Store the tile information in the dictionary
                    tile_key = f"image_{image_id}_tile_{tile_id}"
                    tile_info_dict[tile_key] = {
                        "tile": cropped  # The cropped tile as a numpy array
                    }

                    tile_id += 1
            image_id += 1

    return tile_info_dict


