import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
def visualize_instance_mask(json_path, image_id):
    # Load COCO annotations
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create an empty mask of 250x250 with zeros (background)
    mask = np.zeros((250, 250), np.uint8)

    # Counter for instance value
    instance_value = 1

    # For each annotation related to the image_id, update the mask
    for annotation in data["annotations"]:
        if annotation["image_id"] == image_id:
            # Get the polygon
            polygon = np.array(annotation["segmentation"][0], np.int32).reshape((-1, 1, 2))

            # Fill the polygon area with the instance_value
            cv2.fillPoly(mask, [polygon], instance_value)

            # Increase the instance_value
            instance_value += 1

    # Display the mask
    plt.imshow(mask, cmap='tab20b')  # using 'tab20b' colormap to better distinguish instances
    plt.axis('off')
    plt.colorbar()
    plt.show()

# Provide path to the JSON
json_path = "coco/annotations/instances_val2017.json"

visualize_instance_mask(json_path, 1)