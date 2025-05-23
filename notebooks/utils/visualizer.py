import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualize_segmentation_annotations(image_path, txt_path, label_map):
    """
    Visualize the segmentation mask on a image deom the dataset.

    Args:
        image_path: str: Path to the image file
        txt_path: str: Path to the annotation file
        label_map: List[str]: List of the class labels

    Returns:
        numpy.ndarray: Processed OpenCV image
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    colors = plt.cm.rainbow(np.linspace(0, 1, len(label_map)))

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue

        class_id = int(parts[0])

        if class_id not in label_map:
            continue

        color = tuple(int(c * 255) for c in colors[class_id][:3])

        polygon_points = []
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                x = float(parts[i]) * w
                y = float(parts[i + 1]) * h
                polygon_points.append([x, y])

        if polygon_points:
            polygon_points = np.array(polygon_points, dtype=np.int32)

            x_coords = polygon_points[:, 0]
            y_coords = polygon_points[:, 1]
            x1, y1 = np.min(x_coords), np.min(y_coords)
            x2, y2 = np.max(x_coords), np.max(y_coords)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = label_map[class_id]
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image, (x1, y1 - 20), (x1 + label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            overlay = image.copy()
            cv2.fillPoly(overlay, [polygon_points], color)
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
            cv2.polylines(image, [polygon_points], True, color, 2)

    return image


def load_result_image(eval_dir, batch_name, image_type, convert_color=True):
    """
    Loads an image file from evaluation results and returns a processed OpenCV image.

    Args:
        eval_dir: str: Directory where evaluation results are stored
        batch_name: str: Name of the batch (e.g., 'val_batch0')
        image_type: str: Type of image ('labels' or 'pred')
        convert_color: bool: Whether to convert from BGR to RGB

    Returns:
        numpy.ndarray: Processed OpenCV image
    """
    image_path = os.path.join(eval_dir, f'{batch_name}_{image_type}.jpg')
    image = cv2.imread(str(image_path))

    if convert_color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image