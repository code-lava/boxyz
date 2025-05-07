import os
import json
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm


def convert_coco_to_yolo_seg(coco_dir, output_dir, description='COCO-style carton dataset converted to YOLO format'):
    """
    Convert COCO-style dataset to YOLO segmentation (ultralytics) format.

    Args:
        coco_dir: str: Path to the COCO dataset directory
        output_dir: str: Path to output the YOLO dataset
        description: str: Description of the YOLO dataset
    """
    os.makedirs(output_dir, exist_ok=True)

    for split in ['train', 'valid', 'test']:
        if split == 'train':
            coco_split = 'train2017'
            anno_file = 'instances_train2017.json'
        elif split == 'valid':
            coco_split = 'val2017'
            anno_file = 'instances_val2017.json'
        else:
            if not os.path.exists(os.path.join(coco_dir, 'images', 'test2017')):
                continue
            coco_split = 'test2017'
            anno_file = 'instances_test2017.json'

        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

        anno_path = os.path.join(coco_dir, 'annotations', anno_file)
        if not os.path.exists(anno_path):
            print(f"Warning: {anno_path} not found, skipping {split} split")
            continue

        print(f"Processing {split} split...")
        with open(anno_path, 'r') as f:
            coco_data = json.load(f)

        images = {img['id']: img for img in coco_data['images']}

        print(f"Organizing annotations for {split}...")
        image_annotations = {}
        for ann in tqdm(coco_data['annotations'], desc="Processing annotations"):
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(ann)

        print(f"Converting images and labels for {split}...")
        for img_id in tqdm(images.keys(), desc=f"Processing {split} images"):
            img_info = images[img_id]
            src_img_path = os.path.join(coco_dir, 'images', coco_split, img_info['file_name'])
            dst_img_path = os.path.join(output_dir, split, 'images', img_info['file_name'])

            if os.path.exists(src_img_path):
                shutil.copy(src_img_path, dst_img_path)
            else:
                print(f"Warning: Image {src_img_path} not found")
                continue

            if img_id in image_annotations:
                label_file = os.path.join(output_dir, split, 'labels',
                                          os.path.splitext(img_info['file_name'])[0] + '.txt')

                with open(label_file, 'w') as f:
                    for ann in image_annotations[img_id]:
                        category_id = ann['category_id'] - 1
                        iscrowd = ann['iscrowd']  # TODO (fabawi): This was part of the ultralytics annotation in the past. Not anymore? Check!
                        if len(ann['segmentation']) > 0 and len(ann['segmentation'][0]) > 0:
                            points = ann['segmentation'][0]
                            width, height = img_info['width'], img_info['height']
                            normalized_points = []

                            for i in range(0, len(points), 2):
                                if i + 1 < len(points):
                                    x, y = points[i], points[i + 1]
                                    normalized_points.append(f"{x / width:.18f}")
                                    normalized_points.append(f"{y / height:.18f}")

                            if normalized_points:
                                f.write(f"{category_id} {' '.join(normalized_points)}\n")

    create_data_yaml(output_dir, coco_data, description)
    print(f"Conversion complete. Output saved to {output_dir}")


def create_data_yaml(output_dir, coco_data, description='COCO-style carton dataset converted to YOLO format'):
    """
    Create data.yaml file for the dataset.

   Args:
        output_dir: str: Path to output the YOLO dataset
        coco_data: dict: Dict of COCO box dataset
        description: str: Description of the YOLO dataset
    """
    categories = {cat['id'] - 1: cat['name'] for cat in coco_data.get('categories', [])}

    train_count = len(os.listdir(os.path.join(output_dir, 'train', 'images'))) if os.path.exists(
        os.path.join(output_dir, 'train', 'images')) else 0
    val_count = len(os.listdir(os.path.join(output_dir, 'valid', 'images'))) if os.path.exists(
        os.path.join(output_dir, 'valid', 'images')) else 0
    test_count = len(os.listdir(os.path.join(output_dir, 'test', 'images'))) if os.path.exists(
        os.path.join(output_dir, 'test', 'images')) else 0

    data = {
        # WARNING: path is absolute because utralytics will assume any dataset is in the datasets folder relative to training script.
        # Change the `path` manually in `data.yaml` if you relocate the dataset
        'path': f'{os.path.abspath(output_dir)}',
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images' if test_count > 0 else '',
        'names': categories
    }

    print("Creating data.yaml file...")
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(f"# {description}\n")
        f.write('# Example usage: yolo train data=data.yaml\n\n')

        f.write('# Train/val/test sets\n')
        f.write(f"path: {data['path']}  # dataset root dir\n")
        f.write(f"train: {data['train']}  # train images (relative to 'path') [DEBUG:{train_count} images]\n")
        f.write(f"val: {data['val']}  # val images (relative to 'path') [DEBUG:{val_count} images]\n")
        # TODO (fabawi): no test set in SCD, try to get a test set from somewhere. Leave it for now
        if test_count > 0:
            f.write(f"test: {data['test']}  # test images (relative to 'path') [DEBUG:{test_count} images]\n\n")
        else:
            f.write("\n")

        f.write('# Classes\n')
        f.write('names:\n')
        for class_id, class_name in sorted(categories.items()):
            f.write(f"  {class_id}: {class_name}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert COCO-style dataset to YOLO segmentation format')
    parser.add_argument('--coco-dir', type=str, required=True, help='Path to COCO dataset directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to output YOLO dataset')

    args = parser.parse_args()

    convert_coco_to_yolo_seg(args.coco_dir, args.output_dir)