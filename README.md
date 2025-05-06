# BoXYZ - Carton Pose Estimation & Segmentation 

Take home assignment

**author**: Fares Abawi

**date**: 05.05.2025

Download the reports for [Task 1](assets/reports/1_depth_map_processing.pdf) and [Task 2](assets/reports/2_box_segmentation_yolo_eval.pdf)

## Task 1 - Depth Map Processing

Run the Jupyter notebook [`notebooks/1_depth_map_processing.ipynb`](notebooks/1_depth_map_processing.ipynb) to process the depth maps.

```bash
INTERACTIVE_VISUALIZATION=true jupyter notebook notebooks/1_depth_map_processing.ipynb
```

**To Export the notebook as PDF**

```bash
#sudo apt-get install texlive-xetex texlive-fonts-recommended texlive-plain-generic pandoc
jupyter nbconvert --execute --to pdf notebooks/1_depth_map_processing.ipynb
mv notebooks/1_depth_map_processing.pdf assets/reports/1_depth_map_processing.pdf
```

**NOTE** Download the generated Jupyter PDF report [here](assets/reports/1_depth_map_processing.pdf)

## Task 2 - Box Segmentation

### Preprocess dataset 

First, run the Jupyter notebook `notebooks/2.1_download_preprocess_datasets.ipynb` 
to download and convert the datasets to an ultralytics-compatible format.

Set the `DS_LOCATION` to where the SCD dataset should be downloaded (it will be downloaded automatically).

```bash
DS_LOCATION='datasets/segment' jupyter notebook notebooks/2.1_download_preprocess_datasets.ipynb
```

### Train

Next, run the Jupyter notebook `notebooks/2.2_box_segmentation_yolo_train.ipynb` to train the YOLOv9 and YOLOv11 on
the SCD datset (OSCD followed by LSCD).

```bash
DS_LOCATION='datasets/segment' jupyter notebook notebooks/2.2_box_segmentation_yolo_train.ipynb
```

### Evaluate

Finally, run the Jupyter notebook `notebooks/2.3_box_segmentation_yolo_eval.ipynb` to evaluate the trained models,

This will evaluate the `train_...` prefixed trained models if specified with `MODEL_NAME_PREFIX="train_"`, otherwise, 
it will use the trained model checkpoints trained on the SCD carton box dataset.

```bash
MODEL_NAME_PREFIX='train_' DS_LOCATION='datasets/segment' jupyter notebook notebooks/2.3_box_segmentation_yolo_eval.ipynb
```

**To Export the notebook as PDF**

remove the `MODEL_NAME_PREFIX='train_'` if you would like to evaluate on the trained model checkpoints 

```bash
#sudo apt-get install texlive-xetex texlive-fonts-recommended texlive-plain-generic pandoc
MODEL_NAME_PREFIX='train_' DS_LOCATION='datasets/segment' jupyter nbconvert --execute --to pdf notebooks/2.3_box_segmentation_yolo_eval.ipynb
mv notebooks/2.3_box_segmentation_yolo_eval.pdf assets/reports/2_box_segmentation_yolo_eval.pdf
```

**NOTE** Download the generated Jupyter PDF report [here](assets/reports/2_box_segmentation_yolo_eval.pdf)
