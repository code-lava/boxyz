# BoXYZ - Carton Pose Estimation & Segmentation 


**author**: Fares Abawi

**date**: 05.05.2025

View the reports for [Task 1](assets/reports/1_depth_map_processing.md) and [Task 2](assets/reports/2_box_segmentation_yolo_eval.md)

## Requirements

All requirements are installed in their corresponding notebook. Models were trained on RTX 3080 Ti. 
Versions:
- Python 3.10
- PyTorch 2.6
- Ultralytics 8.3.123
- Open3D 0.19.0
- NumPy 2.2.5
- OpenCV 4.10.0

## Task 1 - Depth Map Processing

Run the Jupyter notebook [`notebooks/1_depth_map_processing.ipynb`](notebooks/1_depth_map_processing.ipynb) to process the depth maps.

```bash
INTERACTIVE_VISUALIZATION=true jupyter notebook notebooks/1_depth_map_processing.ipynb
```

**To Execute and Export the Notebook**

```bash
jupyter nbconvert --execute --to markdown notebooks/1_depth_map_processing.ipynb
```

**NOTE** Report can be found [here](assets/reports/1_depth_map_processing.md)

## Task 2 - Box Segmentation

### Preprocess dataset

First, run the Jupyter notebook `notebooks/2.1_download_preprocess_datasets.ipynb` 
to download and convert the datasets to an ultralytics-compatible format.

Set the `DS_LOCATION` to where the SCD dataset should be downloaded (it will be downloaded automatically).

```bash
DS_LOCATION='datasets/segment' jupyter notebook notebooks/2.1_download_preprocess_datasets.ipynb
```

### Train

Run the Jupyter notebook `notebooks/2.2_box_segmentation_yolo_train.ipynb` to train the YOLOv9 and YOLOv11 on
the SCD datset (OSCD followed by LSCD).

```bash
DS_LOCATION='datasets/segment' jupyter notebook notebooks/2.2_box_segmentation_yolo_train.ipynb
```

### Evaluate

⚠️⚠️⚠️ WARNING ⚠️⚠️⚠️ **TRAINING THE MODEL IS NOT NECESSARY TO RUN THE EVALUATION BUT YOU NEED TO [DOWNLOAD AND PROCESS THE DATASET](#preprocess-dataset)**

You can either train the model or use the pretrained models on the [SCD dataset](https://arxiv.org/abs/2102.12808).

Download the YOLO checkpoints pretrained on the SCD dataset from [here](https://1drv.ms/f/c/68347683875c28db/EkdgNnKKAqJPrrGnipEBb0YBBwWiGve1j4QCmm6hDR6OFQ?e=75zlm1) and place them in `notebooks/runs/` folder e.g., `notebooks/runs/segment/2.2G_2_ft_mscd_yolo11m_epoch12`


Run the Jupyter notebook `notebooks/2.3_box_segmentation_yolo_eval.ipynb` to evaluate the trained models,

This will evaluate the `train_...` prefixed trained models if specified with `MODEL_NAME_PREFIX="train_"`, otherwise, 
it will use the model checkpoints already trained on the SCD carton box dataset.

```bash
MODEL_NAME_PREFIX='train_' DS_LOCATION='datasets/segment' jupyter notebook notebooks/2.3_box_segmentation_yolo_eval.ipynb
```

**To Execute and Export the Notebook**

⚠️⚠️⚠️ WARNING ⚠️⚠️⚠️ **THE FOLLOWING REPORT-GENERATING SCRIPT MAY TAKE 5 MINUTES ON A GPU LIKE RTX 3080 TI OR MORE THAN AN HOUR ON CPU**

remove the `MODEL_NAME_PREFIX='train_'` if you would like to evaluate on the trained model checkpoints 

```bash
MODEL_NAME_PREFIX='train_' DS_LOCATION='datasets/segment' jupyter nbconvert --execute --to markdown notebooks/2.3_box_segmentation_yolo_eval.ipynb
```

**NOTE** Report can be found [here](assets/reports/2_box_segmentation_yolo_eval.md)
