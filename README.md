# Dynamic Heatmap Video to Gaze Points

## Overview
This project aims to extract gaze points from dynamic heatmap videos using deep learning models. It processes video frames with heatmaps, detects hotspots and retrieves gaze points.

## Project Structure
```
.
├── dataset
│   ├── video1
│   │   ├── annotations.json
│   │   ├── frames/
│   ├── video2
│   ├── video3
│   ├── video4
├── models
│   ├── model1.pth
│   ├── model2.pth
│   ├── model3.pth
├── training
│   ├── train_model1.py
│   ├── train_model2.py
│   ├── train_model3.py
├── inference
│   ├── infer_model1.py
│   ├── infer_model2.py
│   ├── infer_model3.py
├── testing
│   ├── test_model1.py
│   ├── test_model2.py
│   ├── test_model3.py
├── README.md
└── requirements.txt
```

## Dataset Structure
The dataset consists of frames extracted from heatmap videos, along with annotations containing bounding boxes.

```
annotation_paths = [
    '../dataset/video1/annotations.json',
    '../dataset/video2/annotations.json',
    '../dataset/video3/annotations.json',
    '../dataset/video4/annotations.json'
]

image_dirs = [
    '../dataset/video1/frames',
    '../dataset/video2/frames',
    '../dataset/video3/frames',
    '../dataset/video4/frames'
]
```

## Installation
### Prerequisites
- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Pandas

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Training
To train a model, run:
```sh
python training/train_model1.py
```

## Inference
To run inference on test frames:
```sh
python inference/infer_model1.py
```

## Testing
To evaluate model performance:
```sh
python testing/test_model1.py
```

## Results
The output consists of bounding boxes on frames where the original bounding box is drawn in **black** and the predicted bounding box is in **pink**.
