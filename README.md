# EfficientNet-B0 + SVM Image Classification Project

## Overview

This project focuses on image classification by combining **deep feature extraction** using the **EfficientNet-B0** model and **Support Vector Machine (SVM)** classification. The dataset is augmented with various transformations to enhance model performance and generalization.

## Project Features

- **EfficientNet-B0** is used as a feature extractor.
- **SVM classifier** is trained on the extracted features.
- **Data augmentation** is applied 10x per image to enrich training data.
- **Average test accuracy** achieved: **96%**.
- Custom dataset loader supports both original and augmented images.
- Experiments are repeated 5 times with different train-test splits.

## Dataset

The dataset must be organized in ImageFolder format under the path:

/dataset/
   ├── class_1/
   │    ├── image1.jpg
   │    ├── image2.jpg
   │    └── ...
   ├── class_2/
   │    └── ...
   └── ...

All images are resized to 224x224 and normalized to match the input requirements of EfficientNet.

Requirements:
Python 3.x
PyTorch
torchvision
scikit-learn
numpy
PIL

Install dependencies with:

pip install torch torchvision scikit-learn numpy pillow

## Performance

✅ Average Test Accuracy: 96%
✅ Evaluation Time: ~ a few minutes depending on GPU availability

## Contact

Furkan Arslan
📧 furkan0tr0arslan@gmail.com
