# Garbage Classification Model Using Neural Networks

This project implements a garbage classification system that categorizes images of garbage into 6 different classes using transfer learning with EfficientNetB3. The model is trained on a dataset containing 2000 images (400 images per class).

---

## Project Overview

- **Dataset:** 6 classes (e.g., cardboard, glass, metal, paper, plastic, trash), 400 images each.
- **Model:** EfficientNetB3 pretrained on ImageNet, with a custom classification head.
- **Framework:** TensorFlow and Keras.
- **Techniques:**
  - Data augmentation for better generalization.
  - Fine-tuning of pretrained layers.
  - Evaluation with confusion matrix and classification report.
- **Environment:** Google Colab (recommended for GPU acceleration).

---

## Folder Structure

data/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/


## Getting Started

### Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- scikit-learn
- matplotlib
- seaborn

#Install required libraries via:  ```bash
pip install tensorflow scikit-learn matplotlib seaborn
Usage```

## Prepare dataset
Organize your images into folders as shown above.


## Upload dataset to Google Drive
Upload your ```data``` folderto Google Drive for easy access in Colab.

## Run notebook
Open the provided Jupyter notebook garbage_classification_transfer_learning.ipynb in Google Colab

## Mount your Google Drive.
Set the dataset path to your Google Drive folder.
Run cells sequentially to preprocess data, train the model, fine-tune, and evaluate.

## Upload and test images:
Use the image upload cell to test predictions on individual images.

## Model Architecture
Base: EfficientNetB3 pretrained on ImageNet (weights frozen initially).

## Custom head
GlobalAveragePooling → Dense(512, ReLU) → BatchNormalization → Dropout(0.5) → Dense(6, Softmax).

## Training and Fine-tuning
Initial training with frozen base layers for 10 epochs.
Fine-tuning top layers of EfficientNetB3 with a low learning rate for an additional 20 epochs.
Early stopping and learning rate reduction callbacks are used to improve convergence.

## Evaluation
Accuracy and loss curves are plotted.
Confusion matrix and classification report show detailed per-class performance.

## Future Improvements
Experiment with other architectures (e.g., EfficientNetV2, ResNet).
Increase dataset size or use synthetic data augmentation.
Deploy the model as a web or mobile application for real-time garbage classification fo using in automated recycling tasks.

Author:
P.A.W.Pelawaththa
Rajarata University of Sri Lanka
wasana200094@gmail.com

Co-authors:
M.A.N.K.Wanigasekara,
T.L.D.Priyadarshani
