# Skin_Cancer_Detection_CNN
This project implements a deep learning model for automatic melanoma detection from dermoscopic images. Using the Melanoma Skin Cancer Dataset (10,000 images), the model distinguishes between benign and malignant skin lesions.


## Dataset
- **Source:** [Kaggle – Melanoma Skin Cancer Dataset of 10,000 Images](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images)
- **Structure:**
- 

train/
├── benign/
└── malignant/
test/
├── benign/
└── malignant/


- **Size:**
- 9,600 images for training  
- 1,000 images for testing  

## Features
- Developed using **TensorFlow** and **Keras**.
- Data augmentation for better generalization.
- CNN architecture with **Batch Normalization** and **Dropout**.
- **Early Stopping** and **Model Checkpointing** to prevent overfitting.
- Evaluation on a held-out test set with **accuracy** and **loss** metrics.

## How to Run
```bash
pip install tensorflow kagglehub
python melanoma_classification.py
