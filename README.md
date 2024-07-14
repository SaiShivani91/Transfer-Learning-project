# Dog Vs Cat Classification using Transfer Learning

This project focuses on classifying images of dogs and cats using transfer learning. Transfer learning involves pre-trained models to solve new tasks, reducing the need for large datasets and extensive training.

## Tools and Technologies:

- Python: Scripting and model building.
- Kaggle API: Downloading the dataset.
- TensorFlow and TensorFlow Hub: Deep learning libraries.
- PIL (Python Imaging Library): Image processing.
- OpenCV: Image reading and preprocessing.
- Matplotlib: Visualizing images.
- Scikit-learn: Data splitting and preprocessing.
- Google Colab: Execution environment with GPU access.

## Transfer Learning:

Transfer learning involves using a model trained for one task as the starting point for a model on a second task. It is beneficial because it saves computational resources and achieves higher accuracy with smaller datasets.

There are several pretrained models used for different tasks such as:

- VGG-16: Known for image recognition tasks.
- ResNet50: Handles very deep networks effectively.
- Inceptionv3: Efficient for various image recognition tasks.
- MobileNet V2: Lightweight and suitable for mobile and embedded vision applications.
- YOLO (You Only Look Once): Used for real-time object detection.

## Workflow:

1. Dataset Acquisition: Using the Kaggle API to download the dog vs. cat dataset.
2. Image Processing: Resizing and normalizing images to fit the input requirements of the MobileNet V2 model.
3. Train-Test Split: Dividing the dataset into training and testing sets.
4. Model Selection and Training: Using MobileNet V2 for transfer learning.
5. Prediction: Predicting whether an image is of a dog or a cat.

## Dataset:

The dataset used in this project is sourced from Kaggle's "Dogs vs. Cats" competition. It contains 25,000 images labeled as either dogs or cats. For this project, a subset of 2,000 images was used to train and evaluate the model. Images are resized to 224x224 pixels to match the input size required by MobileNet V2.

MobileNet V2

MobileNet V2 is a lightweight and efficient model designed for mobile and embedded vision applications. It is trained on the ImageNet dataset, which contains millions of images across thousands of classes. MobileNet V2 uses depthwise separable convolutions, significantly reducing the number of parameters and computation while maintaining performance.

## Finetuning:

Loading the Pre-trained Model: MobileNet V2 is loaded with pre-trained weights from TensorFlow Hub.

Freezing the Base Layers: The base layers of MobileNet V2 are kept non-trainable to retain the learned features.

Adding a Classification Layer: A dense layer with two output units (representing dogs and cats) is added on top of the base model.

Compiling the Model: The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss function.

Training the Model: The model is trained on the resized and normalized images for a few epochs.

## Model Performance:

The trained model achieved a test loss of 0.0624 and a test accuracy of 97%. This high accuracy demonstrates the effectiveness of transfer learning with MobileNet V2 for this classification task.

## Alternative Models:

While MobileNet V2 is an excellent choice for this project due to its efficiency and performance, several other pre-trained models could also be used:

- VGG-16: Known for its simplicity and effectiveness in image classification.
- ResNet50: Provides robustness for deep learning tasks.
- Inceptionv3: Offers high performance for complex image recognition tasks.
- DenseNet: Known for its ability to build very deep networks efficiently.

## Conclusion:

This project successfully demonstrates the use of transfer learning with MobileNet V2 for classifying images of dogs and cats. Transfer learning significantly reduces training time and improves accuracy by leveraging pre-trained models. For similar projects, models like VGG-16, ResNet50, Inceptionv3, and DenseNet can also be considered as effective alternatives.
