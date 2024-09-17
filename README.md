

# Handwriting Recognition System

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

This project is an end-to-end solution for recognizing handwritten text using deep learning. It combines Convolutional Neural Networks (CNNs) and Bidirectional Long Short-Term Memory (BiLSTM) networks with residual connections for improved recognition performance. The model is trained on the **IAM Handwriting Database**, a well-known dataset for handwritten text recognition.

## Features
- **End-to-End Handwriting Recognition**: Recognizes entire handwritten sentences.
- **Residual Blocks for Feature Extraction**: Efficient convolutional blocks to extract features from input images.
- **Bidirectional LSTMs for Sequence Modeling**: Sequence modeling to handle time-dependent data like handwriting.
- **Preprocessing and Augmentation**: Includes data preprocessing and augmentation techniques to improve the model’s generalization.
- **Customizable Parameters**: Modify training configurations such as batch size, learning rate, and dropout in the `configs.py` file.

## File Structure

```bash
Handwriting_Recognization/
│
├── configs.py              # Configuration file for model parameters
├── inferenceModel.py        # Script to run inference on an image
├── model.py                # Contains the model architecture
├── train.py                # Script to train the model
├── requirements.txt        # List of dependencies
└── mltu/                   # Utility folder for model helpers
    ├── __init__.py
    └── tensorflow/
        └── model_utils.py  # Contains utility functions like residual_block
```

## Dataset

The model is trained on the **[IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)**, which contains handwritten text samples from 657 writers. The database includes labeled images of handwritten English text at both word and sentence levels.

To download the dataset, you need to register and request access via the official website:  
**[IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)**

After obtaining the dataset, ensure the files are organized as required for model training.

## Model Architecture

The model combines Convolutional Neural Networks (CNNs) with Bidirectional Long Short-Term Memory (BiLSTM) layers for effective handwriting recognition. Below is an overview of the architecture:

```python
from keras import layers
from keras.models import Model
from mltu.tensorflow.model_utils import residual_block

def train_model(input_dim, output_dim, activation='leaky_relu', dropout=0.2):
    
    inputs = layers.Input(shape=input_dim, name="input")
    
    # Normalize images
    input = layers.Lambda(lambda x: x / 255)(inputs)

    # Residual blocks for feature extraction
    x1 = residual_block(input, 32, activation=activation, skip_conv=True, strides=1, dropout=dropout)
    x2 = residual_block(x1, 32, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x3 = residual_block(x2, 32, activation=activation, skip_conv=False, strides=1, dropout=dropout)
    x4 = residual_block(x3, 64, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x5 = residual_block(x4, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout)
    x6 = residual_block(x5, 128, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x7 = residual_block(x6, 128, activation=activation, skip_conv=True, strides=1, dropout=dropout)
    x8 = residual_block(x7, 128, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x9 = residual_block(x8, 128, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    # Reshape output
    squeezed = layers.Reshape((x9.shape[-3] * x9.shape[-2], x9.shape[-1]))(x9)

    # Bidirectional LSTM layers for sequence modeling
    blstm = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(squeezed)
    blstm = layers.Dropout(dropout)(blstm)
    blstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(blstm)
    blstm = layers.Dropout(dropout)(blstm)

    # Output layer with softmax activation
    output = layers.Dense(output_dim + 1, activation='softmax', name="output")(blstm)

    # Define model
    model = Model(inputs=inputs, outputs=output)
    return model
```

### Key Components:
1. **Residual Blocks**: These blocks help in retaining information across deeper layers by adding the input directly to the output. This helps avoid vanishing gradient problems.
2. **BiLSTM Layers**: Used to capture contextual information from both directions in sequential data like handwritten text.
3. **CTC Loss**: The model is designed for sequence-to-sequence tasks, using Connectionist Temporal Classification (CTC) loss for training.

## Installation

To install and run the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/devgupta2619/Handwriting_Recognization.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Handwriting_Recognization
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model
To train the handwriting recognition model, run the `train.py` script:
```bash
python train.py
```
Make sure the **IAM Handwriting Database** is downloaded and properly structured. The training process will use the images and labels provided in the dataset. Training parameters, such as batch size and learning rate, can be adjusted in `configs.py`.

### Inference
To perform inference on handwritten images and convert them into text, use the following command:
```bash
python inferenceModel.py --image <path-to-image>
```
This will load a pre-trained model and return the recognized text.

## Configurations
You can adjust the model’s configurations by modifying the `configs.py` file. Some of the parameters you can change include:
- **Input Image Dimensions**
- **Batch Size**
- **Learning Rate**
- **Number of Epochs**
- **Dropout Rate**

