# Excercise_1
## Logistic Regression on MNIST Dataset
This project demonstrates a basic implementation of Logistic Regression using PyTorch to classify digits from the MNIST dataset.

## Overview
The provided code performs the following steps:

Loads and Preprocesses the MNIST Dataset: Fetches the MNIST dataset from OpenML, splits it into training and testing sets, and normalizes the data.
Defines and Trains a Logistic Regression Model: Uses PyTorch to define a logistic regression model, trains it using Stochastic Gradient Descent (SGD), and monitors the training loss.
Evaluates the Model: Tests the trained model on the test dataset and calculates the classification accuracy.

## Prerequisites
Python 3.x
PyTorch
Scikit-Learn
NumPy

You can install the necessary packages using pip:
bash
pip install torch torchvision scikit-learn numpy

## Code Explanation
### Configuration
GPU Configuration: The code checks for GPU availability and sets the device accordingly.
### Data Loading and Preprocessing
Load MNIST Dataset: The dataset is fetched from OpenML.
Split Data: The data is split into training and testing sets.
Normalize Data: The features are standardized using StandardScaler.
Convert to Tensors: Data is converted to PyTorch tensors and moved to GPU if available.
### Model Definition
Logistic Regression Model: A simple logistic regression model with one fully connected layer is defined. It maps 784 input features (28x28 images) to 10 output classes (digits 0-9).
### Training
Loss Function and Optimizer: The CrossEntropyLoss is used with Stochastic Gradient Descent (SGD) for optimization.
Training Loop: The model is trained for 10 epochs, with loss printed after each epoch.
### Evaluation
Testing: After training, the model is evaluated on the test set, and accuracy is computed and printed.
Usage

## Clone the repository:
bash
git clone <repository-url>
cd <repository-directory>
Install the required packages (if not already installed).

Run the script:

bash
python <script-name>.py
Replace <repository-url>, <repository-directory>, and <script-name> with the appropriate values.

## Results
The script will output the loss for each epoch during training and the final accuracy of the model on the test set.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
