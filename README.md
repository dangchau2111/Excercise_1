# Softmax Regression for MNIST Dataset

## Introduction

This repository contains an implementation of Softmax Regression for the MNIST dataset, which is a classic benchmark in machine learning for handwritten digit classification. Softmax Regression is a generalized linear model for multiclass classification problems. In this implementation, we use NumPy to perform all necessary computations.

## Explanation

### Softmax Regression

Softmax Regression (or Multinomial Logistic Regression) is used to classify data into multiple classes. The model estimates the probability distribution of each class given the input features. 

**Softmax Function**:
The Softmax function converts logits (raw prediction values) into probabilities. It is defined as:

\[ P(y = k \mid \mathbf{x}; \mathbf{\theta}) = \frac{\exp(z_k)}{\sum_{j} \exp(z_j)} \]

where \( z_k \) is the logit for class \( k \) and the denominator sums over all possible classes \( j \).

**Loss Function**:
The Categorical Cross-Entropy Loss is used to quantify the difference between the predicted probabilities and the actual class labels. It is defined as:

\[ \text{Loss} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_{i,k} \log(p_{i,k}) \]

where \( m \) is the number of samples, \( K \) is the number of classes, \( y_{i,k} \) is a binary indicator (0 or 1) if class label \( k \) is the correct classification for sample \( i \), and \( p_{i,k} \) is the predicted probability of class \( k \) for sample \( i \).

**Gradient Descent**:
Gradient Descent is used to optimize the Softmax Regression model by updating the model parameters (weights) iteratively. The gradient of the loss function with respect to the model parameters is computed and used to update the weights:

\[ \theta := \theta - \alpha \cdot \nabla_\theta \text{Loss} \]

where \( \alpha \) is the learning rate and \( \nabla_\theta \text{Loss} \) is the gradient of the loss function with respect to \( \theta \).

**Flow of Computation**:
1. Compute logits using the linear model: \( z = X \cdot \theta \)
2. Apply the Softmax function to obtain class probabilities: \( p = \text{softmax}(z) \)
3. Compute the Cross-Entropy Loss: \( \text{Loss} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_{i,k} \log(p_{i,k}) \)
4. Update model parameters using Gradient Descent.

## Execution

### Training Results

The model was trained for 1000 iterations with the following loss values recorded at various checkpoints:
Iteration 0: Loss = 1.5888747638492926
Iteration 100: Loss = 0.34419817129761715
Iteration 200: Loss = 0.3092600330260409
Iteration 300: Loss = 0.2938721864571801
Iteration 400: Loss = 0.284682588933372
...
Iteration 800: Loss = 0.26691662627460555
Iteration 900: Loss = 0.26435794986878547
Accuracy: 91.89%
