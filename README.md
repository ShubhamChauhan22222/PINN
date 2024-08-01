# Lens Classification using ResNet-18 and Physics-Informed Neural Networks (PINN)

## Overview

This repository contains the implementation of two models for classifying images of gravitational lenses using PyTorch. The first model employs a standard ResNet-18 architecture, while the second model enhances the ResNet-18 with a physics-informed neural network (PINN) that incorporates the gravitational lensing equation.

## Common Test: Multi-Class Classification

### Task

Build a model for classifying images into lenses using PyTorch or Keras. The model should be trained and validated using a suitable approach to achieve high accuracy.

### Approach

I utilized ResNet-18 for this task due to its ability to capture intricate patterns and features in the data, leading to better generalization and higher accuracy. The model was trained using 5-fold cross-validation, with each fold consisting of 5 epochs on a dataset of 30,000 training images and 7,500 test images.

### Results

The model achieved an ROC-AUC score of 0.99 on the test data.

### Notebook

You can find the detailed implementation and results of this approach in the following Jupyter notebook: [ResNet-18 Approach](https://github.com/ShubhamChauhan22222/PINN/blob/main/Resnet18_Approach.ipynb)

## Specific Test V: Physics-Guided ML

### Task

Build a model for classifying images into lenses using PyTorch or Keras. The architecture should take the form of a physics-informed neural network (PINN) that incorporates the gravitational lensing equation to improve network performance over the common test results.

### Approach 1

For classifying gravitational lenses into three types (no lensing, vortex, and halo substructure), I incorporated the lens equation, which describes how light is bent by the gravitational field of a massive object. The mass distribution of the lensing object is assumed to follow a Singular Isothermal Sphere (SIS) model, with a proportionality parameter \( k \) to correct potential distortions.

#### Implementation Steps

1. Define the lens equation:
    \[
    \beta = \theta - \alpha
    \]
    where \( \beta \) is the apparent position of the source, \( \theta \) is the observed position, and \( \alpha \) is the deflection angle.
2. Incorporate the mass distribution due to galaxies and dark matter:
    \[
    \beta + cX = \theta - kr^2
    \]
3. Utilize feature vectors \( \theta \) and \( k \) from ResNet-18.
4. Apply three neural layers on the resulting vector to extract features for lens classification.

### Results

The model achieved an ROC-AUC score of 0.92 on the test data.

### Notebook

You can find the detailed implementation and results of this approach in the following Jupyter notebook: [PINN Approach 1](https://github.com/ShubhamChauhan22222/PINN/blob/main/PINN%20approach_1.ipynb)

### Approach 2

In this approach, the original image vector \( I \) represents \( \theta \). Another feature vector \( k \) from ResNet-18 is used as follows:
\[
C = I - B_{\text{features}} \cdot r^2
\]
A concatenated feature vector \( D \) is formed from another ResNet-18 feature vector \( A_{\text{features}} \) and \( C \). This vector is then processed through three neural layers to classify the lenses.

### Notebook

You can find the detailed implementation and results of this approach in the following Jupyter notebook: [PINN Approach 2](https://github.com/ShubhamChauhan22222/PINN/blob/main/PINN%20approach_2.ipynb)

---

