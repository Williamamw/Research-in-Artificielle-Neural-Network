# Research-in-Artificielle-Neural-Network
Third year's project : Implementation of a Neural Network from scratch to understand the mathematical foundations of Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Academic_Research-success)

## 📖 Overview
This project involves the manual implementation of Artificial Neural Networks (ANN) to deconstruct the "Black Box" of machine learning.

Developed as part of my Bachelor's Thesis (Travaux d'Études et de Recherche), this repository contains raw Python implementations of:
1.  **A Simple Perceptron:** For linear classification problems.
2.  **A Multilayer Perceptron (MLP):** For solving non-linear problems (e.g., XOR) using Backpropagation.

## 🎯 Key Features
* **No Frameworks:** Built entirely without TensorFlow or PyTorch to ensure a First-Principles understanding.
* **Mathematical Rigor:** Implements the Chain Rule for **Backpropagation** and **Gradient Descent** manually.
* **Activation Functions:** Manual implementation of the **Sigmoid** function and its derivative.
* **Vectorization:** Utilizes `numpy.dot` for efficient matrix operations.

## 📂 Repository Structure

### 1. Multilayer Perceptron (The Core)
* **File:** `Perceptron multicouche.py`
* **Description:** An implementation of a neural network with a hidden layer.
* **Application:** Solves the **XOR problem** (Exclusive OR), demonstrating the network's ability to learn non-linear boundaries—something a simple perceptron cannot do.

### 2. Simple Perceptron
* **File:** `Perceptron Simple.py`
* **Description:** A single-layer perceptron implementation.
* **Application:** Used for simple linear classification tasks to demonstrate the fundamentals of weights and biases.

### 3. Theoretical Foundation (Thesis)
* **File:** `TER.pdf`
* **Description:** My comprehensive research paper (in French).
    * **Chapter 2.8:** Detailed mathematical derivation of the **Backpropagation algorithm**.
    * **Chapter 3:** Experimental results and Python implementation details.

## 🛠️ How It Works (The Math)
The network learns by minimizing the error between the predicted output and the target.
* **Forward Pass:** $y = \sigma(W \cdot x)$
* **Backward Pass:** Weights are updated based on the gradient of the error with respect to the weights:
    $$W_{new} = W_{old} + \text{adjustment}$$

*(See the attached PDF for the full Jacobian matrix derivations)*

## 🙏 Acknowledgements
The core code structure was inspired by **Milo Spencer-Harper's** tutorial: *[How to build a simple neural network in 9 lines of Python code](https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1)*.

I extended this work by providing the comprehensive mathematical proofs in the attached thesis (`TER.pdf`) and analyzing the performance differences between simple and multilayer architectures.

---
*Author: Ming Wei ANG*
