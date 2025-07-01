# Perceptron-Based ANN Simulation

---

### Name: Aruna Thapa
### CRN: 021-311
### Subject: Artificial Intelligence

---

##  About

This project implements a **simple Artificial Neural Network (ANN)** using the **Perceptron learning algorithm** from scratch in **Python**, relying only on **NumPy**.  
It is designed for **educational purposes** to demonstrate the internal working of a **binary classifier** using a **step (threshold) activation function**.

---

##  Algorithm Overview

The Perceptron learns by adjusting weights based on the prediction error using the formula:

```
w = w + α * error * input
```

Where:  
* `w` → weight  
* `α` → learning rate  
* `error` → actual output - predicted output  

The model iteratively updates weights over multiple **epochs** until:  
* The total error becomes zero, or  
* The maximum number of epochs is reached.

---

##  Features

* Binary classification using the Perceptron model  
* Step activation function for binary output  
* Adjustable learning rate (`alpha`), threshold (`theta`), and epoch count  
* Easy to test on logic gates like **OR**, **AND**, etc.  
* Only external dependency: **NumPy**

---

## 🧪 Sample Dataset (OR Gate)

| Input 1 | Input 2 | Expected Output |
|---------|---------|-----------------|
|   0     |    0    |        0        |
|   0     |    1    |        1        |
|   1     |    0    |        1        |
|   1     |    1    |        1        |

---

## 🧾 Code

```python
import numpy as np

class Perceptron:
    def __init__(self, input_size, alpha=0.1, theta=0.5, epochs=100):
        self.alpha = alpha  # learning rate
        self.theta = theta  # threshold
        self.epochs = epochs
        self.weights = np.random.rand(input_size + 1)  # +1 for bias

    def activation(self, x):
        return 1 if x >= self.theta else 0

    def predict(self, inputs):
        inputs = np.insert(inputs, 0, 1)  # Add bias input
        weighted_sum = np.dot(inputs, self.weights)
        return self.activation(weighted_sum)

    def train(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # Insert bias term in input

        for epoch in range(self.epochs):
            total_error = 0
            for i in range(len(X)):
                output = self.activation(np.dot(X[i], self.weights))
                error = y[i] - output
                self.weights += self.alpha * error * X[i]
                total_error += abs(error)
            if total_error == 0:
                break

# Training with OR gate dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 1, 1, 1])

model = Perceptron(input_size=2, alpha=0.1, theta=0.5, epochs=10)
model.train(X, y)

# Prediction
for sample in X:
    print(f"Input: {sample}, Predicted: {model.predict(sample)}")
```

---

## Output Obtained

```
Input: [0 0], Predicted: 0  
Input: [0 1], Predicted: 1  
Input: [1 0], Predicted: 1  
Input: [1 1], Predicted: 1
```

*Screenshot:*

![Output Screenshot](Screenshot_2025-06-22_194309.png)

---

## 🛠 Technologies Used

* **Language**: Python 3  
* **Library**: NumPy  
* **IDE**: VS Code or Jupyter Notebook

---

## 📘 Objective

To simulate the learning process of a **Perceptron neural network**, understand weight updates, and test it using simple datasets like logic gates.

---
