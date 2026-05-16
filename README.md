# Recognizing Handwritten Digits using Scikit-Learn

## Overview
This project demonstrates how to build and train a Multi-Layer Perceptron (MLP) neural network to recognize handwritten digits using the `scikit-learn` library. `scikit-learn` is a widely adopted machine learning library in Python, renowned for its clean API and comprehensive suite of algorithms for classification, regression, clustering, and data preprocessing. It integrates seamlessly with Python's scientific stack (NumPy and SciPy).

## Dataset
The dataset utilized is the standard digits dataset provided within `scikit-learn` (`sklearn.datasets.load_digits`). 
- **Total Images:** 1797
- **Resolution:** 8x8 pixels (64 pixels per image)
- **Features:** Pixel intensities (0-16)

The original digits possessed a much higher resolution. The resolution was intentionally reduced when preparing the dataset for `scikit-learn` to facilitate faster training and easier recognition for machine learning algorithms.

## Implementation Details

### 1. Data Loading and Exploration
The dataset is loaded directly from `sklearn.datasets`. The `digits.images` attribute contains a 3D array (1797, 8, 8), representing a stack of 8x8 pixel images.

```python
from sklearn import datasets

digits = datasets.load_digits()
print("Available attributes:", dir(digits))
print("First image matrix:\n", digits.images[0])
```

### 2. Data Visualization
To visualize the data, a helper function utilizes `matplotlib` to display a batch of 16 images along with their corresponding ground truth labels.

```python
import matplotlib.pyplot as plt

def plot_multi(i):
    nplots = 16
    fig = plt.figure(figsize=(15, 15))
    for j in range(nplots):
        plt.subplot(4, 4, j+1)
        plt.imshow(digits.images[i+j], cmap='binary')
        plt.title(digits.target[i+j])
        plt.axis('off')
    plt.show()

plot_multi(0)
```

### 3. Data Preprocessing and Splitting
Neural networks typically expect one-dimensional input vectors for dense layers. Therefore, the 2D (8x8) images are flattened into 1D arrays of 64 elements. Next, the data is normalized (scaled to between 0 and 1) to help the neural network train faster and more reliably. The dataset is then partitioned into training and testing subsets using `train_test_split` to ensure the model's performance can be evaluated on unseen data.

```python
from sklearn.model_selection import train_test_split

# Flattening the images
y = digits.target
x = digits.images.reshape((len(digits.images), -1))
print("Data shape after flattening:", x.shape) # Output: (1797, 64)

# Normalize the data to range [0, 1]
x = x / 16.0

# Train-test split (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

### 4. Neural Network Architecture (MLPClassifier)
A Multi-Layer Perceptron (MLP) is utilized for the classification task. The input layer consists of 64 nodes corresponding to the input pixels. This is a dense neural network where every node in a layer connects to all nodes in the subsequent layer.

```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(15,),
    activation='logistic',
    alpha=1e-4, 
    solver='sgd',
    tol=1e-4, 
    random_state=1,
    learning_rate_init=.1,
    verbose=True
)
```

### 5. Model Training
The model is trained on the first 1000 samples. The optimization process attempts to minimize the loss over multiple iterations.

```python
mlp.fit(x_train, y_train)
```

During training, the loss progressively decreases. The training algorithm automatically halts when the loss does not improve by more than the specified tolerance (`tol=0.0001`) for 10 consecutive epochs.

The loss curve can be plotted to analyze the training progression:
```python
fig, axes = plt.subplots(1, 1)
axes.plot(mlp.loss_curve_, 'o-')
axes.set_xlabel("number of iteration")
axes.set_ylabel("loss")
plt.show()
```

## Model Evaluation
The trained model is evaluated using the test partition (the remaining 797 samples) to measure its generalization capability.

```python
from sklearn.metrics import accuracy_score

predictions = mlp.predict(x_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
```

**Results:**
The Multi-Layer Perceptron achieves an accuracy of approximately **91.47%** on the test dataset, demonstrating a strong capability to recognize the handwritten digits despite the low resolution of the input images.
