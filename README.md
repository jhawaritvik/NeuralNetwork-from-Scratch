# Neural Network from Scratch - MNIST Classification

This project implements Neural Networks from scratch without the use of libraries like TensorFlow, PyTorch, or scikit-learn.

### Main Dependencies:
- **NumPy** – for calculations and scientific constants
- **Matplotlib** – for visualization of results
- **gzip** – to unzip files

### Features:
- Fully connected 3-layered Neural Network
- Layers with ReLU and Softmax activation functions
- Cross-Entropy Loss to calculate loss
- Gradient Descent used as optimizer
- Epoch Count: 15
- Training accuracy: 94.675%
- Testing accuracy: 93.26%

### Network Architecture:
- **Input Layer**: 784 neurons (28x28 pixel values)
- **Hidden Layer**: Customizable (default: 50 neurons)
- **Output Layer**: 10 neurons (digits 0-9)
- **Activation Functions**: ReLU, Softmax
- **Loss Function**: Cross-Entropy
- **Optimizer**: Gradient Descent
