# Workshop

Day3
Introduction to Neural Networks

First line of code creates a PyTorch tensor named X. A tensor is a fundamental data structure in PyTorch, similar to an array or matrix. In this case, X is a 2-dimensional tensor (a matrix) containing three rows and two columns, with the specified integer values.

Second Line code defines a neural network called model using PyTorch's nn.Sequential container. nn.Sequential stacks layers and passes the output of one layer as the input to the next.

Here's a breakdown of the layers:

1.nn.Linear(2, 4): This is a linear (or fully connected) layer. It takes an input of size 2 and produces an output of size 4. This is typically where features are transformed.

2.nn.ReLU(): This is a Rectified Linear Unit activation function. It introduces non-linearity into the model by outputting the input directly if it's positive, otherwise it outputs zero. This helps the network learn complex patterns.

3.nn.Linear(4, 1): This is another linear layer. It takes the output of the ReLU layer (which has 4 features) and produces a single output feature. This final layer often represents the model's prediction or classification.
