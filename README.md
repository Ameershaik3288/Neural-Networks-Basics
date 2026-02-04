Day 3 – Introduction to Neural Networks (PyTorch)

This project is part of a workshop and explains the basics of neural networks using PyTorch.
It shows how a simple neural network can learn patterns from data using training.

What is This Project About?

In this project, we train a small neural network to learn an XOR-like problem, which cannot be solved using simple linear rules.
This helps beginners understand why neural networks are useful.

Key Concepts Used:

👉PyTorch Tensors
👉Neural Networks
👉Forward Pass
👉Loss Function
👉Backpropagation
👉Optimizer (Adam)

Dataset Explanation
Input (X)

Each row represents one data point with two features:
[0, 0]
[0, 1]
[1, 1]
[1, 0]

Output (y)

Each value is the expected result for the corresponding input:
[0]
[1]
[1]
[0]

This pattern is called XOR, which requires a neural network to solve.

Neural Network Structure :

The model is created using nn.Sequential, which connects layers one after another.
Input Layer (2 values)
   ↓
Linear Layer (2 → 4)
   ↓
ReLU Activation
   ↓
Linear Layer (4 → 1)
   ↓
Output

Layer Explanation :
1.Linear (2 → 4): Converts input into 4 features
2.ReLU: Adds non-linearity so the model can learn complex patterns
3Linear (4 → 1): Produces the final output

Training the Model:
The model is trained for 1000 iterations.
Each training step includes:
👉Clearing old gradients
👉Passing input through the model
👉Calculating error using MSE Loss
👉Updating weights using Adam Optimizer

This process helps the model improve step by step.

Final Output ✅
After training, the model gives the following predictions:
tensor([[ 0.0000],
        [ 1.0000],
        [ 1.0000],
        [-0.0000]])
