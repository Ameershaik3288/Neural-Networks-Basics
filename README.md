# Workshop

Day3
Introduction to Neural Networks

First line of code creates a PyTorch tensor named X. A tensor is a fundamental data structure in PyTorch, similar to an array or matrix. In this case, X is a 2-dimensional tensor (a matrix) containing three rows and two columns, with the specified integer values.

Second Line code defines a neural network called model using PyTorch's nn.Sequential container. nn.Sequential stacks layers and passes the output of one layer as the input to the next.

Here's a breakdown of the layers:

1.nn.Linear(2, 4): This is a linear (or fully connected) layer. It takes an input of size 2 and produces an output of size 4. This is typically where features are transformed.

2.nn.ReLU(): This is a Rectified Linear Unit activation function. It introduces non-linearity into the model by outputting the input directly if it's positive, otherwise it outputs zero. This helps the network learn complex patterns.

3.nn.Linear(4, 1): This is another linear layer. It takes the output of the ReLU layer (which has 4 features) and produces a single output feature. This final layer often represents the model's prediction or classification.

Next line code initializes two PyTorch tensors, X and y.

1.X = torch.tensor([[0.,0.],[0.,1.],[1.,1.],[1.,0.]]): This line creates a 2D tensor named X. It represents the input features, with each inner list [a., b.] being a data point (or sample) and the a. and b. values being its features. In this case, there are four data points, each with two features.

2.y = torch.tensor([[0.],[1.],[1.],[0.]]): This line creates another 2D tensor named y. It represents the target labels or outputs corresponding to the input data in X. Each inner list [c.] is the label for the respective data point in X. Here, there are four labels, each a single floating-point value.

Next code line performs a forward pass of the input tensor X through the trained neural network model. Essentially, it's asking the model to make predictions based on the X input data after it has been trained. The print() statement then displays these predictions, which in this case are the model's learned outputs for the XOR-like input patterns.

Nextcline This code block implements the training loop for the neural network. Let's break down each step:

👉for _ in range(1000): This line starts a loop that will run 1000 times. Each iteration of this loop is often referred to as an 'epoch' or a 'training step', where the model learns from the entire dataset.

👉opt.zero_grad(): Before calculating new gradients, it's crucial to clear the old ones. This line sets the gradients of all parameters optimized by opt (our Adam optimizer) to zero. If you don't do this, gradients would accumulate from previous iterations.

👉y_pred = model(X): This is the forward pass. Here, the input data X is passed through the neural network model to produce predictions, y_pred.
loss = loss_fn(y_pred, y): This line calculates the loss. The loss_fn (which is nn.MSELoss() in your case) compares the model's predictions (y_pred) with the actual target values (y) and quantifies how 'wrong' the predictions are.

👉loss.backward(): This is the backward pass. It computes the gradient of the loss with respect to every learnable parameter in the model. These gradients indicate how much each parameter needs to change to reduce the loss.

👉opt.step(): This performs an optimization step. The optimizer (opt, which is Adam) uses the gradients computed during the backward pass to update the model's parameters (weights and biases), moving them slightly in the direction that minimizes the loss.

Next line of code takes the input data X and passes it through the trained neural network model. This is known as a forward pass, where the model processes the input through all its layers to generate an output. The print() function then displays these outputs, which are the model's predictions for the given input data X.

✅OUTPUT :
        tensor([[ 0.0000e+00],
        [ 1.0000e+00],
        [ 1.0000e+00],
        [-5.9605e-08]], grad_fn=<AddmmBackward0>)



