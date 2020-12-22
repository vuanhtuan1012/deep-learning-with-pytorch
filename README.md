# Deep Learning with PyTorch: Zero to GANs

## Syllabus

1. **PyTorch Basics and Gradient Descent**
    - PyTorch basics: tensors, gradients, and autograd
    - Linear regression & gradient descent from scratch
    - Using PyTorch modules: nn.Linear & nn.functional
2. **Working with Images and Logistic Regression**
    - Training-validation split on the MNIST dataset
    - Logistic regression, softmax & cross-entropy
    - Model training, evaluation & sample predictions
    - **Assignment 2** - Train Your First Model
3. **Training Deep Neural Networks on a GPU**
    - Multilayer neural networks using nn.Module
    - Activation functions, non-linearity & backprop
    - Training models faster using cloud GPUs
    - **Assignment 3** - Feed Forward Neural Networks
4. **Image Classification with Convolutional Neural Networks**
    - Working with 3-channel RGB images
    - Convolutions, kernels & features maps
    - Training curve, underfitting & overfitting
    - **Course Project** - Train a Deep Learning Model from Scratch
5. **Data Augmentation, Regularization, and ResNets**
    - Adding residual layers with batchnorm to CNNs
    - Learning rate annealing, weight decay & more
    - Training a state-of-the-art model in 5 minutes
6. **Image Generation using Generative Adversarial Networks (GANs)**
    - Generative modeling and applications of GANs
    - Training generator and discriminator networks
    - Generating fake digits & anime faces with GANs
    
## Write Math Formula

```
<img src="https://render.githubusercontent.com/render/math?mode=inline&math=your_url_encoded_formula>
```

- Online Latex Equation Editor: https://www.codecogs.com/latex/eqneditor.php
- URL Encode & Decode: https://www.urlencoder.org/

Example:

```
Given the training set <img src="https://render.githubusercontent.com/render/math?math=%28x%5E%7B%28i%29%7D%2Cy%5E%7B%28i%29%7D%29_%7Bi%3D1%7D%5Em">
```

Given the training set <img src="https://render.githubusercontent.com/render/math?math=%28x%5E%7B%28i%29%7D%2Cy%5E%7B%28i%29%7D%29_%7Bi%3D1%7D%5Em">

## PyTorch Basic

### Tensor
- Create a tensor
```Python
t1 = torch.tensor(4.)  # single number
t2 = torch.tensor([1, 2, 3, 4])  # vector
t3 = torch.tensor([[5, 6]
                   [7, 8],
                   [9, 10]
                 ])  # matrix
```
- Tensor attributes:
    - `t.dtype` : the type of a tensor like `float32`, `double64`, etc.
    - `t.shape` : the size of a tensor like `torch.Size([4])`, `torch.Size([3, 2])`

### Tensor operations and gradients

1. Operations
```Python
x = torch.tensor(3.)
w = torch.tensor(4., requires_grad = True)
b = torch.tensor(5., requires_grad = True)
y = w*x + b
```
- `requires_grad = True` to set that we will compute <img src="https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20w%7Dy" align="middle"> and <img src="https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20b%7Dy" align="middle">

2. Compute gradients
```Python
y.backward()
print('dy/dx = ', x.grad)
print('dy/dw = ', w.grad)
print('dy/db = ', b.grad)
```
- `y.backward()` : computes the derivatives of `y` with respect to the input tensors `x, w, b`.
- the tensor attribute `grad` stores the derivative of `y` of the respective tensors.
    - `x.grad` stores <img src="https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20x%7Dy" align="middle">. In this case, it is `None` since `x` doesn't have `requires_grad = True`
    - `w.grad` stores <img src="https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20w%7Dy" align="middle">. In this case, it is the value of `x`, `tensor(3.)`.
    - `b.grad` stores <img src="https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20b%7Dy" align="middle">. In this case, it is `tensor(1.)`

3. Interoperability with Numpy
- Convert a Numpy array to a PyTorch tensor, using `torch.from_numpy`
```Python
x = np.array([[1, 2],
              [3, 4]
            ])
y = torch.from_numpy(x)
type(x), type(y)
```
- Convert a tensor to a numpy, using the method `numpy`
```Python
z = y.numpy()
type(z)
```

## Linear Regression

This section mentions how to train a linear regression model in PyTorch in two ways:
- from scratch, functions are built manually.
- using PyTorch built-ins function.


### Linear Regression from scratch

#### Workflow
<p align="center">
<img src="images/linear_regression_from_scratch.svg"/>
</p>

#### Convert inputs & targets to tensors
```Python
X = torch.from_numpy(inputs)
Y = torch.from_numpy(targets)
```

#### Initialize parameters
```Python
# get number of samples (m) and features (n)
m, n = X.shape

# get number of outputs
_, a = Y.shape

# initialize parameters
W = torch.randn(a, n, requires_grad=True)  # weights
b = torch.randn(a, requires_grad=True)  # bias
```
- the function `torch.randn()` creates a tensor with the given shape with elements picked from a *normal distribution* with *mean 0* and *standard deviation 1*.
- In steps after we will optimize parameters `W, b` by using gradient descent, so we will need to compute `dy/dW` and `dy/db`. That's why `W` and `b` are set `requires_grad=True`.

#### Define functions
##### Hypothesis function (model)
- predicts `y` from `x` and parameters `W, b`.
```Python
def model(X, W, b):
    Y_hat = X @ W.t() + b
    return Y_hat
```
 - the operator `@` indicates that we want to do matrix multiplication.
 - the method `t()` returns the transpose of a tensor.

##### Cost function (loss function)
- computes the difference between predicted values `Y_hat` and output values `Y`.
```Python
def cost_fn(Y_hat, Y):
    diff = Y_hat - Y
    return torch.sum(diff * diff)/diff.numel()
```
 - the function `torch.sum()`  returns the sum of all the elements in a tensor.
 - the method `numel()` returns the number of elements in a tensor.

#### Train the model using gradient descent
```Python
epochs = 100  # number of iteration
lr = 1e-5  # learning rate
for i in range(epochs):
    Y_hat = model(X, W, b)
    cost = cost_fn(Y_hat, Y)
    cost.backward()  # compute derivatives
    # update parameters
    with torch.no_grad():
        W -= W.grad * lr
        b -= b.grad * lr
        W.grad.zero_()
        b.grad.zero_()
```
- **Algorithm:** we repeat the process of adjusting the weights and biases using the gradients multiple times to reduce the loss.
- Each iteration is called an **epoch**.
- the method `cost.backward()` computes the derivatives of `cost` with respect to `W` and `b`.
- the function `torch.no_grad()` indicates to PyTorch that we shouldn't track, calculate, or modify gradients while updating parameters `W` and `b`
- the method `grad.zero_()` resets the gradients to zero. As PyTorch accumulates gradients, we need to reset them before the next time we invoke  `backward()` on the loss.

#### Predict
```Python
x = torch.tensor([[75, 63, 44.]])
y_hat = model(x)
print(y_hat.tolist())
```
- the method `tolist()` converts a vector tensor in list
- the method `item()` returns the value of a single tensor.

### Linear Regression using PyTorch built-ins

#### Workflow
<p align="center">
<img src="images/linear_regression_pytorch_built_ins.svg"/>
</p>

#### Libraries
```Python
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
```

#### Convert inputs & targets to tensors
```Python
X = torch.from_numpy(inputs)
Y = torch.from_numpy(targets)
```

#### Define dataset & data loader
```Python
dataset = TensorDataset(X, Y)
batch_size = 5
dataloader = DataLoader(dataset, batch_size, shuffle=True)
```
- `TensorDataset` returns a tuple of two elements in which the first one contains the inputs and the second one contains the outputs.
    - Allow to access a small section of the dataset using the array indexing notation.
- `DataLoader` splits the dataset into batches of a predefined size while training.
    - `batch_size` indicates how many samples in a batch. For example, if `dataset` contains `15` samples and `batch_size = 5`, `dataloader` will point to `3` batches, each batch contain `5` samples.
    - `shuffle=True` means the dataset will be shuffled before creating batches. It helps randomize the input to the optimization algorithm, leading to a faster reduction in the loss.
    - Access the elements of data loader by using `for` loop.
```Python
for batch in dataloader:
    print(batch)
    xs, ys = batch
    print(xs); print(ys)
```
- **The idea of data loader** is that if the dataset is too big it takes time to train the whole dataset multiple times. Therefore, instead of training whole dataset, we devide the dataset into batches and at each batch iteraton (`for batch in dataloader`), we only train samples in one batch. We need some (`len(dataset)/batch_size`) iterations to train the whole dataset.

#### Define functions
##### Hypothesis function (model)
```Python
# get number of samples (m) and of features (n)
m, n = X.shape

# get number of outputs
_, a = Y.shape

# define hypothesis function
model = nn.Linear(n, a)

print(model.weight)
print(model.bias)
print(list(model.parameters()))
```
- Model attributes `weight` and `bias` contains the weights and bias of a model.
- the method `parameters()` return a generator of a list containing the weights and bias of a model.
##### Cost function (loss function)
```Python
cost_fn = F.mse_loss
```
- the function `mse_loss()` measures the element-wise mean squared error. It takes two obligatory inputs: *input* and *target* ([more detail](https://pytorch.org/docs/stable/nn.functional.html#mse-loss)).

#### Define optimizer
```Python
opt = torch.optim.SGD(model.parameters(), lr=1e-5)
```
- the function `torch.optim.SGD` optimizes parameters which are passed in `model.parameters()` with the learning rate passed in the parameter `lr`.
- `SGD` stands for *stochastic gradient descent*. The terms *stochastic* indicates that samples are selected in random batches instead of as a single group.

#### Train the model
```Python
def fit(epochs, model, cost_fn, opt, dataloader):
    for i in range(epochs):
        for batch in dataloader:
            xs, ys = batch
            ys_hat = model(xs)
            cost = cost_fn(ys_hat, ys)
            cost.backward()
            opt.step()  # update parameters
            opt.zero_grad()  # reset gradients to zero

fit(100, model, cost_fn, opt, dataloader)
```
- the optimizer method `step()` updates parameters (weights and bias).
- the optimizer method `zero_grad()` resets the gradients to zero.

#### Predict
```Python
x = torch.tensor([[75, 63, 44.]])
y_hat = model(x)
print(y_hat.tolist())
```
