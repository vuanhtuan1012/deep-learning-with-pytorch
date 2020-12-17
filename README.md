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
```
x = np.array([[1, 2],
              [3, 4]
            ])
y = torch.from_numpy(x)
type(x), type(y)
```
- Convert a tensor to a numpy, using the method `numpy`
```
z = y.numpy()
type(z)
```