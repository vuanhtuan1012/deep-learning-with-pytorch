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
Given the training set <img src="https://render.githubusercontent.com/render/math?mode=inline&math=%28x%5E%7B%28i%29%7D%2Cy%5E%7B%28i%29%7D%29_%7Bi%3D1%7D%5Em">
```

Given the training set <img src="https://render.githubusercontent.com/render/math?mode=inline&math=%28x%5E%7B%28i%29%7D%2Cy%5E%7B%28i%29%7D%29_%7Bi%3D1%7D%5Em">

# Machine Learning

## Supervised Learning

### Problem

Given the training set <img src="https://render.githubusercontent.com/render/math?mode=inline&math=%28x%5E%7B%28i%29%7D%2Cy%5E%7B%28i%29%7D%29_%7Bi%3D1%7D%5Em">
- *m* : number of training set
- <img src="https://render.githubusercontent.com/render/math?mode=inline&math=X%20%3D%20%5Cleft%20%5C%7B%20x%5E%7B%28i%29%7D%20%5Cright%20%5C%7D_%7Bi%3D1%7D%5Em"> : input values
- <img src="https://render.githubusercontent.com/render/math?mode=inline&math=Y%20%3D%20%5Cleft%20%5C%7B%20x%5E%7B%28i%29%7D%20%5Cright%20%5C%7D_%7Bi%3D1%7D%5Em"> : output values

### Objective

Find the function to predict <img src="https://render.githubusercontent.com/render/math?mode=inline&math=Y"> from <img src="https://render.githubusercontent.com/render/math?mode=inline&math=X"> so that <img src="https://render.githubusercontent.com/render/math?mode=inline&math=J%28%5Ctheta%29"> is minimal.
- <img src="https://render.githubusercontent.com/render/math?mode=inline&math=J%28%5Ctheta%29"> measures the difference between predicted values <img src="https://render.githubusercontent.com/render/math?mode=inline&math=%5Cwidehat%7BY%7D"> and <img src="https://render.githubusercontent.com/render/math?mode=inline&math=Y">.
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=J%28%5Ctheta%29%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Em%20Cost%28%5Cwidehat%7By%7D%5E%7B%28i%29%7D%2Cy%5E%7B%28i%29%7D%29">
</p>

- The formula of <img src="https://render.githubusercontent.com/render/math?mode=inline&math=Cost%28%5Cwidehat%7By%7D%2Cy%29"> depends on the type of problem.
	- Linear Regression: <img src="https://render.githubusercontent.com/render/math?mode=inline&math=Cost%28%5Cwidehat%7By%7D%2C%20y%29%20%3D%20%5Cfrac%7B1%7D%7B2%7D%28%5Cwidehat%7By%7D-y%29%5E2">
	- Logistic Regression: <img src="https://render.githubusercontent.com/render/math?mode=inline&math=Cost%28%5Cwidehat%7By%7D%2C%20y%29%20%3D%20-ylog%5Cwidehat%7By%7D%20-%20%281-y%29log%281%20-%20%5Cwidehat%7By%7D%29">
- <img src="https://render.githubusercontent.com/render/math?mode=inline&math=%5Cwidehat%7By%7D%20%3D%20h_%5Ctheta%28x%29"> : predicted value.
- <img src="https://render.githubusercontent.com/render/math?mode=inline&math=h_%5Ctheta%28x%29"> : hypothesis function. Its formula depends on the type of problem.
	- Linear Regression:
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=h_%5Ctheta%28x%29%20%3D%20%5CTheta%5ETX">
</p>

	- Logistic Regression:
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=h_%5Ctheta%28x%29%20%3D%20%5Cfrac%7B1%7D%7B1%20%2B%20e%5E%7B-%5CTheta%5ETX%7D%7D">
</p>