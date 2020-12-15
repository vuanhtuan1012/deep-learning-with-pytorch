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
	- Linear Regression:
	<p align="center"><img src="https://render.githubusercontent.com/render/math?math=Cost%28%5Cwidehat%7By%7D%2C%20y%29%20%3D%20%5Cfrac%7B1%7D%7B2%7D%28%5Cwidehat%7By%7D-y%29%5E2"></p>

-
	- Logistic Regression:
	<p align="center"><img src="https://render.githubusercontent.com/render/math?math=Cost%28%5Cwidehat%7By%7D%2C%20y%29%20%3D%20-ylog%5Cwidehat%7By%7D%20-%20%281-y%29log%281%20-%20%5Cwidehat%7By%7D%29"></p>
- <img src="https://render.githubusercontent.com/render/math?mode=inline&math=%5Cwidehat%7By%7D%20%3D%20h_%5Ctheta%28x%29"> : predicted value.
- <img src="https://render.githubusercontent.com/render/math?mode=inline&math=h_%5Ctheta%28x%29"> : hypothesis function. Its formula depends on the type of problem.
	- Linear Regression:
	<p align="center"><img src="https://render.githubusercontent.com/render/math?math=h_%5Ctheta%28x%29%20%3D%20%5CTheta%5ETX"></p>

-
	- Logistic Regression:
	<p align="center"><img src="https://render.githubusercontent.com/render/math?math=h_%5Ctheta%28x%29%20%3D%20%5Cfrac%7B1%7D%7B1%20%2B%20e%5E%7B-%5CTheta%5ETX%7D%7D"></p>