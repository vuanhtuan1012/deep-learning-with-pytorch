# Deep Learning with PyTorch: Zero to GANs

- [Lesson 1](#lesson-1---pytorch-basics-and-gradient-descent): PyTorch Basics and Gradient Descent
- [Assignment 1](#assignment-1---all-about-torchtensor): All About torch.Tensor
- [Lesson 2](#lesson-2---working-with-images-and-logistic-regression): Working with Images and Logistic Regression
- [Assignment 2](#assignment-2---train-your-first-model): Train Your First Model
- [Lesson 3](#lesson-3---training-deep-neural-networks-on-a-gpu): Training Deep Neural Networks on a GPU
- [Assignment 3](#assignment-3---feed-forward-neural-networks): Feed Forward Neural Networks
- [Lesson 4](#lesson-4---image-classification-with-convolutional-neural-networks): Image Classification with Convolutional Neural Networks
- [Lesson 5](#lesson-5---data-augmentation-regularization--resnets): Data Augmentation, Regularization & ResNets
- [Lesson 6](#lesson-6---generative-adversarial-networks-and-transfer-learning): Generative Adversarial Networks and Transfer Learning
- [Course Project](#course-project---train-a-deep-learning-model-from-scratch): Train a Deep Learning Model from Scratch

## Course Project - Train a Deep Learning Model from Scratch

For the course project, you will pick a dataset of your choice and apply the concepts learned in this course to train deep learning models end-to-end with PyTorch, experimenting with different hyperparameters & metrics.

Find a dataset online (see the [Where to Find Datasets](#where-to-find-datasets) section below)

- Understand and describe the modeling objective clearly
- What type of data is it? (images, text, audio, etc.)
- What type of problem is it? (regression, classification, generative modeling, etc.)
- Clean the data if required and perform exploratory analysis (plot graphs, ask questions)
- Modeling
    - Define a model (network architecture)
    - Pick some hyperparameters
    - Train the model
    - Make predictions on samples
    - Evaluate on the test dataset
    - Save the model weights
    - Record the metrics
    - Try different hyperparameters & regularization
- Conclusions - summarize your learning & identify opportunities for future work
- Publish and submit your Jupyter notebook
- (Optional) Write a blog post to describe your experiments and summarize your work. Use Medium or Github pages.

Note: There is no starter notebook for the course project. Please use the "New Notebook" button on Jovian to create a new notebook, "Run on Colab" to execute it, and "jovian.commit" to record versions.

### Example notebooks for reference:

- https://jovian.ai/aakashns/simple-cnn-starter
- https://jovian.ai/aakashns/transfer-learning-pytorch
- https://jovian.ai/aakashns/06b-anime-dcgan
- https://jovian.ai/aakashns/05b-cifar10-resnet

### Where to Find Datasets

General sources
- https://www.kaggle.com/datasets (use the opendatasets library for downloading datasets)
- https://course.fast.ai/datasets
- https://github.com/ChristosChristofidis/awesome-deep-learning#datasets
- https://www.kaggle.com/competitions (check the "Completed" tab)
- https://www.analyticsvidhya.com/blog/2018/03/comprehensive-collection-deep-learning-datasets/
- https://lionbridge.ai/datasets/top-10-image-classification-datasets-for-machine-learning/
- https://archive.ics.uci.edu/ml/index.php
- https://github.com/awesomedata/awesome-public-datasets
- https://datasetsearch.research.google.com/

Indian stocks data
- https://nsepy.xyz/
- https://nsetools.readthedocs.io/en/latest/usage.html
- https://www.kaggle.com/rohanrao/nifty50-stock-market-data

Indian Air Quality Data
- https://www.kaggle.com/rohanrao/air-quality-data-in-india

Indian Covid-19 Dataset
- https://api.covid19india.org/

World Covid-19 Dataset
- https://www.kaggle.com/imdevskp/corona-virus-report

USA Covid-19 Dataset
- https://www.kaggle.com/sudalairajkumar/covid19-in-usa

Megapixels Dataset for Face Detection, GANs, Human Localization
- https://megapixels.cc/datasets/ (Contains 7 different datasets)

Agriculture based dataset
- https://www.kaggle.com/srinivas1/agricuture-crops-production-in-india
- https://www.kaggle.com/unitednations/global-food-agriculture-statistics
- https://www.kaggle.com/kianwee/agricultural-raw-material-prices-19902020
- https://www.kaggle.com/jmullan/agricultural-land-values-19972017

India Digital Payments UPI
- https://www.kaggle.com/lazycipher/upi-usage-statistics-aug16-to-feb20

India Consumption of LPG
- https://community.data.gov.in/domestic-consumption-of-liquefied-petroleum-gas-from-2011-12-to-2017-18/

India Import/Export Crude OIl
- https://community.data.gov.in/total-import-v-s-export-of-crude-oil-petroleum-products-by-india-from-2011-12-to-2017-18/

US Unemployment Rate Data
- https://www.kaggle.com/jayrav13/unemployment-by-county-us

India Road accident Data
- https://community.data.gov.in/statistics-of-road-accidents-in-india/

Data science Jobs Data
- https://www.kaggle.com/sl6149/data-scientist-job-market-in-the-us
- https://www.kaggle.com/jonatancr/data-science-jobs-around-the-world
- https://www.kaggle.com/rkb0023/glassdoor-data-science-jobs

H1-b Visa Data
- https://www.kaggle.com/nsharan/h-1b-visa

Donald Trump’s Tweets
- https://www.kaggle.com/austinreese/trump-tweets

Hilary Clinton and Trump’s Tweets
- https://www.kaggle.com/benhamner/clinton-trump-tweets

Asteroid Dataset
- https://www.kaggle.com/sakhawat18/asteroid-dataset

Solar flares Data
- https://www.kaggle.com/khsamaha/solar-flares-rhessi

Human face generation GANs
- https://www.kaggle.com/arnaud58/flickrfaceshq-dataset-ffhq

F-1 Race Data
- https://www.kaggle.com/cjgdev/formula-1-race-data-19502017

Automobile Insurance
- https://www.kaggle.com/aashishjhamtani/automobile-insurance

PUBG
- https://www.kaggle.com/skihikingkevin/pubg-match-deaths?

CS GO
- https://www.kaggle.com/mateusdmachado/csgo-professional-matches
- https://www.kaggle.com/skihikingkevin/csgo-matchmaking-damage

Dota 2
- https://www.kaggle.com/devinanzelmo/dota-2-matches

Cricket
- https://www.kaggle.com/nowke9/ipldata
- https://www.kaggle.com/jaykay12/odi-cricket-matches-19712017

Basketball
- https://www.kaggle.com/ncaa/ncaa-basketball
- https://www.kaggle.com/drgilermo/nba-players-stats

Football
- https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017
- https://www.kaggle.com/abecklas/fifa-world-cup
- https://www.kaggle.com/egadharmawan/uefa-champion-league-final-all-season-19552019

## Lesson 6: Generative Adversarial Networks and Transfer Learning

https://www.youtube.com/watch?v=QpR4QEv5Urk

In this lesson, we apply the techniques covered in previous lessons for generating images for anime faces using Generative Adversarial Networks or GANs. Notebooks used in this tutorial:

- Generating anime faces using GANs:
    - On Jovian: https://jovian.ai/aakashns/06b-anime-dcgan
    - On repo: [06b-anime-dcgan](w6/06b-anime-dcgan.zip)
- Generating handwritten digits using GANs:
    - On Jovian: https://jovian.ai/aakashns/06-mnist-gan
    -  On repo: [06-mnist-gan](w6/06-mnist-gan.zip)
- Transfer Learning using pretrained models:
    - On Jovian: https://jovian.ai/aakashns/transfer-learning-pytorch
    - On repo: [transfer-learning-pytorch](w6/transfer-learning-pytorch.zip)
- Zero to GANs using Tensorflow:
    - On Jovian: https://jovian.ai/kartik.godawat/collections/tf-zero-to-gan
    - On repo: [01-tensorflow-basics](w6/01-tensorflow-basics.zip), [02-tf-linear-regression.zip](w6/02-tf-linear-regression.zip), [03-tf-logistic-regression.zip](w6/03-tf-logistic-regression.zip), [04-tf-feedforward-nn.zip](w6/04-tf-feedforward-nn.zip)

## Lesson 5 - Data Augmentation, Regularization & ResNets

https://www.youtube.com/watch?v=JN7-ZBFYSCU

This lesson covers some advanced techniques like data augmentation, regularization, and adding residual layers to convolutional neural networks. We train a state-of-the-art model from scratch in just five minutes. Notebooks used in this lesson:

- ResNets, Regularization & Data Augmentation:
    - On Jovian: https://jovian.ai/aakashns/05b-cifar10-resnet
    - On repo: [05b-cifar10-resnet.zip](w5/05b-cifar10-resnet.zip)
- Simple CNN Starter Notebook:
    - On Jovian: https://jovian.ai/aakashns/simple-cnn-starter
    - On repo: [simple-cnn-starter.zip](simple-cnn-starter.zip)
- Transfer Learning with CNNs:
    - On Jovian: https://jovian.ai/aakashns/transfer-learning-pytorch
    - On repo: [transfer-learning-pytorch.zip](w5/transfer-learning-pytorch.zip)
- Image Classification with CNNs:
    - On Jovian: https://jovian.ai/aakashns/05-cifar10-cnn
    - On repo: [05-cifar10-cnn.zip](w5/05-cifar10-cnn.zip)


## Lesson 4 - Image Classification with Convolutional Neural Networks

https://www.youtube.com/watch?v=d9QHNkD_Pos

This lesson introduces convolutional neural networks (CNNs) that are well-suited for image datasets and computer vision problems. We also cover underfitting, overfitting and techniques to improve model performance. Notebooks used in this lesson:

- Convolutional Neural Networks:
    - On Jovian: https://jovian.ai/aakashns/05-cifar10-cnn
    - On repo: [05-cifar10-cnn.zip](w4/05-cifar10-cnn.zip)
- CIFAR10 Feed Forward Model:
    - On Jovian: https://jovian.ai/aakashns/03-cifar10-feedforward
    - On repo: [03-cifar10-feedforward.zip](w4/03-cifar10-feedforward.zip)

## Assignment 3 - Feed Forward Neural Networks

The ability to try many different neural network architectures to address a problem is what makes deep learning really powerful, especially compared to shallow learning techniques like linear regression, logistic regression, etc. In this assignment, you will:
- Explore the CIFAR10 dataset
- Set up a training pipeline to train a neural network on a GPU
- Experiment with different network architectures & hyperparameters

Assignment Notebook:
- On Jovian: https://jovian.ai/aakashns/03-cifar10-feedforward
- On repo: [03-cifar10-feedforward-v-7.zip](w3/03-cifar10-feedforward-v-7.zip)

Use the starter notebook(s) to get started with the assignment. Read the problem statement, follow the instructions, add your solutions, and make a submission.

## Lesson 3 - Training Deep Neural Networks on a GPU

https://www.youtube.com/watch?v=Qn5DDQn0fx0

In this lesson, we learn how to build and train a deep neural network with hidden layers and non-linear activations using cloud-based GPUs. Notebooks used in this lesson:

- Training deep neural networks on a GPU:
    - On Jovian: https://jovian.ai/aakashns/04-feedforward-nn
    - On repo: [04-feedforward-nn.zip](w3/04-feedforward-nn.zip)
- Starter notebook for training a model:
    - On Jovian: https://jovian.ai/aakashns/fashion-feedforward-minimal
    - On repo: [fashion-feedforward-minimal.zip](w3/fashion-feedforward-minimal.zip)
- Data visualization cheatsheet:
    - On Jovian: https://jovian.ai/aakashns/dataviz-cheatsheet
    - On repo: [dataviz-cheatsheet.zip](w3/dataviz-cheatsheet.zip)

## Assignment 2 - Train Your First Model

In this assignment, you’re going to use information like a person’s age, sex, BMI, no. of children, and smoking habit to predict the price of yearly medical bills. You will train a model with the following steps:
- Download and explore the dataset
- Prepare the dataset for training
- Create a linear regression model
- Train the model to fit the data
- Make predictions using the trained model

Assignment Notebook:
- On Jovian: https://jovian.ai/aakashns/02-insurance-linear-regression
- On repo: [02-insurance-linear-regression.zip](w2/02-insurance-linear-regression.zip)

Use the starter notebook(s) to get started with the assignment. Read the problem statement, follow the instructions, add your solutions, and make a submission.

## Lesson 2 - Working with Images and Logistic Regression

https://www.youtube.com/watch?v=uuuzvOEC0zw

This lesson covers how to work with images from the MNIST dataset, creating training & validation datasets to avoid overfitting, and training a logistic regression model using the softmax activation & the cross-entropy loss. Notebooks used in this lesson:

- Logistic regression:
    - On Jovian: https://jovian.ai/aakashns/03-logistic-regression
    - On repo: [03-logistic-regression.zip](w2/03-logistic-regression.zip)
- Logistic regression starter:
    - On Jovian: https://jovian.ai/aakashns/mnist-logistic-minimal
    - On repo: [mnist-logistic-minimal.zip](w2/mnist-logistic-minimal.zip)
- Linear regression starter:
    - On Jovian: https://jovian.ai/aakashns/housing-linear-minimal
    - On repo: [housing-linear-minimal.zip](w2/housing-linear-minimal.zip)

## Assignment 1 - All About torch.Tensor

The objective of this assignment is to develop a solid understanding of PyTorch tensors. In this assignment you will:
- Pick 5 interesting functions related to PyTorch tensors by reading the documentation
- Create a Jupyter notebook using a starter template to illustrate their usage
- Upload and showcase your Jupyter notebook on your Jovian profile
- (optional) Write a blog post to accompany and showcase your Jupyter notebook.
- Share your work with the community and exchange feedback with other participants

Assignment Notebook:
- On Jovian: https://jovian.ai/aakashns/01-tensor-operations
- On repo: [01-tensor-operations.zip](w1/01-tensor-operations.zip)

Use the starter notebook(s) to get started with the assignment. Read the problem statement, follow the instructions, add your solutions, and make a submission.

## Lesson 1 - PyTorch Basics and Gradient Descent

https://www.youtube.com/watch?v=5ioMqzMRFgM

We start this lesson with the basics of PyTorch: tensors, gradients, and Autograd. We then proceed to implement linear regression and gradient descent from scratch, first using matrix operations and then using PyTorch built-ins. Notebooks used in this lesson:

- PyTorch Basics:
    - On Jovian: https://jovian.ai/aakashns/01-pytorch-basics
    - On repo: [01-pytorch-basics.zip](w1/01-pytorch-basics.zip)
- Linear Regression:
    - On Jovian: https://jovian.ai/aakashns/02-linear-regression
    - On repo: [02-linear-regression.zip](w1/02-linear-regression.zip)
- Machine Learning:
    - On Jovian: https://jovian.ai/aakashns/machine-learning-intro
    - On repo: [machine-learning-intro.zip](w1/machine-learning-intro.zip)
