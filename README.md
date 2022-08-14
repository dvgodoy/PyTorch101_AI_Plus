# PyTorch 101: Building a Model Step-by-Step

Learn the basics of building a PyTorch model using a structured, incremental and from first principles approach. Find out why PyTorch is the fastest growing Deep Learning framework and how to make use of its capabilities: autograd, dynamic computation graph, model classes, data loaders and more.

The main goal of this training is to show you how PyTorch works: we will start with a simple and familiar example in Numpy and "torch" it! At the end of it, you should be able to understand PyTorch's key components and how to assemble them together into a working model.

This training covers the majority of the content in the first volume of my series of books, "Deep Learning with PyTorch Step-by-Step". You can find it on [Amazon](https://www.amazon.com/dp/B09QR4M768/) (paperback and Kindle) or [Gumroad](https://dvgodoy.gumroad.com/l/pytorch) (PDF).

## Learning Objectives

- Understand the basic building blocks of PyTorch: tensors, autograd, models, optimizers, losses, datasets, and data loaders
- Identify the basic steps of gradient descent, and how to use PyTorch to make each one of them more automatic
- Understand the relationship between the gradient descent algorithm, the loss, the learning rate, and feature scaling
- Build, train, and evaluate a model using mini-batch gradient descent

## Course Outline

- Module 0: Introduction
  - Introduction
  - Motivation
  - Agenda

- Module 1: PyTorch: tensors, tensors, tensors 
  - Introducing a simple and familiar example: linear regression
  - Generating synthetic data
  - Tensors: what they are and how to create them
  - CUDA: GPU vs CPU tensors
  - Parameters: tensors meet gradients
  - Quiz #1
  - Exercise #1

- Module 2: Gradient Descent in Five Easy Steps
  - Step 0: initializing parameters
  - Step 1: making predictions in the forward pass
  - Step 2: computing the loss, or “how bad is my model?”
  - Loss surface
  - Step 3: computing gradients, or “how to minimize the loss?”
  - Step 4: updating parameters
  - Exercise #2.1
  - Learning rate, the most important hyper-parameter
  - Exercise #2.2
  - Gradient Descent and the importance of feature scaling
  - Step 5: Rinse and repeat
  - Quiz #2
  - Exercise #2.3

- Module 3: Autograd, your companion for all your gradient needs!
  - Computing gradients automatically with the backward method
  - Dynamic Computation Graph: what is that?
  - Optimizers: updating parameters, the PyTorch way
  - Loss functions in PyTorch
  - Quiz #3
  - Exercise #3

- Module 4: Building a Model in PyTorch 
  - Your first custom model in PyTorch
  - Peeking inside a model with state dictionaries
  - The importance of setting a model to training mode
  - Nested models, layers, and sequential models
  - Organizing our code: the training step
  - Quiz #4
  - Exercise #4

- Module 5: Datasets and data loaders    
  - Your first custom dataset in PyTorch   
  - Data loaders and mini-batches    
  - Evaluation phase: setting up the stage   
  - Taking a break: saving and loading models
  - Quiz #5
  - Exercise #5
  - The StepByStep class

- Bonus Module: Are my data points separable? (if time allows)

## Setup Guide (Local Installation)

If you'd like to use a local environment, please follow these steps (assuming you use Anaconda):

- Install GraphViz: https://www.graphviz.org/

- Create a conda environment: `conda create -n pytorch101 pip conda python==3.8.5`

- Activate the conda environment: `conda activate pytorch101`

- Install PyTorch: https://pytorch.org/get-started/locally/

- Install other packages: `conda install scikit-learn==0.23.2 matplotlib==3.3.2 jupyter==1.0.0 ipywidgets==7.5.1 plotly==4.14.3 -c anaconda`

- Install torchviz: `pip install torchviz`

- Clone this repo: `git clone https://github.com/dvgodoy/PyTorch101_AI_Plus.git`

- Start Jupyter: `jupyter notebook`
