:feet:A simple and quick guideline: 

*  Introduced **neurons**, the building blocks of neural networks.
*  Used the **sigmoid activation function** in our neurons.
*  Saw that neural networks are just neurons connected together.
*  Created a dataset with Weight and Height as inputs (or **features**) and Gender as the output (or **label**).
*  Learned about **loss functions** and the **mean squared error** (MSE) loss.
*  Realized that training a network is just minimizing its loss.
*  Used **backpropagation** to calculate partial derivatives.
*  Used **stochastic gradient descent** (SGD) to train our network.


# 1. Building Blocks: Neurons :seedling:

**Neurons** : the basic unit of a neural network. **A neuron takes inputs, does some math with them, and produces one output.** Here’s what a 2-input neuron looks like:  
![image of a 2-input neuron](https://victorzhou.com/media/neural-network-post/perceptron.svg)

3 things are happening here.   
*  First, each input is multiplied by a weight::apple:  
![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20x_1%5Crightarrow%20x_1*w_1)  
![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20x_2%5Crightarrow%20x_2*w_2)

*  Next, all the weighted inputs are added together with a bias *b*: :green_apple:  
![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20%28x_1*w_1%20%29%20&plus;%20%28x_2*w_2%20%29%20&plus;%20b)

*  Finally, the sum is passed through an activation function::tangerine:  
![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20y%3D%20f%28x_1*w_1%20&plus;%20x_2*w_2%20&plus;%20b%29)

The activation function is used to turn an unbounded input into an output that has a nice, predictable form. A commonly used activation function is the [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) function:  
![image of sigmoid function](https://victorzhou.com/media/neural-network-post/sigmoid.png)

>The **sigmoid function** only outputs numbers in the range (0, 1). You can think of it as compressing (−∞,+∞) to (0, 1) - big negative numbers become ~0, and big positive numbers become ~1.  
Often, sigmoid function refers to the special case of the **logistic function**.

:raising_hand:This process of passing inputs forward to get an output is known as **feedforward.**

## A Simple Example :musical_note:
Assume we have a 2-input neuron that uses the sigmoid activation function and has the following parameters:  
![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20w%3D%20%5Bw_1%2C%20w_2%5D%3D%5B0%2C1%5D%20%2C%7E%7E%20b%3D4)

Now, let’s give the neuron an input of *x* = [2,3]. We’ll use the dot product to write things more concisely:  
![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20y%20%3D%20f%5Cleft%20%28%20%7E%28w*x%29&plus;b%20%5Cright%20%29%20%3D%20w_1*x_1%20&plus;%20w_2*x_2%20&plus;%20b%3D%200.999)

Thus, the neuron outputs 0.999 given the inputs x=[2,3].

## Coding a Neuron in Python :notes:

```Python
import numpy as np

def sigmoid(x):
  #our activation function : f(x) = 1 / (1+e^(-x))
  return 1 / (1+e^(-x))

class Neuron:
  def __init__(self,weights, bias):
    self.weights = weights
    self.bias = bias

  def feedforward(self, inputs):
    # weights inputs, add bias, then use the activation function
    total = np.dot(self.weights, inputs) + self.bias
    return sigmoid(total)

# apply the Example
weights = np.array([0,1]) # w1 =0, w2 =1
bias = 4                  # b = 4
n= Neuron(weights,bias)

x= np.array([2,3])      # x1 = 2, x2 = 3
print(n.feedforward(x)) # 0.9990889488055994
```

# 2. Combining Neurons into a Neural Network :herb:

A neural network is nothing more than a bunch of neurons connected together. Here’s what a simple neural network might look like:  
![](https://victorzhou.com/media/neural-network-post/network.svg)

This network has 2 inputs, a hidden layer with 2 neurons ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20h_1%20%7E%20and%20%7E%20h_2) and an output layer with one neuron ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20o_1). Notice that the inputs for ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20o_1) are the outputs from ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20h_1%20%7E%20and%20%7E%20h_2)- that’s what makes this a network.

> A **hidden layer** is any layer between the input (first) layer and output (last) layer. There can be multiple hidden layers!

## An Example: Feedforward :running:
## Coding a Neuron in Python :dancer:

# 3. Training a Neural Network, Part 1 :evergreen_tree:

## Loss :droplet:
## An Example Loss Calculation :sweat_drops:
## Code: MSE Loss :dash:

# 4. Training a Neural Network, Part 2 :deciduous_tree:
## Example: Calculating the Partial Derivative  :star2:
## Training: Stochastic Gradient Descent :sparkles:
## Code: A Complete Neural Network :dizzy:

# Now What? :christmas_tree:

There’s still much more to do:

*  Experiment with bigger / better neural networks using proper machine learning libraries like [Tensorflow](https://www.tensorflow.org/), [Keras](https://keras.io/), and [PyTorch](https://pytorch.org/).
*  Tinker with [a neural network in your browser](https://playground.tensorflow.org/).
*  Discover [other activation functions](https://keras.io/activations/) besides sigmoid.
*  Discover [other optimizers](https://keras.io/optimizers/) besides SGD.
*  Learn about [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network), which revolutionized the field of Computer Vision.
*  Learn about [Recurrent Neural Networks](https://en.wikipedia.org/wiki/Recurrent_neural_network), often used for Natural Language Processing (NLP).

# Article Source :tanabata_tree:
[*Victor Zhou*'s Blog : Machine Learning for Beginners: An Introduction to Neural Networks](https://victorzhou.com/blog/intro-to-neural-networks/)
