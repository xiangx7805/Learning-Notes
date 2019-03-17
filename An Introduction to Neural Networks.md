# :feet:A simple and quick guideline:

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

Let’s use the network pictured above and assume all neurons have the same weights *w*=[0,1], the same bias *b*=0, and the same sigmoid activation function. Let ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20h_1%2C%20h_2%20%2C%20o_1) denote the outputs of the neurons they represent.

If we pass in the input x=[2,3]:  
![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20h_1%20%3D%20h_2%20%3D%20f%28w%5Ccdot%20x%20&plus;%20b%29%20%3D%20f%280*2%20&plus;%201*3%20&plus;0%29%20%3D%200.9526)  
![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20o_1%20%3D%20f%28%20w*%5Bh_1%20%2Ch_2%5D&plus;b%29%20%3Df%280*h_1%20&plus;1*h2%20&plus;0%29%20%3D%20f%280.9526%29%20%3D%200.7216)

Thus, the output of the neural network for input x=[2,3] is 0.7216.

 >A neural network can have **any number of layers** with **any number of neurons** in those layers.   
 The basic idea stays the same: feed the input(s) forward through the neurons in the network to get the output(s) at the end.

 For simplicity, we’ll keep using the network pictured above for the rest.

## Coding a Neuron in Python :dancer:

```python

# follow the previous section

class OurNeuralNetwork:
  '''
  A neural network with:
    - 2 inputs
    - a hidden layer with 2 neuros (h1,h2)
    - an output layer with  neuron (o1)
  Each neuron has the same weights and bias:
    - w = [0,1]
    - b = 0  
  '''
  def __init__(self):
    weight = np.array([0,1])
    bias = 0

    # The Neuron class here is from the previous section
    self.h1 = Neuron(weights, bias)
    self.h2 = Neuron(weights, bias)
    self.o1 = Neuron(weights, bias)

  def feedforward(self, x):
    out_h1 = self.h1.feedforward(x)
    out_h2 = self.h2.feedforward(x)

    # The inputs for o1 are the outputs from h1 and h2
    out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

    return out_o1

# test
network = OurNeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x)) # 0.7216325609518421

```

# 3. Training a Neural Network, Part 1 :evergreen_tree:

Let's start with another example. Say we have the following measurements:

Name | Weight (lb) |	Height (in) |	Gender
------------ | ------------- | ------------ | -------------
Alice	|133 |65 |	F
Bob	|160 |72 |	M
Charlie |152	|70 |	M
Diana	|120	|60 |	F

Let’s train our network to predict someone’s gender given their weight and height:
![](https://victorzhou.com/media/neural-network-post/network2.svg)

Make some data transformation :  
(*Normally, shift by the mean. Here choose 135 & 66 just to make numbers look nice.*)

Name | Weight (minus 135) |	Height (minus 66) |	Gender
------------ | ------------- | ------------ | -------------
Alice	|-2 |-1 |	1
Bob	|25 |6 | 0
Charlie |17	|4 |	0
Diana	|-15	|-6 |	1

## Loss :droplet:

**Loss:** Before training the network, we first need a way to quantify how “good” it’s doing so that it can try to do “better”.

We’ll use the **mean squared error** (MSE) loss:   ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20%5Clarge%20MSE%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum%20%5En_%7Bi%3D1%7D%20%28y_%7Btrue%7D%20-y_%7Bpred%7D%20%29%5E%7B%5E%7B2%7D%7D)

  where

  *  *n* is the number of samples, which is 4 (Alice, Bob, Charlie, Diana).
  *  *y* represents the variable being predicted, which is Gender.
  *  ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20y_%7Btrue%7D) is the true value of the variable (the “correct answer”). For example,![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20y_%7Btrue%7D) for Alice would be 1 (Female).
  *  ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20y_%7Bpred%7D) is the predicted value of the variable. It’s whatever our network outputs.

![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20%28y_%7Btrue%7D%20-y_%7Bpred%7D%20%29%5E%7B%5E%7B2%7D%7D) is known as the **squared error**. Our loss function is simply taking the average over all squared errors (hence the name mean squared error). The better our predictions are, the lower our loss will be!

Better predictions = Lower loss.

:sunglasses:**Training a network = trying to minimize its loss.**

## An Example Loss Calculation :sweat_drops:
Let’s say our network always outputs 00 - in other words, it’s confident all humans are Male 🤔. What would our loss be?

Name | ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20y_%7Btrue%7D) |	![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20y_%7Bpred%7D)  |	![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20%28y_%7Btrue%7D%20-y_%7Bpred%7D%20%29%5E%7B%5E%7B2%7D%7D)
------------ | ------------- | ------------ | -------------
Alice	|1 |0 |	1
Bob	|0 |0 | 0
Charlie |0	|0 |	0
Diana	|1	|0 |	1

Thus, MSE = (1+0+0+1) / 4 = 0.5


## Code: MSE Loss :dash:

```python
def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred) ** 2).mean()

# test
y_true = np.array([1, 0, 0, 1])
y_pred = np.array([0, 0, 0, 0])

print(mse_loss(y_true, y_pred)) # 0.5
```

# 4. Training a Neural Network, Part 2 :deciduous_tree:

Now the goal: **minimize the loss** of the neural network.

:question:We can change the network’s weights and biases to influence its predictions, but how do we do so in a way that decreases loss?

For simplicity, pretend we only have Alice in our dataset:

Name | Weight (minus 135) |	Height (minus 66) |	Gender
------------ | ------------- | ------------ | -------------
Alice	|-2 |-1 |	1

:speech_balloon:Then the mean squared error loss is just Alice’s squared error: ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20%5Clarge%20MSE%20%3D%20%5Cfrac%7B1%7D%7B1%7D%20%5Csum%20%5E1_%7Bi%3D1%7D%20%28y_%7Btrue%7D%20-y_%7Bpred%7D%20%29%5E%7B%5E%7B2%7D%7D%20%3D%281%20-y_%7Bpred%7D%20%29%5E%7B%5E%7B2%7D%7D)

:thought_balloon:Another way to think about loss is as a function of weights and biases. Let’s label each weight and bias in our network:  
![](https://victorzhou.com/media/neural-network-post/network3.svg)  
Then, we can write loss as a multivariable function:
![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20%5Clarge%20L%28w_1%2Cw_2%2Cw_3%2Cw_4%2Cw_5%2Cw_6%2Cb_1%2Cb_2%2Cb_3%29)

Imagine we wanted to tweak ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20%5Clarge%20w_1). How would loss *L* change if we changed ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20%5Clarge%20w_1)? :arrow_right: That’s a question the partial derivative ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_jvn%20%5Clarge%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20w_1%7D) can answer.

*Here gloss the math part, can check it in the original post.*

>The system of calculating partial derivatives by working backwards is known as **backpropagation**, or “backprop”.

Through **backpropagation** we have : ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Clarge%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20w_1%7D%20%3D%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20y_%7Bpred%7D%7D%20*%20%5Cfrac%7B%5Cpartial%20%7Bpred%7D%7D%7B%5Cpartial%20h_1%7D%20*%20%5Cfrac%7B%5Cpartial%20h_1%7D%7B%5Cpartial%20w_1%7D),
where ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Clarge%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20y_%7Bpred%7D%7D%20%3D%20%5Cfrac%7B%5Cpartial%20%281-y_%7Bpred%7D%29%5E2%7D%7B%5Cpartial%20y_%7Bpred%7D%7D%20%3D%20-2%281-y_%7Bpred%7D%29),  
      ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Clarge%20%5Cfrac%7B%5Cpartial%20y_%7Bpred%7D%7D%7B%5Cpartial%20h_1%7D%20%3D%20%5Cfrac%7B%5Cpartial%20o_1%7D%7B%5Cpartial%20h_1%7D%20%3D%20%5Cfrac%7B%5Cpartial%20f%28w_5h_1&plus;w_6h_2&plus;b_3%29%7D%7B%5Cpartial%20h_1%7D%20%3D%20w_5%20*f%27%28w_5h_1&plus;w_6h_2&plus;b_3%29),  
      ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Clarge%20%5Cfrac%7B%5Cpartial%20h_1%7D%7B%5Cpartial%20w_1%7D%20%3D%20%5Cfrac%7B%5Cpartial%20f%28w_1x_1%20&plus;%20w_2x_2&plus;b_1%29%7D%7B%5Cpartial%20w_1%7D%20%3D%20x_1f%27%28w_1x_1%20&plus;%20w_2x_2&plus;b_1%29),   
      ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Clarge%20x_1%20%2Cx_2) are weight and height,  
      ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Clarge%20f%27%28x%29%20%3D%20%5Cleft%20%28%20%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-x%7D%7D%20%5Cright%20%29%27%20%3D%20%5Cfrac%7Be%5E%7B-x%7D%7D%7B%281&plus;e%5E%7B-x%7D%29%5E2%7D%20%3D%20f%28x%29*%5Cleft%20%28%201-f%28x%29%20%5Cright%20%29).


## Example: Calculating the Partial Derivative  :star2:

Name | Weight (minus 135) |	Height (minus 66) |	Gender
------------ | ------------- | ------------ | -------------
Alice	|-2 |-1 |	1

And initialize all the weights to 1 and all the biases to 0.

We have ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Clarge%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20w_1%7D%20%3D%200.0214)

This tells us that if we were to increase ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Clarge%20w_1), L would increase a tiiiny bit as a result.

## Training: Stochastic Gradient Descent (SGD):sparkles:
**We have all the tools we need to train a neural network now!**

Next we’ll use an optimization algorithm called [stochastic gradient descent (SGD)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) that tells us how to change our weights and biases to minimize loss. It’s basically just this update equation:  
![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Clarge%20w_1%5Cleftarrow%20w_1%20-%20%5Ceta%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20w_1%7D)

> **η** is a constant called the learning rate that controls how fast we train.

All we’re doing is subtracting ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Clarge%20%5Ceta%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20w_1%7D) from ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Clarge%20w_1):  
*  If ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Clarge%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20w_1%7D) is positive, ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Clarge%20w_1) will decrease, which makes *L* decrease.  
*  If ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Clarge%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20w_1%7D) is negative, ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Clarge%20w_1) will increase, which makes *L* decrease.

If we do this for every weight and bias in the network, the loss will slowly decrease and our network will improve.

Our training process will look like this:

1.  Choose **one** sample from our dataset. This is what makes it **stochastic gradient descent** - we only operate on one sample at a time.  
2.  Calculate all the partial derivatives of loss with respect to weights or biases(e.g.![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Clarge%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20w_1%7D),![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cm%20%5Clarge%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20w_2%7D),etc)  
3.  Use the update equation to update each weight and bias.  
4.  Go back to step 1.


## Code: A Complete Neural Network :dizzy:
Implement a Complete neural network:

Name | Weight (minus 135) |	Height (minus 66) |	Gender
------------ | ------------- | ------------ | -------------
Alice	|-2 |-1 |	1
Bob	|25 |6 | 0
Charlie |17	|4 |	0
Diana	|-15	|-6 |	1

![](https://victorzhou.com/media/neural-network-post/network3.svg)  

```python
import numpy as np

def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:
  '''
  A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)

  *** DISCLAIMER ***:
  The code below is intended to be simple and educational, NOT optimal.
  Real neural net code looks nothing like this. DO NOT use this code.
  Instead, read/run it to understand how this specific network works.
  '''
  def __init__(self):
    # Weights
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()

    # Biases
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()

  def feedforward(self, x):
    # x is a numpy array with 2 elements.
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    return o1

  def train(self, data, all_y_trues):
    '''
    - data is a (n x 2) numpy array, n = # of samples in the dataset.
    - all_y_trues is a numpy array with n elements.
      Elements in all_y_trues correspond to those in data.
    '''
    learn_rate = 0.1
    epochs = 1000 # number of times to loop through the entire dataset

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        # --- Do a feedforward (we'll need these values later)
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = sigmoid(sum_h1)

        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = sigmoid(sum_h2)

        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
        o1 = sigmoid(sum_o1)
        y_pred = o1

        # --- Calculate partial derivatives.
        # --- Naming: d_L_d_w1 represents "partial L / partial w1"
        d_L_d_ypred = -2 * (y_true - y_pred)

        # Neuron o1
        d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
        d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
        d_ypred_d_b3 = deriv_sigmoid(sum_o1)

        d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
        d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

        # Neuron h1
        d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
        d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
        d_h1_d_b1 = deriv_sigmoid(sum_h1)

        # Neuron h2
        d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
        d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
        d_h2_d_b2 = deriv_sigmoid(sum_h2)

        # --- Update weights and biases
        # Neuron h1
        self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
        self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
        self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

        # Neuron h2
        self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
        self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
        self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

        # Neuron o1
        self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
        self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
        self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

      # --- Calculate total loss at the end of each epoch
      if epoch % 10 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
        print("Epoch %d loss: %.3f" % (epoch, loss))

# Define dataset
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# Train our neural network!
network = OurNeuralNetwork()
network.train(data, all_y_trues)
```

*Complete code also available on [Github](https://github.com/vzhou842/neural-network-from-scratch).*

Our loss steadily decreases as the network learns:
![](https://victorzhou.com/media/neural-network-post/loss.png)

Now use the network to predict genders:

```python
# Make some predictions
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - M
```

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
