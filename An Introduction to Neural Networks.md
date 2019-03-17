# Building Blocks: Neurons

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

>The sigmoid function only outputs numbers in the range (0, 1). You can think of it as compressing (-\infty, +\infty)(−∞,+∞) to (0, 1) - big negative numbers become ~0, and big positive numbers become ~1.
















# Article Source :information_desk_person:
[*Victor Zhou*'s Blog : Machine Learning for Beginners: An Introduction to Neural Networks](https://victorzhou.com/blog/intro-to-neural-networks/)
