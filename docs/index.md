---
layout: page
title: One LEGO at a time&#58; Explaining the Math of How Neural Networks Learn
tagline:
description: Tutorial on back-propagation
---

>A neural network is a clever arrangement of linear and non-linear modules. When we choose and connect them wisely,
we have a powerful tool to approximate any mathematical function. For example one that separates classes as a non-linear
decision boundary. A topic that is not always explained in depth, despite of its intuitive and modular nature, is the
back-propagation technique responsible for updating trainable parameters. Let’s explore this algorithm to see the internal
functioning of a neural network using LEGO pieces as a modular analogy, one brick at a time.

The below figure depicts some of the Math used for training a neural network. We will make sense of this during this article.
The reader may find interesting that a neural network is a stack of modules with different purposes:

- Input X feeds a neural network with raw data, which is stored in a matrix in which observations are rows and dimensions are columns
- Weights W1 map input X to the first hidden layer. Weights W1 is then a linear kernel
- A Sigmoid function prevents numbers from falling out of range by scaling them to 0-1. This results in the first hidden layer h1

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/overview.png "Logo Title Text 1")

At this point these operations only compute a general linear system and don’t have the capacity to model non-linear interactions.
This changes when we stack one more layer adding depth to this modular structure. The deeper the network, the more subtle non-linear
interactions can be learned, which may explain in part the rise of deep neural models.

## Why is this Important?

Because debugging machine learning models is a complex task. In my experience I often notice that when things do not
 work as expected (e.g., low testing accuracy, longer training times, bad generalization, large amount of false negatives,
 NaN predictions, etc.) it really helps to know the internal parts of the algorithm. Just when you disassemble the black box,
 you can create more complex capabilities or leveraging invariants in its behavior. For example,
 - If it takes so much time to train, it maybe be a good idea to increase the size of a minibatch to reduce the variance
 in the examples and thus helping the algorithm to converge
 - NaN predictions often indicate that the algorithm expected larger gradients, so in presence of small one the negative
 exponential of the Sigmoid activation 1.0/(1.0+np.exp(-WX)) produces memory overflow.

## Concrete Example: Learning the XOR Function

>Let's open the blackbox. We will build now a neural network from scratch that learns the XOR function.
The choice of this non-linear function is by no means random chance. Without backpropagation it would be hard to learn
to separate classes with a straight line.

To illustrate this important concept, note below how a straight line cannot
separate 0s and 1s, the outputs of the XOR function. Real life problems are also non-linearly separable.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/nonlinear_xor.png "Logo Title Text 1")


The topology of the network is simple:
- Input X is a two dimensional vector
- Weights W1 is a 2x3 matrix with randomly initialized values
- Hidden layer h1 consists of three neurons. Each neuron receives as input a weighted sum of observations, this is the inner product
highlighted in green in the below figure: z1 = [x1, x2][w1, w2]
- Weights W2 is a 3x2 matrix with randomly initialized values and
- Output layer h2 consists of two neurons since the XOR function returns either 0 (y1=[0,1]) or 1 (y2 = [1,0])


More visually:

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/overview2.png "Logo Title Text 1")

Let's now train the model. In our simple example the trainable parameters are weights, but be aware that current
research is exploring more types of parameters to be optimized. For example shortcuts between layers, regularized distributions, topologies,
residual, learning rates, etc.

Backpropagation is a method to update the weights towards the direction (gradient) that minimizes a predefined error metric known as Loss function
given a batch of labeled observations. This algorithm has been repeatedly rediscovered and is a special case of a more general technique called
[automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) in reverse accumulation mode.

### Network Initialization

>Let's initialize the network weights with random numbers.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/initialized_network.png "Logo Title Text 1"){:width="1300px"}

### Forward Step:

>This goal of this step is to forward propagate the input X to each layer of the network until computing a vector in
the output layer h2.

This is how it happens:
- Linearly map input data X using weights W1 as a kernel:


![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/z1.png){:width="500px"}

- Scale this weighted sum z1 with a Sigmoid function to get values of the first hidden layer h1. Note that the original
2D vector is now mapped to a 3D space.


![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/h1.png){:width="400px"}

- A similar process takes place for the second layer h2. Let's compute first the weighted sum z2 of the
first hidden layer, which is now input data.


![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/z2.png){:width="500px"}

- And then compute their Sigmoid activation function. This vector [0.37166596 0.45414264] represents the log probability
or predicted vector computed by the network given input X.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/h2.png){:width="300px"}

### Computing the Total Loss

>Also known as "actual minus predicted", the goal of the loss function is to quantify the distance between the predicted
 vector h2 and the actual label provided by humans y.

Note that the loss function Loss contains a regularization component that penalizes large weight values as in a Ridge
regression. In other words, large squared weight values will increase the loss function, an error metric we indeed want to minimize.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/loss.png){:width="500px"}

### Backward step:
>The goal of this step is to update the weights of the neural network in a direction that minimizes its Loss function.
As we will see, this is a recursive algorithm, which can reuse gradients previously computed and heavily relies on
differentiable functions. Since these updates reduce the loss function, a network ‘learns’ to approximate the label
of observations with known classes. A property called generalization.

This step goes in backward order than the forward step. It computes first the partial derivative of the loss function
with respect to the weights of the output layer (dLoss/dW2) and then the hidden layer (dLoss/dW1). Let's explain
in detail each one.

#### dLoss/dW2:

The chain rule says that we can decompose the computation of gradients of a neural network into differentiable pieces:

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/chain_w2.png){:width="500px"}

As a memory helper, these are the function definitions used above and their first derivatives:

| Function       |  First derivative |
|------------------------------------------------------------ |------------------------------------------------------------|
|Loss = (y-h2)^2     | dLoss/dW2 = -(y-h2) |
|h2 = Sigmoid(z2) | dh2/dz2 = h2(1-h2) |
|z2 = h1W2 | dz2/dW2 = h1 |


More visually, we aim to update the weights W2 (in blue) in the below figure. In order to that, we need to compute
three partial derivatives along the chain.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/update_w2.png){:width="500px"}

Replacing these partial derivatives with their corresponding values allow us to compute them as follows.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/chain_w2_detailed.png){:width="600px"}

And putting pieces together results in the 3x2 matrix dLoss/dW2, which will update the original W2 values in the direction
of minimizing the Loss function.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/chain_w2_numbers.png){:width="600px"}

#### dLoss/dW2:

Computing the chain rule for updating the weights of the first hidden layer W1 exhibits the possibility of reusing existing
computations.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/chain_w1.png){:width="500px"}

More visually, the path from the output layer to the weights W1 touches partial derivatives already computed in latter
layers.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/update_w1.png){:width="500px"}

For example, partial derivatives dLoss/dh2 and dh2/dz2 have been already computed as a dependency for learning weights
of the output layer dLoss/dW2 in the previous section.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/chain_w1_numbers.png){:width="700px"}

Placing all derivatives together, we can execute the chain rule again to update the weights of the hidden layer W1:

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/chain_w1_numbers_final.png){:width="700px"}

Finally, we assign the new values of the weights and have completed an iteration on the training of network.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/copy_values.png){:width="100px"}

### Model is Alive

A simple neural network shows us.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/all_3neurons_lr_0.003_reg_0.0.gif){:height="500px"}

Adding more neurons to the network increases its complexity to learn non-linear decision boundaries.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/all_50neurons_lr_0.003_reg_0.0001.gif){:height="500px"}



