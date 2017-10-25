---
layout: page
title: One LEGO at a time&#58; Explaining the Math of how neural networks learn
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

At this point these operations only compute a general linear system and doesn’t have the capacity to model non-linear interactions.
This changes when we stack one more layer adding depth to this modular structure. The deeper the network, the more subtle non-linear interactions
can be learned; hence the rise of deep neural models. Finally, a Softmax module converts the activations of a layer values into a
multinomial probability of k states, one for each class.

## Concrete Example: Learning the XOR Function

>Let's open the blackbox. We will build now a neural network from scratch that learns the XOR function.
The choice of this non-linear function is by no means random chance. Without backpropagation it would be hard to learn
to separate classes with a straight line.

To illustrate this important concept, note below how a straight line cannot
separate 0s and 1s, the outputs of the XOR function. Real life problems are also non-linearly separable.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/nonlinear_xor.png "Logo Title Text 1")


The topology of the network is simple:
- Input X is a two dimensional vector
- Weights W1 is a 2x3 matrix with randonmly initialized values
- Hidden layer h1 consists of three neurons. Each neuron is then a weighted sum of observations, this inner product
is highlighted in green: [x1, x2][w1, w2]
- Weights W2 is a 3x2 matrix with randonmly initialized values and
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

>Let's initialize the network weights with random numbers:

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/initialized_network.png "Logo Title Text 1"){:width="1300px"}

### Forward Step:

>This goal of this step is to forward propagate the input X to each layer of the network until computing a vector in
the output layer h2.

This is how it happens:
- Linearly map input data X using weights W1 as a kernel:


![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/z1.png){:width="500px"}

- Scale this weighted sum Z1 with a Sigmoid function to get values of the first hidden layer h1. Note that the original
2D vector has been mapped to a 3D space after this.


![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/h1.png){:width="400px"}

- A similar process takes place for the second layer h2. Let's compute first the weighted sum z2 of the values in the
first hidden layer.


![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/z2.png){:width="500px"}

- And then compute their Sigmoid activation function. This vector [0.37166596 0.45414264] represents the log probability
(logit) or predicted vector provided by the network given input X.

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
differentiable functions. Since these updates reduce the loss function, a network is ‘learning’ to approximate the label
of observations with known classes. A property called generalization.

This step goes in reverse order than the forward step. It computes the partial derivative of the loss function
with respect to the weights connecting to the output layer (dLoss/dW2) and then the hidden layer (dLoss/dW1).

#### dLoss/dW2:

The chain rule says that we can decompose the modular structure of a neural network into diferentiable pieces:

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/chain_w2.png){:width="500px"}

More visually, we aim to update the weights in blue in the below figure (W2). In order to that, we need to compute the
three partial derivatives along the chain.
![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/update_w2.png){:width="500px"}

Replacing those functions with their partial derivatives give us:
![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/chain_w2_detailed.png){:width="500px"}

And now plugin in the matrices:
![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/chain_w2_numbers.png){:width="500px"}

#### dLoss/dW2:

The chain rule for updaring the weights of the first weights exhibits the posibility of reusing existing computations:


![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/chain_w1.png){:width="500px"}

We can reuse existing computations:
![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/chain_w1_numbers.png){:width="500px"}

Placing all values together:
![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/chain_w1_numbers_final.png){:width="500px"}




[Github Pages](https://pages.github.com) provide a simple way to make a website using
[Markdown](https://daringfireball.net/projects/markdown/) and
[git](https://git-scm.com).

FFor me, the painful aspects of making a website are

- Working with html and css
- Finding a hosting site
- Transferring stuff to the hosting site

With [GitHub Pages](https://pages.github.com), you just write things in
[Markdown](https://daringfireball.net/projects/markdown/),
[GitHub](https://github.com) hosts the site for you, and you just push
material to your GitHub repository with `git add`, `git commit`, and
`git push`.

If you love [git](https://git-scm.com/) and
[GitHub](https://github.com), you'll love
[GitHub Pages](https://pages.github.com), too.

The sites use [Jekyll](https://jekyllrb.com/), a
[ruby](https://www.ruby-lang.org/en/) [gem](https://rubygems.org/), to
convert Markdown files to html, and this part is done
automatically when you push the materials to the `gh-pages` branch
of a GitHub repository.

The [GitHub](https://pages.github.com) and
[Jekyll](https://jekyllrb.com) documentation is great, but I thought it
would be useful to have a minimal tutorial, for those who just want to
get going immediately with a simple site. To some readers, what GitHub
has might be simpler and more direct.  But if you just want to create
a site like the one you're looking at now, read on.

Start by reading the [Overview page](pages/overview.html), which
explains the basic structure of these sites. Then read
[how to make an independent website](pages/independent_site.html). Then
read any of the other things, such as
[how to test your site locally](pages/local_test.html).

- [Overview](pages/overview.html)
- [Making an independent website](pages/independent_site.html)
- [Making a personal site](pages/user_site.html)
- [Making a site for a project](pages/project_site.html)
- [Making a jekyll-free site](pages/nojekyll.html)
- [Testing your site locally](pages/local_test.html)
- [Resources](pages/resources.html)

If anything here is confusing (or _wrong_!), or if I've missed
important details, please
[submit an issue](https://github.com/kbroman/simple_site/issues), or (even
better) fork [the GitHub repository for this website](https://github.com/kbroman/simple_site),
make modifications, and submit a pull request.

---

The source for this minimal tutorial is [on github](https://github.com/kbroman/simple_site).

Also see my [tutorials](http://kbroman.org/pages/tutorials) on
[git/github](http://kbroman.org/github_tutorial),
[GNU make](http://kbroman.org/minimal_make),
[knitr](http://kbroman.org/knitr_knutshell),
[R packages](http://kbroman.org/pkg_primer),
[data organization](http://kbroman.org/dataorg),
and [reproducible research](http://kbroman.org/steps2rr).
