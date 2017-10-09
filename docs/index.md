---
layout: page
title: One LEGO at a time&#58; Explaining the Math of how neural networks learn
tagline: Scratch MLP
description: Tutorial on back-propagation
---

A neural network is a clever arrangement of linear and non-linear modules. When we choose and connect them wisely,
we have a powerful tool to approximate any mathematical function that learns to separate classes with a non-linear
decision boundary. A topic that is not always explained in depth, despite of its intuitive and modular nature, is the
back-propagation technique responsible for updating trainable parameters. Let’s see the internal functioning of a neural
network using LEGO modules, one brick at a time:

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/overview.png "Logo Title Text 1")

- Input X is raw data stored in a matrix in which observations are rows and dimensions are columns
- Things become more interesting when we map such input to a different space using network's weights W as a linear kernel
- To avoid that numbers go out of range, we scale the values to 0-1 using a Sigmoid function

So far this operation is only a general linear system and doesn’t have the capacity to model non-linear interactions.
This changes when we stack one more layer (hence the rise of deep learning models). And we are finally ready for the
Softmax block to transform the hidden layer values into a multinomial probability of k states, one for each class.

## Concrete example: Learning the XOR function

Let's open the blackbox, our neural network is intended to learn the XOR function.
The election of this non-linear function is by no means random chance. Without backpropagation it would be hard to learn
to separate classes with a straight line (see below image). This is a major type of problems that we find in real life.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/nonlinear_xor.png "Logo Title Text 1")

The topology of the network is simple:
- Input X is a two dimensional vector
- Hidden layer h1 consists of three neurons and
- Output layer h2 consists of two neurons since the XOR function returns either 0 (y1=[0,1]) or 1 (y1 = [1,0])

More visually:
![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/overview2.png "Logo Title Text 1")

Let's now find values for the network's trainable parameters. In our simple example these are weights, but be aware that current
research is exploring more type of parameters to be optimized. For example stable distributions, residual, learning rates, etc.

The method to update the weights towards a direction (gradient) that minimizes a predefined error metric (a.k.a.
loss function) given a batch of observations is named backpropagation. The backpropagation algorithm has been
repeatedly rediscovered and is a special case of a more general technique called ![automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) in
reverse accumulation mode.



## Concrete example: Learning XOR function

When we initialize the network weights with numbers, it looks as follows:




A neural network learns to remember reoccurring patterns by




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
