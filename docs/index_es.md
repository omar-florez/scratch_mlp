---
layout: page
title: Un LEGO a la vez&#58; Explicando la Matemática de como las Redes Neuronales Aprenden
tagline:
description: Tutorial de retro-alimentación
---

>Una **red neuronal** es un composición inteligente de módulos lineales y no lineales. Cuando los escogemos sabiamente, tenemos una herramienta muy poderosa para optimizar cualquier función matemática. Por ejemplo una que  **separe clases con un limite de decisión no lineal**.

Un tópico que no es siempre explicado en detalle, a pesar de su naturaleza intuitiva y modular, es el **algoritmo de retro-alimentación** (backpropagation algorithm)
Responsable de actualizar parámetros entrenables en la red. Construyamos una red neuronal desde cero para ver el funcionamiento interno de una red neuronal usando  **piezas de LEGO como una analogía**, un bloque a la vez.

Código implementando estos conceptos pueden ser encontrados en el siguiente repositorio: [https://github.com/omar-florez/scratch_mlp](https://github.com/omar-florez/scratch_mlp)

## Las Redes Neuronales  como una Composición de Piezas

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/overview.png "Logo Title Text 1")

La figura de arriba muestra algo de la matemática usada para entrenar una red neuronal. Haremos sentido de esto durante el articulo.
El lector puede encontrar interesante que una red neuronal es una pila de módulos con diferentes propósitos:

- **Entrada X** alimenta la red neuronal con datos sin procesar, la cual se almacena en una matriz en la cual las observaciones con filas y las dimensiones son columnas
- **Pesos W1** proyectan entrada X a la primera capa escondida  h1. Pesos W1 trabajan entonces como un kernel lineal
- Una **función Sigmoid ** que previene los números de la capa escondida de salir del rango 0-1. El resultado es un **array activaciones neuronales** h1 = Sigmoid(WX)

Hasta este punto estas operaciones solo calculan  un **sistema general lineal **, el cual no tiene la capacidad de modelar interacciones no lineales.
Esto cambia cuando ponemos otro elemento en el pila, añadiendo profundidad a la estructura modular. Mientras más profunda sea la red, más interacciones no-lineales podremos aprender y problemas mas complejos podremos resolver, lo cual puede explicar en parte la popularidad de redes neuronales.

## Porque debería leer esto?

>Si uno entiende las partes internas de una red neuronal, es mas fácil saber **que cambiar primero** cuando el algoritmo no funcione como es esperado y permite definir una estrategia para **probar invariantes ** and **comportamientos esperados** que uno saben son parte del algoritmo. Esto también es útil cuando el lector quiere **crear nuevos algoritmos que actualmente no están implementados en la librería de Machine Learning de preferencia **.

**Porque hacer debugging de modelos de aprendizaje de maquina es una tarea compleja**. Por experiencia,  modelos matemáticos no funcionan como son esperados al primer intento. A veces estos pueden darte una exactitud baja para datos nuevos, tomar mucho tiempo de entrenamiento o mucha memoria RAM, devolver una gran cantidad de falsos negativos o valores NaN (Not a Number), etc. Déjame mostrarte algunos casos donde saber como el algoritmo funciona puede ser útil:

 - Si **toma mucho tiempo para entrenar**, es quizás una buena idea incrementar el tamaño del minbatch o array de observaciones que alimentan a la red neuronal para reducir la varianza en las observaciones y así ayudar al algoritmo a converger
 - Si se observa **valores NaN **, el algoritmo ha recibido gradientes con valores muy altos produciendo desborde de memoria RAM. Piensa esto como una secuencia de multiplicaciones de matrices que explotan después de varias iteraciones. Reducir la velocidad de aprendizaje tendrá el efecto de escalar estos valores. Reduciendo el numero de capas reducirá el numero de multiplicaciones. Y poniendo una cota superior a los gradientes (clipping gradients) controlara este problema explícitamente

## Concrete Example: Learning the XOR Function

>Let's open the blackbox. We will build now a neural network from scratch that learns the **XOR function**.
The choice of this **non-linear function** is by no means random chance. Without backpropagation it would be hard to learn
to separate classes with a **straight line**.

To illustrate this important concept, note below how a straight line cannot
separate 0s and 1s, the outputs of the XOR function. **Real life problems are also non-linearly separable**.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/nonlinear_xor.png "Logo Title Text 1")

The topology of the network is simple:
- **Input X** is a two dimensional vector
- **Weights W1** is a 2x3 matrix with randomly initialized values
- **Hidden layer h1** consists of three neurons. Each neuron receives as input a weighted sum of observations, this is the inner product
highlighted in green in the below figure: **z1 = [x1, x2][w1, w2]**
- **Weights W2** is a 3x2 matrix with randomly initialized values and
- **Output layer h2** consists of two neurons since the XOR function returns either 0 (y1=[0,1]) or 1 (y2 = [1,0])

More visually:

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/overview2.png "Logo Title Text 1")

Let's now train the model. In our simple example the trainable parameters are weights, but be aware that current
research is exploring more types of parameters to be optimized. For example shortcuts between layers, regularized distributions, topologies,
residual, learning rates, etc.

**Backpropagation** is a method to update the weights towards the direction (**gradient**) that minimizes a predefined error metric known as **Loss function**
given a batch of labeled observations. This algorithm has been repeatedly rediscovered and is a special case of a more general technique called
[automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) in reverse accumulation mode.

### Network Initialization

>Let's **initialize the network weights** with random numbers.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/initialized_network.png "Logo Title Text 1"){:width="1300px"}

### Forward Step:

>This goal of this step is to **forward propagate** the input X to each layer of the network until computing a vector in
the output layer h2.

This is how it happens:
- Linearly map input data X using weights W1 as a kernel:


![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/z1.png){:width="500px"}

- Scale this weighted sum z1 with a Sigmoid function to get values of the first hidden layer h1. **Note that the original
2D vector is now mapped to a 3D space**.


![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/h1.png){:width="400px"}

- A similar process takes place for the second layer h2. Let's compute first the **weighted sum** z2 of the
first hidden layer, which is now input data.


![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/z2.png){:width="500px"}

- And then compute their Sigmoid activation function. This vector [0.37166596 0.45414264] represents the **log probability**
or **predicted vector** computed by the network given input X.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/h2.png){:width="300px"}

### Computing the Total Loss

>Also known as "actual minus predicted", the goal of the loss function is to **quantify the distance between the predicted
 vector h2 and the actual label provided by humans y**.

Note that the Loss function contains a **regularization component** that penalizes large weight values as in a Ridge
regression. In other words, large squared weights values will increase the Loss function, **an error metric we indeed want to minimize**.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/loss.png){:width="500px"}

### Backward step:
>The goal of this step is to **update the weights of the neural network** in a direction that minimizes its Loss function.
As we will see, this is a **recursive algorithm**, which can reuse gradients previously computed and heavily relies on
**differentiable functions**. Since these updates reduce the loss function, a network ‘learns’ to approximate the label
of observations with known classes. A property called **generalization**.

This step goes in **backward order** than the forward step. It computes first the partial derivative of the loss function
with respect to the weights of the output layer (dLoss/dW2) and then the hidden layer (dLoss/dW1). Let's explain
in detail each one.

#### dLoss/dW2:

The chain rule says that we can decompose the computation of gradients of a neural network into **differentiable pieces**:

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/chain_w2.png){:width="500px"}

As a memory helper, these are the **function definitions** used above and their **first derivatives**:

| Function       |  First derivative |
|------------------------------------------------------------ |------------------------------------------------------------|
|Loss = (y-h2)^2     | dLoss/dW2 = -(y-h2) |
|h2 = Sigmoid(z2) | dh2/dz2 = h2(1-h2) |
|z2 = h1W2 | dz2/dW2 = h1 |
|z2 = h1W2 | dz2/dh1 = W2 |


More visually, we aim to update the weights W2 (in blue) in the below figure. In order to that, we need to compute
three **partial derivatives along the chain**.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/update_w2.png){:width="500px"}

Plugging in values into these partial derivatives allow us to compute gradients with respect to weights W2 as follows.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/chain_w2_detailed.png){:width="600px"}

The result is a 3x2 matrix dLoss/dW2, which will update the original W2 values in a direction that minimizes the Loss function.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/chain_w2_numbers.png){:width="700px"}

#### dLoss/dW1:

Computing the **chain rule** for updating the weights of the first hidden layer W1 exhibits the possibility of **reusing existing
computations**.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/chain_w1.png){:width="500px"}

More visually, the **path from the output layer to the weights W1** touches partial derivatives already computed in **latter
layers**.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/update_w1.png){:width="500px"}

For example, partial derivatives dLoss/dh2 and dh2/dz2 have been already computed as a dependency for learning weights
of the output layer dLoss/dW2 in the previous section.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/chain_w1_numbers.png){:width="700px"}

Placing all derivatives together, we can execute the **chain rule** again to update the weights of the hidden layer W1:

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/chain_w1_numbers_final.png){:width="700px"}

Finally, we assign the new values of the weights and have completed an iteration on the training of network.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/copy_values.png){:width="150px"}

### Implementation

Let's translate the above mathematical equations to code only using [Numpy](http://www.numpy.org/) as our **linear algebra engine**.
Neural networks are trained in a loop in which each iteration present already **calibrated input data** to the network.
In this small example, let's just consider the entire dataset in each iteration. The computations of **Forward step**,
**Loss**, and **Backwards step** lead to good generalization since we update the **trainable parameters** (matrices w1 and
w2 in the code) with their corresponding **gradients** (matrices dL_dw1 and dL_dw2) in every cycle.
Code is stored in this repository: [https://github.com/omar-florez/scratch_mlp](https://github.com/omar-florez/scratch_mlp)

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/code.png)

### Let's Run This!

See below **some neural networks** trained to approximate the **XOR function** over many iterations.

**Left plot:** Accuracy. **Central plot:** Learned decision boundary. **Right plot:** Loss function.

First let's see how a neural network with **3 neurons** in the hidden layer has small capacity. This model learns to separate 2 classes
with a **simple decision boundary** that starts being a straight line but then shows a non-linear behavior.
The loss function in the right plot nicely gets low as training continues.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/all_3neurons_lr_0.003_reg_0.0.gif)

Having  **50 neurons** in the hidden layer notably increases model's power to learn more **complex decision boundaries**.
This could not only produce more accurate results, but also **exploiting gradients**, a notable problem when training neural networks.
This happens when very large gradients multiply weights during backpropagation and thus generate large updated weights.
This is reason why the **Loss value suddenly increases** during the last steps of the training (step > 90).
The **regularization component** of the Loss function computes the **squared values** of weights that are already very large (sum(W^2)/2N).

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/all_50neurons_lr_0.003_reg_0.0001.gif)

This problem can be avoided by **reducing the learning rate** as you can see below. Or by implementing a policy that reduces
the learning rate over time. Or by enforcing a stronger regularization, maybe L1 instead of L2.
**Exploiding** and **vanishing gradients** are interesting phenomenons and we will devote an entire analysis later.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/all_50neurons_lr_0.003_reg_0.000001.gif)


