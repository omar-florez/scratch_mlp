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
- Una **función Sigmoid** que previene los números de la capa escondida de salir del rango 0-1. El resultado es un **array activaciones neuronales** h1 = Sigmoid(WX)

Hasta este punto estas operaciones solo calculan  un **sistema general lineal**, el cual no tiene la capacidad de modelar interacciones no lineales.
Esto cambia cuando ponemos otro elemento en el pila, añadiendo profundidad a la estructura modular. Mientras más profunda sea la red, más interacciones no-lineales podremos aprender y problemas mas complejos podremos resolver, lo cual puede explicar en parte la popularidad de redes neuronales.

## Porque debería leer esto?

>Si uno entiende las partes internas de una red neuronal, es mas fácil saber **que cambiar primero** cuando el algoritmo no funcione como es esperado y permite definir una estrategia para **probar invariantes** and **comportamientos esperados** que uno saben son parte del algoritmo. Esto también es útil cuando el lector quiere **crear nuevos algoritmos que actualmente no están implementados en la librería de Machine Learning de preferencia**.

**Porque hacer debugging de modelos de aprendizaje de maquina es una tarea compleja**. Por experiencia,  modelos matemáticos no funcionan como son esperados al primer intento. A veces estos pueden darte una exactitud baja para datos nuevos, tomar mucho tiempo de entrenamiento o mucha memoria RAM, devolver una gran cantidad de falsos negativos o valores NaN (Not a Number), etc. Déjame mostrarte algunos casos donde saber como el algoritmo funciona puede ser útil:

 - Si **toma mucho tiempo para entrenar**, es quizás una buena idea incrementar el tamaño del mini-batch o array de observaciones que alimentan a la red neuronal para reducir la varianza en las observaciones y así ayudar al algoritmo a converger
 - Si se observa **valores NaN**, el algoritmo ha recibido gradientes con valores muy altos produciendo desborde de memoria RAM. Piensa esto como una secuencia de multiplicaciones de matrices que explotan después de varias iteraciones. Reducir la velocidad de aprendizaje tendrá el efecto de escalar estos valores. Reduciendo el numero de capas reducirá el numero de multiplicaciones. Y poniendo una cota superior a los gradientes (clipping gradients) controlara este problema explícitamente

## Un Ejemplo Concreto: Aprendiendo la Función XOR

>Abramos la caja negra. Construiremos a continuación una red neuronal desde cero que aprende la **función XOR**.
La elección de esta **función no linear** no es por casualidad. Sin backpropagation seria difícil aprender a separar clases con una **línea recta**.

Para ilustrar este importante concepto, note a continuación como una línea recta no puede separar 0s and 1s, las salidas de la función XOR. **Los problemas reales también son linealmente no separables**.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/nonlinear_xor.png "Logo Title Text 1")

La topología de la red es simple:
- **Entrada X** es un vector de dos dimensiones
- **Pesos W1** son una matriz de 2x3 dimensiones con valores inicializados de forma aleatoria
- **Capa escondida h1** consiste de 3 neuronas. Cada neurona recibe como entrada la suma de sus observaciones escaladas por sus pesos, este es el producto punto resaltado en verde en la figura de abajo: **z1 = [x1, x2][w1, w2]**
- **Pesos W2** son una matroz de 3x2 con valores inicializados de forma aleatoria y
- **Capa de salida h2** consiste de 2 neuronas ya que la función  XOR retorna 0 (y1=[0,1]) o 1 (y2 = [1,0])

Mas visualmente:

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/overview2.png "Logo Title Text 1")

Entrenemos ahora el modelo. En nuestro ejemplo los valores entrenables son los pesos, pero tenga en cuenta que la investigación actual esta explorando nuevos tipos de parámetros a ser optimizados. Por ejemplo, atajos entre capas, distribuciones estables en las capas, topologías, velocidades de aprendizaje, etc.

**Backpropagation** es un método para actualizar los pesos en la dirección (**gradiente**) que minimiza una métrica de error predefinida conocida como  **función Loss**
dado un conjunto de observaciones etiquetadas. Este algoritmo ha sido repetidamente redescubierto y  es un caso especial de una técnica mas general llamada [diferenciación automática](https://en.wikipedia.org/wiki/Automatic_differentiation) en modo acumulativo reverso.

### Inicialización de la Red

>Inicialicemos **los pesos de la red ** con valores aleatorios.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/initialized_network.png "Logo Title Text 1"){:width="1300px"}

### Propagación hacia Adelante:

>El objetivo de este paso es **propagar hacia delante** la entrada X a cada capa de la red hasta calcular un vector en la capa de salida h2.

Es así como sucede:
- Se proyecta linealmente la entrada X usando pesos W1 a manera de kernel:


![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/z1.png){:width="500px"}

- Se escala esta suma z1 con una función Sigmoid para obtener valores de la primera capa escondida. **Note que el vector original de 2D ha sido proyectado ahora a 3D**.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/h1.png){:width="400px"}

- Un proceso similar toma lugar para la segunda capa h2. Calculemos primero la **suma** z2 de la primera capa escondida, la cual es ahora un vector de entrada.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/z2.png){:width="500px"}

- Y luego calculemos su activación Sigmoid. Este vector [0.37166596 0.45414264] representa el **logaritmo de la probabilidad**
o **vector predecido** calculado por la red dado los datos de entrada X.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/h2.png){:width="300px"}

### Calculando el Error Total

>También conocido como "valor real menos predecido", el objetivo de la función Loss es **cuantificar la distancia entre el vector predecido h2 y la etiqueta real proveída por un ser humano, y**.

Note que la función Loss contiene un **componente de regularización** que penaliza valores de los pesos muy altos a manera de una regresión L2. En otras palabras, grandes valores cuadrados de los pesos incrementaran la función Loss, **una métrica de error que en realidad queremos reducir**.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/loss.png){:width="500px"}

### Propagación hacia Atrás:
>El objetivo de este paso es **actualizar los pesos de la red neuronal ** en una dirección que minimiza la función Loss.
Como veremos mas adelante, este es un **algoritmo recursivo**, el cual reutiliza gradientes previamente calculadas y se basada plenamente en
**funciones diferenciables**. Ya que estas actualizaciones reducen la función Loss, una red ‘aprende’ a aproximar las etiquetas de nuevas observaciones. Una propiedad llamada **generalización**.

Este paso va en  **orden reverso** que la propagación hacia adelante. Este calcula la primera derivada de la función Loss con respecto a los pesos de la red neuronal de la capa de salida (dLoss/dW2) y  luego los de la capa escondida (dLoss/dW1). Expliquemos en detalle cada uno.

#### dLoss/dW2:

La regla de la cadena dice que podemos descomponer el calculo de gradientes de una red neuronal en **funciones diferenciables**:

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/chain_w2.png){:width="500px"}

Aquí están las **definiciones de funciones** usadas arriba y sus **primeras derivadas**:

| Función        |  Primera derivada |
|------------------------------------------------------------ |------------------------------------------------------------|
|Loss = (y-h2)^2     | dLoss/dW2 = -(y-h2) |
|h2 = Sigmoid(z2) | dh2/dz2 = h2(1-h2) |
|z2 = h1W2 | dz2/dW2 = h1 |
|z2 = h1W2 | dz2/dh1 = W2 |


Mas visualmente, queremos actualizar los pesos W2 (en azul) en la figura de abajo. Para eso necesitamos calcular tres **derivadas parciales a lo largo de la cadena**.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/update_w2.png){:width="500px"}

Insertando esos valores esas derivadas parciales nos permite calcular gradientes con respecto a los pesos W2 como sigue.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/chain_w2_detailed.png){:width="600px"}

El resultado es una matriz de 3x2 llamada dLoss/dW2, la cual actualizara los valores originales de W2 en una dirección que minimiza la función Loss.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/chain_w2_numbers.png){:width="700px"}

#### dLoss/dW1:

Calculando la **regla de la cadena** para actualizar los pesos de la primera capa escondida W1 exhibe la posibilidad de  **reutilizar cálculos existentes**.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/chain_w1.png){:width="500px"}

Mas visualmente, el **camino desde la capa de salida hasta los pesos W1** toca derivadas parciales ya calculadas en capas mas superiores.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/update_w1.png){:width="500px"}

Por ejemplo, la derivada parcial dLoss/dh2 y dh2/dz2 ha sido ya calculada como una dependencia para aprender los pesos de la capa de salida dLoss/dW2 en la sección anterior.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/chain_w1_numbers.png){:width="700px"}

Ubicando todas las derivadas juntas, podemos ejecutar la **regla de la cadena** de nuevo para actualizar los pesos de la capa escondida W1:

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/chain_w1_numbers_final.png){:width="700px"}

Finalmente, asignamos los nuevos valores de los pesos y hemos completado una iteración del entrenamiento de la red neuronal!

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/copy_values.png){:width="150px"}

### Implementación

Traduzcamos las ecuaciones matemáticas de arriba en código solamente utilizando [Numpy](http://www.numpy.org/) como nuestro **motor de algebra linar**.
Redes neuronales son entrenadas en un loop en el cual cada iteración presenta **datos de entrada ya calibrados** a la red.
En este pequeño ejemplo, consideremos todo el dataset en cada iteración. Los cálculos del paso de **Propagación hacia adelante**,
**Loss**, y **Propagación hacia atrás** conducen a obtener una buena generalización ya que actualizaremos los **parámetros entrenables** (matrices W1 and W2 en el código) con sus correspondientes **gradientes** (matrices dL_dw1 and dL_dw2) en cada ciclo.
El código es almacenado en este repositorio: [https://github.com/omar-florez/scratch_mlp](https://github.com/omar-florez/scratch_mlp)

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/code.png)

### Ejecutemos Esto!

Mire abajo **algunas redes neuronales** entrenadas para aproximar la **función XOR** en múltiple iteraciones.

**Izquierda:** Exactitud. **Centro:** Borde de decisión aprendido. **Derecha:** Función Loss.

Primero veamos como una red neuronal con **3 neuronas** en la capa escondida tiene una pequeña capacidad. Este modelo aprende a separar dos clases con un **simple borde de decisión** que empieza una línea recta, pero luego muestra un comportamiento no lineal.
La función Loss en la derecha suavemente se reduce mientras el proceso de aprendizaje ocurre.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/all_3neurons_lr_0.003_reg_0.0.gif)

Teniendo  **50 neuronas** en la capa escondida notablemente incremental el poder del modelo para aprender  **bordes de decisión mas complejos**.
Esto podría no solo producir resultados mas exactos, pero también **explotar las gradientes**, un problema notable cuando se entrena redes neuronales.
Esto sucede cuando gradientes muy grandes multiplican pesos durante la propagación hacia atrás y así generan pesos actualizados muy grandes.
Esta es la razón por la que **valores de la función Loss repentinamente se incrementan** durante los últimos pasos del entrenamiento (step > 90).
El **componente de regularicion** de la función Loss calcula los **valores cuadrados** de los pesos que ya tienen valores muy altos (sum(W^2)/2N).

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/all_50neurons_lr_0.003_reg_0.0001.gif)

Este problema puede ser evitado **reduciendo la velocidad de aprendizaje** como puede ver abajo. O implementado una política que reduzca la velocidad de aprendizaje con el tiempo. O imponiendo una regularización mas fuerte, quizás L1 en vez de L2.
Gradientes que **explotan** y se **desvanecen** son interesantes fenómenos y haremos un análisis detallada de eso mas adelante.

![alt text](https://raw.githubusercontent.com/omar-florez/scratch_mlp/master/docs/assets/all_50neurons_lr_0.003_reg_0.000001.gif)


