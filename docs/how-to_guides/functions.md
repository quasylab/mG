# How to define functions
The definition of functions is the most important part of the $\mu\mathcal{G}$ workflow. The compiler for $\mu\mathcal{G}$ expressions is instantiated by 
providing the dictionaries for the $\psi$, $\varphi$ and $\sigma$ functions. This guide will show you how to instantiate these functions and how to create 
these dictionaries.

There are three main interfaces for creating function. For the most part they are equivalent, but differences arise when defining trainable functions, as 
will be explained in the following sections.

## Types of functions
The three main types of functions are $\psi$, $\varphi$, and $\sigma$. The $\psi$ functions are used to transform the node labels without using edge or 
neighbour information. The $\varphi$ functions are used to generate messages from a nodes neighbour, and the $\sigma$ functions are used to aggregate these 
messages and update the labels of the node. These two types of functions are used in tandem in the pre-image and post-image expressions.

The $\psi$ functions can be defined using the following classes:

* `PsiLocal`: used for functions that transform individual node labels using only local (the node label itself) information without using global (the labels 
  of all other nodes) 
  information.
    * `Constant`: used for functions that transform a node label to some constant value.
    * `Pi`: used for projection functions that transform a node label to a projection of itself.
* `PsiNonLocal`: used for functions that transform individual node labels using both local (the node label itself) and global (the labels of all other nodes) 
  information. 
* `PsiGlobal`: used for functions that transform all node labels to some value obtained using only global (the labels of all other nodes) information.

The $\varphi$ functions be defined with the `Phi` class and the $\sigma$ functions can be defined with the `Sigma` class.

!!! note
    When defining functions, make sure that their output maintains the shape of the node features matrix $X$, which is always a rank-2 tensor. Even if the 
    nodes have one single feature, the node features matrix should never become a rank-1 tensor.

!!! note
    The two subclasses of `PsiLocal`, `Constant` and `Pi`, are meant to be used with the constructor interface only.

## Constructor interface

The constructor interface consists in instantiating the functions via their constructor. Typically, this consists in passing a lambda function to the 
constructor of the class. A name can also be passed to the constructor to be used for model summaries.

```python
import tensorflow as tf
from libmg import PsiLocal, Phi, Sigma

successor = PsiLocal(lambda x: x + 1)
projection_1 = Phi(lambda i, e, j: i)
aggregate_sum = Sigma(lambda m, i, n, x: tf.math.segment_sum(m, i))
```

We can create trainable functions using `tf.keras.layers.Dense`:

```python
dense = PsiLocal(tf.keras.layers.Dense(5, activation='relu'))
```

Using the constructor interface, whenever we reference to this `dense` function, the same `PsiLocal` instance is being used, therefore there is only one set 
of weights being shared across all usages of this function.

## Subclassing interface

The subclassing interface consists in subclassing the function classes and overriding their `func` method (for `PsiLocal`, `Phi`, and `Sigma` subclasses) or 
their `single_graph_op` and/or `multiple_graph_op` method (for `PsiNonLocal` and `PsiGlobal` subclasses). The model summary in this case uses the (sub)class 
name. The classes are not to be instantiated as this point, as the compiler will do it when needed.

```python
import tensorflow as tf
from libmg import PsiLocal, Phi, Sigma

class Successor(PsiLocal):
    def func(self, x: tf.Tensor) -> tf.Tensor:
        return x + 1

class Projection1(Phi):
    def func(self, src: tf.Tensor, e: tf.Tensor, tgt: tf.Tensor) -> tf.Tensor:
        return src
    
class AggregateSum(Sigma):
    def func(self, m: tf.Tensor, i: tf.Tensor, n: int, x: tf.Tensor) -> tf.Tensor:
        return tf.math.segment_sum(m, i)
```

We can create trainable functions using `tf.keras.layers.Dense`:

```python
class MyDense(PsiLocal):
    def __init__(self):
        self.dense = tf.keras.layers.Dense(5, activation='relu')
    
    def func(self, x: tf.Tensor) -> tf.Tensor:
        return self.dense(x)
```

Using the subclassing interface, whenever we refer to the `MyDense` function a new instance of the class is created, therefore a new function with its own 
set of weights is being used.

## Make interface

The make interface consists in calling the static factory method `make` on the class of the function we want to create. As for the constructor interface, we 
will typically pass a lambda function and a name to be used in summaries. The `make` function does not return an instance of the class, but rather a 
function that returns an instance of the class. The compiler will call this function as needed to get the instance.

```python
import tensorflow as tf
from libmg import PsiLocal, Phi, Sigma

successor = PsiLocal.make('successor', lambda x: x + 1)
projection_1 = Phi.make('projection_1', lambda i, e, j: i)
aggregate_sum = Sigma.make('aggregate_sum', lambda m, i, n, x: tf.math.segment_sum(m, i))
```

We can create trainable functions using `tf.keras.layers.Dense`:
```python
dense = PsiLocal.make('dense', tf.keras.layers.Dense(5, activation='relu'))
```

This time, using the `make` interface, each instance of `dense` will have its own set of weights, as in the subclassing interface. This is because `make` 
regenerates any `tf.keras.layer.Layer` that it receives in input. 

## Constants and Projections
The subclasses of `PsiLocal` can be used to define constant and projection functions. We can define a constant function by instantiating `Constant` with the
rank-1 `Tensor` we want the function to map to.

```python
import tensorflow as tf
from libmg import Constant
zero = Constant(tf.constant([0]))
three_ones = Constant(tf.constant([1, 1, 1]))
```

We can define a projection function by instantiating `Pi` with the initial index and the final index of the projection:

```python
from libmg import Pi
# maps every node label to its first element
first = Pi(0, 1)
# maps every node label to the sub-tensor consisting of its third and fourth element
two_to_four = Pi(2, 4)
```

## Dictionaries of functions
We create dictionaries of functions using the standard Python `dict`. The keys in this dictionary will be the terms with which we can refer to our function 
in a $\mu\mathcal{G}$ expression. The values are the functions we have defined with either interface: class instances if we used the constructor interface, 
(sub)classes if we used the subclassing interface, or functions returning class instances if we used the `make` interface.

```python
psi_functions = {'succ': successor, ...}
phi_functions = {'p1': projection_1, ...}
sigma_functions = {'+': aggregate_sum, ...}
```

!!! note
    The dictionary key is the string that should be used in a $\mu\mathcal{G}$ expression to refer to the corresponding function. The `name` argument used 
    in the constructor interface or the make interface is just part of the layer name that is shown using `model.summary()`.


## Parametrized functions
Sometimes we might want to create many functions which all share the same basic structure except for some value that changes. For example, we might want to 
have not only a successor function, but also an "add two" function, and an "add three" function, and so on. For this purpose, $\mu\mathcal{G}$ has a 
special syntax that allows to send a parameter from the $\mu\mathcal{G}$ expression to the function.

Let $a$ be a function name (as specified in the dictionary). When we write the $\mu\mathcal{G}$ expression $a[1]$ the compiler will send the string `"1"` as 
input to what is saved in the dictionary. 

You can create a parametrized function using any of the constructor, subclassing or make interfaces:

* Using the constructor interface, simply wrap your instance in a one-argument lambda: `#!python lambda y: PsiLocal(lambda x: x + int(y))`
* Using the subclassing interface, add one argument to the `__init__` of the subclass 
  ```python
  class Add(PsiLocal):
    def __init__(self, y):
        self.y = int(y)
    def func(self, x: tf.Tensor) -> tf.Tensor:
        return x + self.y
  ```
* Using the `make` interface, call instead the `make_parametrized` method by passing in either a two-argument lambda or a curried version of it: `#!python 
  PsiLocal.make_parametrized('add', lambda y, x: x + int(y))` or `#!python PsiLocal.make_parametrized('add', lambda y: lambda x: x + int(y))`

For example, if we have bound any of these functions to the word `add` in the compiler, and we write the expression $add[2]$ the compiler will generate and 
use a $\psi$ function that adds 2 to the input node labels.

## Operators

In $\mu\mathcal{G}$ functions are typically written in reverse Polish notation using sequential and parallel composition. For example, to compute the sum 
$+$ of the outputs of two $\psi$ functions $a$ and $b$ one typically has to write 

$$
(a || b) ; +
$$

It is possible to instead write the same expression in Polish notation without all the boilerplate code, that is

$$
+(a, b)
$$

For this purpose it is necessary to define functions with yet another interface, as *operators*. Three types of operators can be defined:

* Unary operators can be defined using `make_uoperator`:
  ```python
  from libmg import make_uoperator
  not = make_uoperator(tf.math.logical_not)
  ```
  Usage in $\mu\mathcal{G}$, assuming the compiler was passed the $\psi$ function dictionary `#!python {'~': not, 'a': ...}`:
  
    $$
    \sim(a)
    $$

* Binary operators can be defined using `make_boperator`(`tf.math.add` is a function in exactly 2 arguments):
    ```python
    import tensorflow as tf
    from libmg import make_boperator
    add = make_boperator(tf.math.add)
    ```
  Usage in $\mu\mathcal{G}$, assuming the compiler was passed the $\psi$ function dictionary `#!python {'+': add, 'a': ..., 'b': ...}`:
  
    $$
    +(a, b)
    $$

* K-ary operators can be defined using `make_koperator`(`tf.math.add_n` is a function in $n$ arguments):
```python
import tensorflow as tf
from libmg import make_koperator
addk = make_koperator(tf.math.add_n)
```
  Usage in $\mu\mathcal{G}$, assuming the compiler was passed the $\psi$ function dictionary `#!python {'+': addk, 'a': ..., 'b': ..., 'c': ...}`:
  
    $$
    +(a, b, c)
    $$

The operators attempt to automatically find their arguments given the node label, which is always a rank-1 tensor. Unary operators are applied directly 
to the node labels in the same way as other $\psi$ functions. Binary operators split in half the node labels tensor, and consider the first partition as 
the left operand and the second partition as the right operand. K-ary operators split the node labels evenly in $k$ slices and treat each slice as an 
operand.