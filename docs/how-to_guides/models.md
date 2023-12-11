# How to create, run and optimize models

This guide shows you how to create models using the $\mu\mathcal{G}$ compiler, how to run them on your datasets, and how to let the compiler trace them before
you run them.

## Creating a model

To create a model, simply pass its $\mu\mathcal{G}$ expression to the `compile` method on the `MGCompiler` instance. 

```python
compiler = MGCompiler(...)
model = compiler.compile('(a || b);c')
```

It is possible to automatically print a summary of the compiled model by setting the `verbose` argument to `True`:

```python
model = compiler.compile('(a || b);c', verbose=True)
```

Finally, to enable automatic memoization, set the `memoize` argument to `True`:

```python
# 'a' is computed only once inside the model
model = compiler.compile('((a || b);c) || a', memoize=True)  
```

## Running a model

The model can be run using any of the TensorFlow APIs that exist for this purpose: directly calling the model, using `call`, using `predict`, or using 
`predict_on_batch`. In each of these cases, it is necessary to use the adequate loader for the dataset we will be feeding into the model. For datasets 
consisting of a single graph we will be using the `SingleGraphLoader`, while for datasets consisting of multiple graphs, we will use the 
`MultipleGraphLoader`. Both of these loaders are implemented in [Spektral](https://graphneural.network/loaders/). The loaders convert a graph, or a batch of 
graphs, into a list of TensorFlow `Tensor` objects, one each corresponding to the `x`, `a`, `e` and `y` arguments used to define the graph. The output of a 
`SingleGraphLoader` will be a two-element tuple `((x, a, [e]), y)` if `y` was specified in the graph, otherwise it will be a one-element tuple `((x, a, [e]),)`.
The `MultipleGraphLoader` shares the same output structure as the `SingleGraphLoader`, additionally adding a `i` tensor to the tail of the three-element 
tuple on the left (e.g. `((x, a, [e], i), y)`). The meaning of this tensor will be explained below.

!!! note
    The loader chosen here must correspond the `CompilerConfig` object we used to instantiate the compiler.

To load a dataset with the `SingleGraphLoader` pass the dataset to it:

```python
from libmg import SingleGraphLoader

dataset = MyDataset(...)
loader = SingleGraphLoader(dataset)
```

It is possible to specify the number of `epochs` that the loader will generate. Set epochs to 1 if you want that the loader simply returns the single graph in 
the dataset once. If epochs is left as `None`, the loader will keep returning the graph every time it is called.

```python
loader = SingleGraphLoader(dataset, epochs=1)
```

Similarly, to load a dataset with the `MultipleGraphLoader`, just pass the dataset to it.

```python
from libmg import MultipleGraphLoader

dataset = MyDataset(...)
loader = MultipleGraphLoader(dataset)
```

It is again possible to specify the number of `epochs`. This time, since the dataset used with this loader has more than one graph, the epochs number specify how
many times to cycle through all the graphs. Set it to 1 to just show each graph once, `None` to keep cycling, or any other integer to cycle that amount of 
times. The number of graphs to batch together is specified with the `batch_size` argument. For example, if `batch_size=2` and the dataset has 10 graphs, the 
loader will return 5 batches of 2 graphs. Each batch is structured according to Spektral's [disjoint mode](https://graphneural.network/data-modes/#disjoint-mode).
The loader therefore adds an `i` Tensor to its outputs. This tensor is used to specify to which graph belongs each row of the node features matrix `x`. For 
example, if we batched two graphs and the `x` features matrix is `[[1], [2], [3], [4]]` (each node has one integer feature) and `i` is `[0, 0, 1, 1]` it 
means that the first graph in the batch has two nodes with features `[1]` and `[2]` and the second graph has two nodes with features `[3]` and `[4]`.

```python
loader = MultipleGraphLoader(dataset, batch_size=2, epochs=1)
```

### Using `call` or by directly calling the model

When using `call` on the model, we should obtain the graphs from the loader by iterating on the `load` method:

```python
for inputs in loader.load():
    model.call(inputs)
```

equivalently, we can also directly call the model:

```python
for inputs in loader.load():
    model(inputs)
```

If there is only one graph, it can also be obtained directly with:

```python
inputs = next(iter(loader.load()))
```

If the true labels are present, they are not to be passed as inputs to the model:

```python
for inputs, y in loader.load():
    model.call(inputs)
```
```python
inputs, y = next(iter(loader.load()))
```

### Using `predict`
Using `predict`, you should fill in the `steps` arguments with the `steps_per_epoch` attribute of the loader:

```python
outputs = model.predict(loader.load(), steps=loader.steps_per_epoch)
```

### Using `predict_on_batch`
Using `predict_on_batch`, you should just pass the batch of graphs:

```python
outputs = model.predict_on_batch(next(iter(loader.load())))
```

!!! warning
    
    For the time being, due to TensorFlow assertion checks, it is not possible to use the `predict_on_batch` API on datasets whose graphs have edge labels.

## Training a model

Models can be trained in the usual manner as in TensorFlow. The model must first be compiled by TensorFlow (a distinct operation from compiling in 
$\mu\mathcal{G}$), specifying the optimizer and the loss function:

```python
# Using stochastic gradient descent and mean squared error
model.compile(optimizer='sgd', loss='mse')
```

Then, we can train the model using `fit`, by specifying the `steps_per_epoch` using the loader and the number of epochs:
```python
model.fit(loader.load(), steps_per_epoch=loader.steps_per_epoch, epochs=100)
```

!!! note
    The loader in this case should have been instantiated with `epochs=None`, since the `fit` method will specify the epochs.

## Tracing a model

Tracing is performed automatically by running the model on some data. At any rate, it is possible to create dummy data which the compiler will use to trace 
the model for you. In order to do that, we simply call the `trace` method on the compiler by passing in the model and the API we will be using to run the 
model (either `call`, `predict`, or `predict_on_batch`):

```python
# Tracing the model for the 'predict' API
traced_model = compiler.trace(model, 'predict')
```

!!! warning
    The `trace` method with the second argument set to `call` returns a `@tf.function` that runs `model.call` on its inputs. Therefore the return value is 
    no longer a TensorFlow model, but a Python `Callable` and the typical TensorFlow methods (`predict`, `fit`, etc.) and the <span 
    style="font-variant:small-caps;">libmg</span> model attributes will be no longer available.