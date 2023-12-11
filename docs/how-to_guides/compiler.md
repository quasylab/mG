# How to define a compiler

This guide shows you how to create a compiler instance for $\mu\mathcal{G}$ expressions. For this purpose we will need to import the following:

```python
from libmg import MGCompiler, CompilerConfig, NodeConfig, EdgeConfig
```
## Node and edge configuration

We start by creating the `NodeConfig` and, if needed, the `EdgeConfig` instances. The `NodeConfig` specifies the type and size of the feature vectors 
associated to each node, that is, the elements on each row of the node features matrix $X$. If, for example, we have that each node is assigned a 
floating-point value, we will have 

```python
node_conf = NodeConfig(tf.float32, 1)
```

The first argument for `NodeConfig` is the type, expressed using TensorFlow types, while the latter is the integer value specifying the size of the feature 
vectors. In the same way we can specify the type and size of the feature vectors associated to each edge, if any. For example, an edge configuration that 
specifies two Boolean values on each edge is:

```python
edge_conf = EdgeConfig(tf.bool, 2)
```

## Compiler configuration
The `CompilerConfig` instance will hold the `NodeConfig` and `EdgeConfig` objects, plus the type of values in the adjacency matrix and the dictionary of the 
tolerance values for the fixpoint computations. The type of the adjacency matrix entries will typically be integers, so `tf.int32` or even `tf.uint8` are 
recommended, but the correct type to use depends on the way the adjacency matrix was defined in the `Graph` objects. 

The tolerance values are specified as a dictionary, where the key is the name of a numeric type (e.g. `float32`, `int64`, etc.) and the value is a 
floating-point number, typically in the order of the thousandth (0.001) to the billionth (0.000000001), specifying the maximum absolute tolerance between two 
values to declare them to be the same value.

The `CompilerConfig` instance is supposed to be created by using one of the static constructor methods offered:

* `xa_config(node_config: NodeConfig, matrix_type: tf.DType, tolerance: dict[str, float])` with alias `single_graph_no_edges_config`
* `xai_config(node_config: NodeConfig, matrix_type: tf.DType, tolerance: dict[str, float])` with alias `multiple_graphs_no_edges_config`
* `xae_config(node_config: NodeConfig, edge_config: EdgeConfig, matrix_type: tf.DType, tolerance: dict[str, float])` with alias 
  `single_graph_with_edges_config` 
* `xaei_config(node_config: NodeConfig, edge_config: EdgeConfig, matrix_type: tf.DType, tolerance: dict[str, float])` with alias `multiple_graphs_with_edges_config`

The correct method to use depends on the graphs and loader that will be used with the compiler. The `xa_config` method is used for datasets containing a 
single graph with no edge labels, `xae_config` is for datasets containing a single graph with edge labels, and `xai_config` and `xaei_config` are used for 
datasets containing multiple graphs. To create the `CompilerConfig` simply call the correct method based on your use case by passing in the required 
`NodeConfig`, `EdgeConfig` and the other parameters.

```python
conf = CompilerConfig.xae_config(node_conf, edge_conf, tf.uint8, {'float32': 0.001})
```
## Instantiating the compiler

The `MGCompiler` class is instantiated by passing in the dictionaries of the $\psi$, $\varphi$, and $\sigma$ functions (see [How to define functions](.
/functions.md)) and the `CompilerConfig` instance.

```python
compiler = MGCompiler(psi_functions=..., phi_functions=..., sigma_functions=..., config=conf)
```