# How to define a dataset

This guide shows you how to define datasets of graphs, which are the inputs of $\mu\mathcal{G}$ models. `Dataset`s are containers for one or more `Graph` 
objects, and they are implemented in [Spektral](https://graphneural.network/creating-dataset/). <span style="font-variant:small-caps;">libmg</span> imports 
their implementation, and in fact you will see that this guide is very similar to the one linked above. Datasets are defined by subclassing them and 
overriding the `read` and `download` methods. Then they can be instantiated by providing a name. In the following sections, we will go over these steps.

## Defining the dataset

We start by importing the `Dataset` and `Graph` classes from `libmg`. We will also be importing `os`, `numpy` and `scipy` which will be useful later.

```python
import os
import numpy as np
from scipy.sparse import coo_matrix
from libmg import Dataset, Graph
```

We can define a new dataset by subclassing `Dataset`. In the `#!python __init__` method we are only required to pass a string name to the parent class, but as
usual we can also provide additional arguments that we may need. 

```python
class MyDataset(Dataset):
    def __init__(self, name, arg1, arg2, ...):
        super().__init__(name)
        self.arg1 = arg1
        self.arg2 = arg2
        ...
    
    def read(self):
        pass

    def download(self):
        pass
```
When a `Dataset` is instantiated, the `__init__` will call `download` first (if necessary, see below), followed by `read`. The list of `Graph` objects 
returned by `read` will constitute the contents of our dataset.

The `download` method is supposed to create the raw data of the dataset. It is called if a directory named `~/spektral/datasets/[ClassName]` is missing. In such
directory the `download` method should store the data, so that in future instantiations `read` can load this data without calling `download` again. Thus, the 
`download` method will usually create this directory and save some data there, e.g. `.npz` files, `.csv` files, etc.

The `read` method is called on every instantiation and must return a `list` of `Graph` objects. If we defined a `download` method, these `Graph` objects will 
usually come from the files we saved in the `~/spektral/datasets/[ClassName]` directory. If we didn't define a `download` method, we also have the possibility of 
generating these graphs on-the-fly.

### Defining a `Graph`
A `Graph` object can be instantiated using the constructor `Graph(x=None, a=None, e=None, y=None)`. All these four arguments are optional, but
in $\mu\mathcal{G}$ you should always at least provide `x` and `a`. 

#### The node features matrix `X`
Each node in the graph is assigned a vector of features. For example, when considering a citation network, each node represents a paper and will be assigned a 
vector of floating-point numbers that encode the contents of that paper.

The node features matrix `X` stores all these vectors, such that in row $i$ is stored the feature vector for node $i$. Therefore, this matrix will have rows 
equal to the number of nodes in the graph, and columns equal to the length of the feature vectors (which will all have the same length). 

When creating a `Graph`, the `x` argument that encodes the node features matrix should be passed in as a NumPy array (`np.array`).

#### The adjacency matrix `A`
The adjacency matrix encodes the connections of a graph. In $\mu\mathcal{G}$ adjacency matrices are binary, i.e. they only contain zeros and ones. A value of 
1 at row $i$ and column $j$ means that there exists a directed edge going from node $i$ to node $j$. A value of 0 means that there is no such edge. 

The adjacency matrix is supposed to be created as a SciPy sparse matrix in [COOrdinate format](https://scipy-lectures.org/advanced/scipy_sparse/coo_matrix.html)
(`coo_matrix` from `scipy.sparse`). A sparse matrix only contains the coordinates of the non-zero elements. In this format the adjacency matrix is specified 
from three arrays: the row indices, the column indices, and the values. The values will be an array of 1s equal to the number of edges in the graph. The row and
column indices are the indices of the nodes that the edges connect: the row index is the source node of the edge and the column index is the target node of 
the edge. The indices should be in row-major order, that is, they are ordered according to the rows first and to the columns second.


#### The edge features matrix `E`
As is the case for the nodes, edges can have features as well. The edge features matrix `E` has a row for each edge in the graph and columns equal to the length
of their feature vectors. In this format, the feature vector in row $i$ corresponds to the $i$-th edge of the graph. The $i$-th edge of the graph is the 
edge corresponding to the $i$-th row index and column index of the adjacency matrix $A$.

When creating a `Graph`, the `e` argument that encodes the edge features matrix should be passed in as a NumPy array (`np.array`).


#### The true labels matrix `Y`
Usually in machine learning we are also provided the true labels to be used for training or testing models. In $\mu\mathcal{G}$ the labels are always meant to
be node labels, that is, we are given a label for each node of the graph. 

The true labels features matrix `Y` stores the true labels vector, such that in row $i$ is stored the true labels vector for node $i$. Therefore, this matrix 
will have rows equal to the number of nodes in the graph, and columns equal to the length of the true label vectors. 

When creating a `Graph`, the `y` argument that encodes the true labels matrix should be passed in as a NumPy array (`np.array`).

### Overriding `download`
The `download` method will usually create the `~/spektral/datasets/[ClassName]` (available through `self.path`) directory and populate it with data. The data 
can be generated according to some specification or downloaded from the web. So the general structure of a `download` method will be:

```python
def download(self):
    os.mkdir(self.path)
    data = ...  # Obtain the data

    # Save the data
    np.savez('mydata', ...)
```

### Overriding `read`
The `read` method will either load up the data in `~/spektral/datasets/[ClassName]` or create it on-the-fly. What it matters is that it returns a `list` of
`Graph` objects.

```python
def read(self):
    output = []
    mydata = np.load(os.path.join(self.path, 'mydata.npz'))
    
    ...
    
    X = np.array(...)
    A = coo_matrix(...)
    output.append(Graph(x=X, a=A))
    
    ...
    
    return output
```

## Instantiating the dataset

The dataset can now be instantiated by calling the constructor. 
```python
mydataset = MyDataset('MyFirstDataset', ...)
```