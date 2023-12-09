"""Defines a container for graphs and the means to process them in TensorFlow.

This package defines a container (a dataset) of graphs, based on the class of the same name in [Spektral](https://graphneural.network/data/#dataset).
The package also defines two loaders for datasets, based on the loaders defined in [Spektral](https://graphneural.network/loaders/).
The `Graph` class from [Spektral](https://graphneural.network/data/#graph) can be imported from this package, and used for the definition of datasets.

The package contains the following classes:

- `Graph(x=None, a=None, e=None, y=None, **kwargs)`
- `Dataset(name, **kwargs)`
- ``SingleGraphLoader(dataset, epochs=None, sample_weights=None)``
- ``MultipleGraphLoader(dataset, node_level=False, batch_size=1, epochs=None, shuffle=True)``
"""

from spektral.data import Graph
from .dataset import Dataset
from .loaders import SingleGraphLoader, MultipleGraphLoader

__all__ = ['Graph', 'Dataset', 'SingleGraphLoader', 'MultipleGraphLoader']
