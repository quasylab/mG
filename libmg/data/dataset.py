"""Defines a container for graphs.

This module defines a container (a dataset) of graphs, based on the class of the same name in the Spektral library.

The module contains the following classes:

- `Dataset(name, **kwargs)`
"""
from typing import Any

from spektral.data import Dataset as _Dataset


class Dataset(_Dataset):
    """Container for graphs.

    This class is supposed to be extended by overriding the `read` and `download` methods. See Spektral's [documentation](
    https://graphneural.network/creating-dataset) for additional information, as this class is directly derived from Spektral's ``Dataset`` class.

    Attributes:
        name: A string name for the dataset.
    """
    def __init__(self, name: str, **kwargs: Any):
        """Initializes the instance with the given name, then the superclass will call the ``download`` and ``read`` methods as needed.

        Args:
            name: The name for the dataset.
            **kwargs: The keyword arguments to pass to Spektral's ``Dataset`` class constructor.
        """
        self.name = name
        super().__init__(**kwargs)

    def read(self):
        raise NotImplementedError

    def download(self):
        pass
