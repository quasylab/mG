from __future__ import annotations
from spektral.data import Dataset as _Dataset


class Dataset(_Dataset):
    """
    A ``Dataset`` with the given name.
    See Spektral's `documentation <https://graphneural.network/creating-dataset/>`_ for additional information,
    as this class is directly derived from Spektral's ``Dataset``.

    :param name: A name for the dataset.
    """

    def __init__(self, name, **kwargs):
        self._name = name
        super().__init__(**kwargs)

    def read(self):
        raise NotImplementedError

    def download(self):
        pass

    @property
    def name(self):
        return self._name
