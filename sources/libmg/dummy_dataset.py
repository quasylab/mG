import numpy as np
from scipy.sparse import coo_matrix

from spektral.data import Dataset, Graph


class DummyDataset(Dataset):
    def __init__(self, num_node_features, type_node_features, type_adj_matrix, num_edge_features=None, type_edge_features=None, **kwargs):
        self.num_node_features = num_node_features
        self.type_node_features = type_node_features
        self.num_edge_features = num_edge_features
        self.type_edge_features = type_edge_features
        self.type_adj_matrix = type_adj_matrix
        super().__init__(**kwargs)

    def download(self):
        pass

    def read(self):
        dummy_node_value = [0] + [0] * (self.num_node_features - 1)
        x = np.array([dummy_node_value, dummy_node_value], dtype=self.type_node_features.as_numpy_dtype)
        a = coo_matrix(([1, 1, 1, 1], ([0, 0, 1, 1], [0, 1, 0, 1])), shape=(2, 2), dtype=self.type_adj_matrix.as_numpy_dtype)
        if self.num_edge_features is not None and self.type_edge_features is not None:
            dummy_edge_value = [0] + [0] * (self.num_edge_features - 1)
            e = np.array([dummy_edge_value, dummy_edge_value, dummy_edge_value, dummy_edge_value], dtype=self.type_edge_features.as_numpy_dtype)
        else:
            e = None
        return [Graph(x=x, a=a, e=e)]
