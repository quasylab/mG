from spektral.data import Dataset as _Dataset


class Dataset(_Dataset):
    """
       A container for Graph objects. This class can be extended to represent a
       graph dataset.

       To create a `Dataset`, you must implement the `Dataset.read()` method, which
       must return a list of `spektral.data.Graph` objects:

       ``py
       class MyDataset(Dataset):
           def read(self):
               return [Graph(x=x, adj=adj, y=y) for x, adj, y in some_magic_list]
       ```

       The `download()` method is automatically called if the path returned by
       `Dataset.path` does not exists (default `~/.spektral/datasets/ClassName/`).

       In this case, `download()` will be called before `read()`.

       Datasets should generally behave like Numpy arrays for any operation that
       uses simple 1D indexing:

       ```py

       Graph(...)

       >> dataset[[1, 2, 3]]
       Dataset(n_graphs=3)

       >> dataset[1:10]
       Dataset(n_graphs=9)

       >> np.random.shuffle(dataset)  # shuffle in-place

       >> for graph in dataset[:3]:
       >>     print(graph)
       Graph(...)
       Graph(...)
       Graph(...)
       ```

       Datasets have the following properties that are automatically computed:

           - `n_nodes`: the number of nodes in the dataset (always None, except
           in single and mixed mode datasets);
           - `n_node_features`: the size of the node features (assumed to be equal
           for all graphs);
           - `n_edge_features`: the size of the edge features (assumed to be equal
           for all graphs);
           - `n_labels`: the size of the labels (assumed to be equal for all
           graphs); this is computed as `y.shape[-1]`.

       Any additional `kwargs` passed to the constructor will be automatically
       assigned as instance attributes of the dataset.

       Datasets also offer three main manipulation functions to apply callables to
       their graphs:

       - `apply(transform)`: replaces each graph with the output of `transform(graph)`.
       See `spektral.transforms` for some ready-to-use transforms.<br>
       Example: `apply(spektral.transforms.NormalizeAdj())` normalizes the
       adjacency matrix of each graph in the dataset.
       - `map(transform, reduce=None)`: returns a list containing the output
       of `transform(graph)` for each graph. If `reduce` is a `callable`, then
       returns `reduce(output_list)`.<br>
       Example: `map(lambda: g.n_nodes, reduce=np.mean)` will return the
       average number of nodes in the dataset.
       - `filter(function)`: removes from the dataset any graph for which
       `function(graph) is False`.<br>
       Example: `filter(lambda: g.n_nodes < 100)` removes from the dataset all
       graphs bigger than 100 nodes.

       Datasets in mixed mode (one adjacency matrix, many instances of node features)
       are expected to have a particular structure.
       The graphs returned by `read()` should not have an adjacency matrix,
       which should be instead stored as a singleton in the dataset's `a` attribute.
       For example:

       ```py
       class MyMixedModeDataset(Dataset):
           def read(self):
               self.a = compute_adjacency_matrix()
               return [Graph(x=x, y=y) for x, y in some_magic_list]
       ```
       """
    def __init__(self, name, transforms=None, **kwargs):
        """
        Creates a Dataset object with the given name

        :param name: a name for the dataset
        :type name: str
        :param transforms: a callable or list of callables that are automatically applied to the graphs after loading
         the dataset.
        :type transforms: typing.Callable | list[typing.Callable]
        :param kwargs:
        :type kwargs:
        """
        self._name = name
        super().__init__(transforms, **kwargs)

    def read(self):
        raise NotImplementedError

    def download(self):
        pass

    @property
    def name(self):
        return self._name
