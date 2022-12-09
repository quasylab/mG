from typing import Callable, Iterable
import tensorflow as tf
import os
import time
import csv

from .dataset import Dataset
from .loaders import SingleGraphLoader, MultipleGraphLoader


class PerformanceTest:
    def __init__(self, model_constructor: Callable[[Dataset], tf.keras.Model],
                 loader_constructor: Callable[[Dataset], SingleGraphLoader | MultipleGraphLoader]):
        """
        Base class for measuring performance of the model by overriding the __call__ method

        :param model_constructor: A function from a Dataset to a Model
        :param loader_constructor: A function from a Dataset to a SingleGraphLoader or MultipleGraphLoader
        :returns: A PerformanceTest object
        """
        self.model_constructor = model_constructor
        self.loader_constructor = loader_constructor

    def __call__(self, dataset):
        raise NotImplementedError


class PredictPerformance(PerformanceTest):
    def __call__(self, dataset: Dataset) -> float:
        """
        Builds a model and a loader given the dataset, then runs and times model.predict

        :param dataset: A dataset on which to measure the model's performance
        :return: Execution time in seconds
        """
        loader = self.loader_constructor(dataset)
        model = self.model_constructor(dataset)
        start = time.perf_counter()
        model.predict(loader.load(), steps=loader.steps_per_epoch)
        end = time.perf_counter()
        print("Using model.predict", end - start, sep=' ')
        return end - start


class CallPerformance(PerformanceTest):
    def __call__(self, dataset: Dataset) -> float:
        """
        Builds a model and a loader given the dataset, then runs and times model.call on each element of the dataset

        :param dataset: A dataset on which to measure the model's performance
        :return: Execution time in seconds
        """
        loader = self.loader_constructor(dataset)
        model = self.model_constructor(dataset)
        tot = 0.0
        for x, y in loader.load():
            start = time.perf_counter()
            model(x)
            end = time.perf_counter()
            tot += end - start
        print("Using model.__call__ and tf.function", tot, sep=' ')
        return tot


def save_output_to_csv(dataset_generator: Iterable[Dataset], methods: list[PerformanceTest], names: list[str],
                       filename: str) -> None:
    """
    For each dataset in `dataset_generator`, every method in `methods` is called on the given dataset. The output is
    collected row by row, each column labelled with the name provided in `names`. The resulting data is saved in a csv
    file with name given by `filename` in the 'data' folder.

    :param dataset_generator: An iterable of datasets
    :param methods: A list of PerformanceTest objects to call
    :param names: A list of names, corresponding to each method
    :param filename: The name of the file where to save the data
    :return: Nothing
    """
    labels = ['index'] + names
    values = []
    for dataset in dataset_generator:
        print('Evaluating dataset: ', dataset.name)
        row = [dataset.name]
        for i in range(len(methods)):
            out = methods[i](dataset)
            if type(out) is tuple:
                row.extend(out)
            else:
                row.append(out)
        values.append(row)

    filename = 'data/' + filename + '.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        w = csv.writer(f)
        w.writerow(labels)
        for row in values:
            w.writerow(row)
