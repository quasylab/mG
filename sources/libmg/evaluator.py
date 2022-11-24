import os
import time
import csv


class PerformanceTest:
    def __init__(self, model_constructor, loader_constructor):
        """
        Base class for measuring performance of the model by overriding the __call__ method

        :param model_constructor: A function from a Dataset to a Model
        :type model_constructor: (libmg.Dataset) -> tensorflow.keras.Model
        :param loader_constructor: A function from a Dataset to a SingleGraphLoader or MultipleGraphLoader
        :type loader_constructor: (libmg.Dataset) -> libmg.loaders.SingleGraphLoader |
         libmg.loaders.MultipleGraphLoader
        :returns: A PerformanceTest object
        :rtype: PerformanceTest
        """
        self.model_constructor = model_constructor
        self.loader_constructor = loader_constructor

    def __call__(self, dataset):
        raise NotImplementedError


class PredictPerformance(PerformanceTest):
    def __call__(self, dataset):
        """
        Builds a model and a loader given the dataset, then runs and times model.predict

        :param dataset: A dataset on which to measure the model's performance
        :type dataset: libmg.Dataset
        :return: Execution time in seconds
        :rtype: float
        """
        loader = self.loader_constructor(dataset)
        model = self.model_constructor(dataset)
        start = time.perf_counter()
        model.predict(loader.load(), steps=loader.steps_per_epoch)
        end = time.perf_counter()
        print("Using model.predict", end - start, sep=' ')
        return end - start


class CallPerformance(PerformanceTest):
    def __call__(self, dataset):
        """
        Builds a model and a loader given the dataset, then runs and times model.call on each element of the dataset

        :param dataset: A dataset on which to measure the model's performance
        :type dataset: libmg.Dataset
        :return: Execution time in seconds
        :rtype: float
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


def save_output_to_csv(dataset_generator, methods, names, filename):
    """

    :param dataset_generator: An iterable of datasets
    :type dataset_generator: typing.Iterable[libmg.Dataset]
    :param methods: A list of PerformanceTest objects to call
    :type methods: list[PerformanceTest]
    :param names: A list of names, corresponding to each method
    :type names: list[str]
    :param filename: The name of the file where to save the data
    :type filename: str
    :return: Nothing
    :rtype: None
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
