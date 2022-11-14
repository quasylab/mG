import os
import time
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"


class PerformanceTest:
    def __init__(self, model_constructor, loader_constructor):
        self.model_constructor = model_constructor
        self.loader_constructor = loader_constructor

    def __call__(self, dataset):
        raise NotImplementedError


class PredictPerformance(PerformanceTest):
    def __call__(self, dataset):
        loader = self.loader_constructor(dataset)
        model = self.model_constructor(dataset)
        start = time.perf_counter()
        model.predict(loader.load(), steps=loader.steps_per_epoch)
        end = time.perf_counter()
        print("Using model.predict", end - start, sep=' ')
        return end - start


class CallPerformance(PerformanceTest):
    def __call__(self, dataset):
        loader = self.loader_constructor(dataset)
        model = self.model_constructor(dataset)
        tot = 0
        for x, y in loader.load():
            start = time.perf_counter()
            model(x)
            end = time.perf_counter()
            tot += end - start
        print("Using model.__call__ and tf.function", tot, sep=' ')
        return tot


def save_output_to_csv(dataset_generator, methods, names, filename):
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

    filename = '../../../data/' + filename + '.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        w = csv.writer(f)
        w.writerow(labels)
        for row in values:
            w.writerow(row)