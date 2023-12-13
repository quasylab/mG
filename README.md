# <span style="font-variant:small-caps;">libmg</span>

[![Tests](https://github.com/quasylab/mG/actions/workflows/testing.yml/badge.svg)](https://github.com/Unicam-mG/mG/actions/workflows/testing.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/libmg?logo=pypi?stile=flat)](https://pypi.org/project/libmg/)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg?style=flat)](http://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://mit-license.org/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/libmg?logo=python)
[![tensorflow](https://img.shields.io/badge/TensorFlow-2.12-FF6F00.svg?style=flat&logo=tensorflow)](https://www.tensorflow.org)


<span style="font-variant:small-caps;">libmg</span> is a Python library for compiling μG expressions into TensorFlow model. It allows the parsing, unparsing,
normalization, compilation of μG expressions. It also provides functionalities to visualize graphs and model outputs.

## Pre-requisites

* A Linux operating system (preferably Ubuntu 16.04 or later as per the TensorFlow recommendation).
* Python 3.11 environment.

The library can run both on the CPU or the GPU. To enable the GPU, the specific dependencies needed are those of TensorFlow 2.12, that is:

* GCC 9.3.1
* Bazel 5.3.0
* NVIDIA GPU drivers version 450.80.02 or higher
* CUDA 11.8
* cuDNN 8.6
* (Optional) TensorRT 7

## Installation
<span style="font-variant:small-caps;">libmg</span> can be installed via pip or from source.

### Pip installation

<span style="font-variant:small-caps;">libmg</span> can be installed from the Python Package Index PyPI, by simply running the following command in your 
shell or virtual environment:

``` commandline
$ pip install libmg
```

### Source installation

You can install <span style="font-variant:small-caps;">libmg</span> from source using git. You can start by downloading the repo archive or by cloning the repo:

```commandline
git clone https://github.com/quasylab/mG.git
```

Then proceed by opening a shell into the `mG` directory you have just downloaded. To build the library you will need to use [Poetry](https://python-poetry.
org/). Run the following command:

```commandline
poetry install
```
and Poetry will install <span style="font-variant:small-caps;">libmg</span> in your Python environment. To install the development dependencies as well, install
with:

```commandline
poetry install --with tests --with docs
```

This will add the testing dependencies (pytest, mypy, and flake8) as well as the documentation dependencies (mkdocs and plugins).

## Usage
- Create a `Dataset` object with the `Graph` instances to process.
- Define dictionaries of `Psi`, `Phi`, `Sigma` objects as needed by your application.
- Define a `CompilerConfig` that is adequate for the graphs in your `Dataset`
- Create a `MGCompiler` using the dictionaries and the `CompilerConfig`
- Create an adequate `Loader` for your `Dataset`: use the `SingleGraphLoader` if your dataset contains a single graph and use the `MultipleGraphLoader` 
  otherwise.
- Build a model from your μG formulas using the compiler's `compile(expr)` method.
- Train your model as you would in Tensorflow
- Use `output = model.predict(loader.load(), steps=loader.steps_per_epoch)` or a loop like
    ```
    for x in loader.load():
        output = model(x)
    ```
  to run your model on the dataset.
- Visualize the outputs on the browser using `print_layer(model, inputs, layer_idx=-1)`

## Documentation
You can find the official documentation [here](https://quasylab.github.io/mG/).

## Research articles
Matteo Belenchia, Flavio Corradini, Michela Quadrini, and Michele Loreti. 2023. Implementing a CTL Model Checker with μG, a Language for Programming
Graph Neural Networks. In Formal Techniques for Distributed Objects, Components, and Systems: 43rd IFIP WG 6.1 International Conference, FORTE 2023,
Held as Part of the 18th International Federated Conference on Distributed Computing Techniques, DisCoTec 2023, Lisbon, Portugal, June 19–23, 2023,
Proceedings. Springer-Verlag, Berlin, Heidelberg, 37–54. <https://doi.org/10.1007/978-3-031-35355-0_4>.
Preprint: <https://www.researchgate.net/publication/371467699_Implementing_a_CTL_Model_Checker_with_mu_mathcal_G_a_Language_for_Programming_Graph_Neural_Networks>