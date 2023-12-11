# Installation

This guide shows you how to install <span style="font-variant:small-caps;">libmg</span>.

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

## Verify the installation

To verify your installation, you can print the library's version number:

```pycon
>>> import libmg
>>> libmg.__version__
'1.0.5'
```

Additionally, it is possible to run the <span style="font-variant:small-caps;">libmg</span> test suite. For that, it is necessary to have
[pytest](https://doc.pytest.org/en/latest/) and [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/) installed.

If you are installing from source, these dependencies can be installed by running `poetry install` with the `--with tests` option.

The tests can be run using the `run_tests` function:

```pycon
>>> import libmg
>>> libmg.run_tests()
============================= test session starts ==============================
platform linux -- Python 3.11.6, pytest-7.4.3, pluggy-1.3.0
rootdir: /home/matteo/PycharmProjects/mG
configfile: pyproject.toml
testpaths: libmg
plugins: cov-4.1.0
collected 169 items
libmg/compiler/functions.py ...........                                  [  6%]
libmg/language/grammar.py .                                              [  7%]
libmg/normalizer/normalizer.py .                                         [  7%]
libmg/tests/test_compiler.py ............s...                            [ 17%]
libmg/tests/test_cuda.py .s                                              [ 18%]
libmg/tests/test_explainer.py .s                                         [ 19%]
libmg/tests/test_functions.py ...............................s..s..s..s. [ 44%]
.s..s.s...s                                                              [ 50%]
libmg/tests/test_grammar.py .................................            [ 70%]
libmg/tests/test_normalizer.py ......................................... [ 94%]
......                                                                   [ 98%]
libmg/tests/test_visualizer.py ..s                                       [100%]
---------- coverage: platform linux, python 3.11.6-final-0 -----------
Name                             Stmts   Miss  Cover
----------------------------------------------------
libmg/__init__.py                   10     10     0%
libmg/compiler/__init__.py           3      3     0%
libmg/compiler/compiler.py         680    157    77%
libmg/compiler/functions.py        171     69    60%
libmg/compiler/layers.py           172     50    71%
libmg/data/__init__.py               4      4     0%
libmg/data/dataset.py               10      7    30%
libmg/data/loaders.py               51     23    55%
libmg/explainer/__init__.py          2      2     0%
libmg/explainer/explainer.py       109     34    69%
libmg/language/__init__.py           2      2     0%
libmg/language/grammar.py           23     12    48%
libmg/normalizer/__init__.py         2      2     0%
libmg/normalizer/normalizer.py     105     36    66%
libmg/tests/__init__.py              2      2     0%
libmg/tests/run_tests.py             3      3     0%
libmg/tests/test_compiler.py       174      0   100%
libmg/tests/test_cuda.py             7      0   100%
libmg/tests/test_explainer.py       38      0   100%
libmg/tests/test_functions.py      213      3    99%
libmg/tests/test_grammar.py        147      0   100%
libmg/tests/test_normalizer.py      21      0   100%
libmg/tests/test_visualizer.py      46      0   100%
libmg/visualizer/__init__.py         2      2     0%
libmg/visualizer/visualizer.py      94     24    74%
----------------------------------------------------
TOTAL                             2091    445    79%
================== 157 passed, 12 skipped in 92.74s (0:01:32) ==================
```

