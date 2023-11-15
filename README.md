# mG

[![Tests](https://github.com/quasylab/mG/actions/workflows/testing.yml/badge.svg)](https://github.com/Unicam-mG/mG/actions/workflows/testing.yml)

## Usage
- Install libmg by running `pip install git+https://github.com/Unicam-mG/mG.git`
- Create a `Dataset` object containing the `Graph` instances to process
- Define dictionaries of `Psi`, `Phi`, `Sigma` objects as needed by your application
- Define a `CompilationConfig` that is appropriate for your `Dataset`
- Create a `GNNCompiler` using the dictionaries and the `CompilationConfig`
- Create an appropriate `Loader` for your `Dataset`: use the `SingleGraphLoader` if your Dataset contains a single graph and use the `MultipleGraphLoader` otherwise.
- Build a model from your mG formulas using the `model = GNNCompiler.compile(expr)` method.
- Use `output = model.predict(loader.load(), steps=loader.steps_per_epoch)` or a loop like
    ```
    for x, y in loader.load():
        output = model(x)
    ```
  to run your model on the dataset.
- Check the tests folder for some examples of the above steps.

## Compatibility
Python 3.10
