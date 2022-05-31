# Bayesian Optimization

[![PyPI](https://img.shields.io/pypi/v/bayes-opt)](https://pypi.org/project/bayes-opt/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/bayes-opt)](https://pypi.org/project/bayes-opt/)
![Loc](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/HansBug/d9b45d4c1b12045384046990db092098/raw/loc.json)
![Comments](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/HansBug/d9b45d4c1b12045384046990db092098/raw/comments.json)

[![Code Test](https://github.com/HansBug/bayes-opt/workflows/Code%20Test/badge.svg)](https://github.com/HansBug/bayes-opt/actions?query=workflow%3A%22Code+Test%22)
[![Badge Creation](https://github.com/HansBug/bayes-opt/workflows/Badge%20Creation/badge.svg)](https://github.com/HansBug/bayes-opt/actions?query=workflow%3A%22Badge+Creation%22)
[![Package Release](https://github.com/HansBug/bayes-opt/workflows/Package%20Release/badge.svg)](https://github.com/HansBug/bayes-opt/actions?query=workflow%3A%22Package+Release%22)
[![codecov](https://codecov.io/gh/HansBug/BayesianOptimization/branch/main/graph/badge.svg?token=LGB44A91FL)](https://codecov.io/gh/HansBug/BayesianOptimization)

[![GitHub stars](https://img.shields.io/github/stars/HansBug/bayes-opt)](https://github.com/HansBug/bayes-opt/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/HansBug/bayes-opt)](https://github.com/HansBug/bayes-opt/network)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/HansBug/bayes-opt)
[![GitHub issues](https://img.shields.io/github/issues/HansBug/bayes-opt)](https://github.com/HansBug/bayes-opt/issues)
[![GitHub pulls](https://img.shields.io/github/issues-pr/HansBug/bayes-opt)](https://github.com/HansBug/bayes-opt/pulls)
[![Contributors](https://img.shields.io/github/contributors/HansBug/bayes-opt)](https://github.com/HansBug/bayes-opt/graphs/contributors)
[![GitHub license](https://img.shields.io/github/license/HansBug/bayes-opt)](https://github.com/HansBug/bayes-opt/blob/master/LICENSE)

An extended implementation of Bayesian Optimization.

This is a forked project based on [fmfn/BayesianOptimization v1.2.0](https://github.com/fmfn/BayesianOptimization). Most of the usage and features from the original repository will be kept for a long time.

## Installation

You can simply install it with `pip` command line from the official PyPI site.

```shell
pip install bayes-opt
```

For more information about installation, you can refer to [Installation](https://hansbug.github.io/bayes-opt/main/tutorials/installation/index.html).


## Documentation

The detailed documentation are hosted on [https://hansbug.github.io/bayes-opt/main/index.html](https://hansbug.github.io/bayes-opt/main/index.html).

Only english version is provided now, the chinese documentation is still under development.


## Quick Start

A painless example

```python
from bayes_opt import BayesianOptimization


def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -x ** 2 - (y - 1) ** 2 + 1


# Bounded region of parameter space
pbounds = {'x': (2, 4), 'y': (-3, 3)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
    verbose=2,
)

optimizer.maximize(
    init_points=10,
    n_iter=25,
)

print(optimizer.max)

```

The output should be

```
|   iter    |  target   |     x     |     y     |
-------------------------------------------------
|  1        | -7.135    |  2.834    |  1.322    |
|  2        | -7.78     |  2.0      | -1.186    |
|  3        | -16.13    |  2.294    | -2.446    |
|  4        | -8.341    |  2.373    | -0.9266   |
|  5        | -7.392    |  2.794    |  0.2329   |
|  6        | -7.069    |  2.838    |  1.111    |
|  7        | -6.412    |  2.409    |  2.269    |
|  8        | -3.223    |  2.055    |  1.023    |
|  9        | -7.455    |  2.835    |  0.3521   |
|  10       | -12.11    |  2.281    | -1.811    |
|  11       | -7.0      |  2.0      |  3.0      |
|  12       | -19.0     |  4.0      |  3.0      |
|  13       | -3.383    |  2.0      |  0.3812   |
|  14       | -3.43     |  2.0      |  1.656    |
|  15       | -3.035    |  2.0      |  0.8129   |
|  16       | -17.03    |  4.0      | -0.4244   |
|  17       | -3.012    |  2.0      |  1.109    |
|  18       | -3.0      |  2.0      |  0.9813   |
|  19       | -3.0      |  2.0      |  0.9911   |
|  20       | -3.0      |  2.0      |  0.994    |
|  21       | -3.0      |  2.0      |  0.9957   |
|  22       | -3.0      |  2.0      |  0.9971   |
|  23       | -3.0      |  2.0      |  0.9994   |
|  24       | -3.0      |  2.0      |  1.004    |
|  25       | -3.0      |  2.0      |  0.978    |
|  26       | -3.001    |  2.0      |  1.024    |
|  27       | -3.001    |  2.0      |  0.9735   |
|  28       | -3.001    |  2.0      |  1.024    |
|  29       | -3.001    |  2.0      |  0.9729   |
|  30       | -3.001    |  2.0      |  1.024    |
|  31       | -3.0      |  2.0      |  1.021    |
|  32       | -3.001    |  2.0      |  0.9709   |
|  33       | -3.001    |  2.0      |  0.9749   |
|  34       | -3.001    |  2.0      |  1.023    |
|  35       | -3.001    |  2.0      |  0.9755   |
=================================================
{'target': -3.00000039014846, 'params': {'x': 2.0, 'y': 0.9993753813483197}}
```

For more tutorial of usages and practices, take a look at [Best Practice](https://hansbug.github.io/bayes-opt/main/best_practice/advanced-tour.html) in documentation.


# Contributing

We appreciate all contributions to improve `bayes-opt`, both logic and system designs. Please refer to CONTRIBUTING.md for more guides.


# License

`bayes-opt` released under the MIT license.

