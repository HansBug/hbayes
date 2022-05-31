import pytest

from hbayes import BayesianOptimization
from hbayes import SequentialDomainReductionTransformer


def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -x ** 2 - (y - 1) ** 2 + 1


@pytest.mark.unittest
def test_bound_x_maximize():
    bounds_transformer = SequentialDomainReductionTransformer()
    pbounds = {'x': (-10, 10), 'y': (-10, 10)}
    n_iter = 10

    standard_optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
    )

    standard_optimizer.maximize(
        init_points=2,
        n_iter=n_iter,
    )

    mutated_optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
        bounds_transformer=bounds_transformer
    )

    mutated_optimizer.maximize(
        init_points=2,
        n_iter=n_iter,
    )

    assert len(standard_optimizer.space) == len(mutated_optimizer.space)
    assert not (standard_optimizer._space.bounds == mutated_optimizer._space.bounds).any()
