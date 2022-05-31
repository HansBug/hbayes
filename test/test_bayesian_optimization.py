import random

import numpy as np
import pytest

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from bayes_opt.event import OptimizationEvent
from bayes_opt.logger import ScreenLogger
from bayes_opt.target_space import FuncFailed


def target_func(**kwargs):
    # arbitrary target func
    return sum(kwargs.values())


PBOUNDS = {'p1': (0, 10), 'p2': (0, 10)}


@pytest.mark.unittest
def test_register():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    assert len(optimizer.space) == 0

    optimizer.register(x={"p1": 1, "p2": 2}, y=3)
    assert len(optimizer.res) == 1
    assert len(optimizer.space) == 1

    optimizer.space.register(x={"p1": 5, "p2": 4}, y=9)
    assert len(optimizer.res) == 2
    assert len(optimizer.space) == 2

    with pytest.raises(KeyError):
        optimizer.register(x={"p1": 1, "p2": 2}, y=3)
    with pytest.raises(KeyError):
        optimizer.register(x={"p1": 5, "p2": 4}, y=9)


@pytest.mark.unittest
def test_probe_lazy():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)

    optimizer.probe(params={"p1": 1, "p2": 2}, lazy=True)
    assert len(optimizer.space) == 0
    assert len(optimizer._queue) == 1

    optimizer.probe(params={"p1": 6, "p2": 2}, lazy=True)
    assert len(optimizer.space) == 0
    assert len(optimizer._queue) == 2

    optimizer.probe(params={"p1": 6, "p2": 2}, lazy=True)
    assert len(optimizer.space) == 0
    assert len(optimizer._queue) == 3


@pytest.mark.unittest
def test_probe_eager():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)

    optimizer.probe(params={"p1": 1, "p2": 2}, lazy=False)
    assert len(optimizer.space) == 1
    assert len(optimizer._queue) == 0
    assert optimizer.max["target"] == 3
    assert optimizer.max["params"] == {"p1": 1, "p2": 2}

    optimizer.probe(params={"p1": 3, "p2": 3}, lazy=False)
    assert len(optimizer.space) == 2
    assert len(optimizer._queue) == 0
    assert optimizer.max["target"] == 6
    assert optimizer.max["params"] == {"p1": 3, "p2": 3}

    optimizer.probe(params={"p1": 3, "p2": 3}, lazy=False)
    assert len(optimizer.space) == 2
    assert len(optimizer._queue) == 0
    assert optimizer.max["target"] == 6
    assert optimizer.max["params"] == {"p1": 3, "p2": 3}


@pytest.mark.unittest
def test_suggest_at_random():
    util = UtilityFunction(kind="poi", kappa=5, xi=0)
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)

    for _ in range(50):
        sample = optimizer.space.params_to_array(optimizer.suggest(util))
        assert len(sample) == optimizer.space.dim
        assert all(sample >= optimizer.space.bounds[:, 0])
        assert all(sample <= optimizer.space.bounds[:, 1])


@pytest.mark.unittest
def test_suggest_with_one_observation():
    util = UtilityFunction(kind="ucb", kappa=5, xi=0)
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)

    optimizer.register(x={"p1": 1, "p2": 2}, y=3)

    for _ in range(5):
        sample = optimizer.space.params_to_array(optimizer.suggest(util))
        assert len(sample) == optimizer.space.dim
        assert all(sample >= optimizer.space.bounds[:, 0])
        assert all(sample <= optimizer.space.bounds[:, 1])

    # suggestion = optimizer.suggest(util)
    # for _ in range(5):
    #     new_suggestion = optimizer.suggest(util)
    #     assert suggestion == new_suggestion


@pytest.mark.unittest
def test_prime_queue_all_empty():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    assert len(optimizer._queue) == 0
    assert len(optimizer.space) == 0

    optimizer._prime_queue(init_points=0)
    assert len(optimizer._queue) == 1
    assert len(optimizer.space) == 0


@pytest.mark.unittest
def test_prime_queue_empty_with_init():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    assert len(optimizer._queue) == 0
    assert len(optimizer.space) == 0

    optimizer._prime_queue(init_points=5)
    assert len(optimizer._queue) == 5
    assert len(optimizer.space) == 0


@pytest.mark.unittest
def test_prime_queue_with_register():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    assert len(optimizer._queue) == 0
    assert len(optimizer.space) == 0

    optimizer.register(x={"p1": 1, "p2": 2}, y=3)
    optimizer._prime_queue(init_points=0)
    assert len(optimizer._queue) == 0
    assert len(optimizer.space) == 1


@pytest.mark.unittest
def test_prime_queue_with_register_and_init():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    assert len(optimizer._queue) == 0
    assert len(optimizer.space) == 0

    optimizer.register(x={"p1": 1, "p2": 2}, y=3)
    optimizer._prime_queue(init_points=3)
    assert len(optimizer._queue) == 3
    assert len(optimizer.space) == 1


@pytest.mark.unittest
def test_prime_subscriptions():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    optimizer._prime_subscriptions()

    # Test that the default observer is correctly subscribed
    for event in OptimizationEvent.__members__.values():
        assert all([
            isinstance(k, ScreenLogger) and hasattr(k, 'update')
            for k, callback in optimizer.subscriptions(event)
        ])

    test_subscriber = "test_subscriber"

    def test_callback():
        pass

    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    optimizer.subscribe(
        event=OptimizationEvent.START,
        subscriber=test_subscriber,
        callback=test_callback,
    )
    # Test that the desired observer is subscribed
    assert all([
        k == test_subscriber and v == test_callback
        for k, v in optimizer.subscriptions(OptimizationEvent.START)
    ])

    # Check that prime subscriptions won't overight manual subscriptions
    optimizer._prime_subscriptions()
    assert all([
        k == test_subscriber and v == test_callback
        for k, v in optimizer.subscriptions(OptimizationEvent.START)
    ])

    assert optimizer.subscriptions(OptimizationEvent.STEP) == []
    assert optimizer.subscriptions(OptimizationEvent.END) == []

    with pytest.raises(KeyError):
        optimizer.subscriptions("other")


@pytest.mark.unittest
def test_set_bounds():
    pbounds = {
        'p1': (0, 1),
        'p3': (0, 3),
        'p2': (0, 2),
        'p4': (0, 4),
    }
    optimizer = BayesianOptimization(target_func, pbounds, random_state=1)

    # Ignore unknown keys
    optimizer.set_bounds({"other": (7, 8)})
    assert all(optimizer.space.bounds[:, 0] == np.array([0, 0, 0, 0]))
    assert all(optimizer.space.bounds[:, 1] == np.array([1, 2, 3, 4]))

    # Update bounds accordingly
    optimizer.set_bounds({"p2": (1, 8)})
    assert all(optimizer.space.bounds[:, 0] == np.array([0, 1, 0, 0]))
    assert all(optimizer.space.bounds[:, 1] == np.array([1, 8, 3, 4]))


@pytest.mark.unittest
def test_set_gp_params():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    assert optimizer._gp.alpha == 1e-6
    assert optimizer._gp.n_restarts_optimizer == 5

    optimizer.set_gp_params(alpha=1e-2)
    assert optimizer._gp.alpha == 1e-2
    assert optimizer._gp.n_restarts_optimizer == 5

    optimizer.set_gp_params(n_restarts_optimizer=7)
    assert optimizer._gp.alpha == 1e-2
    assert optimizer._gp.n_restarts_optimizer == 7


@pytest.mark.unittest
def test_maximize():
    class Tracker:
        def __init__(self):
            self.start_count = 0
            self.step_count = 0
            self.skip_count = 0
            self.end_count = 0

        def update_start(self):
            self.start_count += 1

        def update_step(self):
            self.step_count += 1

        def update_skip(self):
            self.skip_count += 1

        def update_end(self):
            self.end_count += 1

        def reset(self):
            self.__init__()

    optimizer = BayesianOptimization(target_func, PBOUNDS,
                                     random_state=np.random.RandomState(1))

    tracker = Tracker()
    optimizer.subscribe(
        event=OptimizationEvent.START,
        subscriber=tracker,
        callback=tracker.update_start,
    )
    optimizer.subscribe(
        event=OptimizationEvent.STEP,
        subscriber=tracker,
        callback=tracker.update_step,
    )
    optimizer.subscribe(
        event=OptimizationEvent.SKIP,
        subscriber=tracker,
        callback='update_skip',
    )
    optimizer.subscribe(
        event=OptimizationEvent.END,
        subscriber=tracker,
        callback=tracker.update_end,
    )

    optimizer.maximize(init_points=0, n_iter=0)
    assert optimizer._queue.empty
    assert len(optimizer.space) == 1
    assert tracker.start_count == 1
    assert tracker.step_count == 1
    assert tracker.skip_count == 0
    assert tracker.end_count == 1

    optimizer.maximize(init_points=2, n_iter=0, alpha=1e-2)
    assert optimizer._queue.empty
    assert len(optimizer.space) == 3
    assert optimizer._gp.alpha == 1e-2
    assert tracker.start_count == 2
    assert tracker.step_count == 3
    assert tracker.skip_count == 0
    assert tracker.end_count == 2

    optimizer.maximize(init_points=0, n_iter=2)
    assert optimizer._queue.empty
    assert len(optimizer.space) == 5
    assert tracker.start_count == 3
    assert tracker.step_count == 5
    assert tracker.skip_count == 0
    assert tracker.end_count == 3


@pytest.mark.unittest
def test_define_wrong_transformer():
    with pytest.raises(TypeError):
        _ = BayesianOptimization(
            target_func, PBOUNDS,
            random_state=np.random.RandomState(1),
            bounds_transformer=3
        )


@pytest.mark.unittest
@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_actual_use():
    _is_first = True

    def black_box_function(x, y):
        nonlocal _is_first
        _first, _is_first = _is_first, False
        if _first or random.random() < 0.1:
            raise FuncFailed(x, y)
        return -x ** 2 - (y - 1) ** 2 + 1

    pbounds = {'x': (2, 4), 'y': (-3, 3)}
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=1,
        verbose=2,
    )
    optimizer.maximize(
        init_points=15,
        n_iter=30,
    )

    target = optimizer.max['target']
    params = optimizer.max['params']
    assert target == pytest.approx(-3.0, abs=1e-3)
    assert params['x'] == pytest.approx(2.0, abs=1e-4)
    assert params['y'] == pytest.approx(1.0, abs=1e-2)


if __name__ == '__main__':
    r"""
    CommandLine:
        python test/test_bayesian_optimization.py
    """
    pytest.main([__file__])
