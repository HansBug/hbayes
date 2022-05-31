import warnings
from typing import Dict, Union, Tuple

import numpy as np
from hbutils.design import Observable
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from .domain_reduction import DomainTransformer
from .event import OptimizationEvent
from .logger import _get_default_logger
from .target_space import TargetSpace
from .util import UtilityFunction, acq_max, ensure_rng


class Queue:
    def __init__(self):
        self._queue = []

    @property
    def empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._queue)

    def __next__(self):
        if self.empty:
            raise StopIteration("Queue is empty, no more objects to retrieve.")
        obj = self._queue[0]
        self._queue = self._queue[1:]
        return obj

    def next(self):
        return self.__next__()

    def add(self, obj):
        """
        Add object to end of queue.
        """
        self._queue.append(obj)


class BayesianOptimization(Observable):
    """
    Overview:
        This class takes the function to optimize as well as the parameters bounds \
        in order to find which values for the parameters yield the maximum value \
        using bayesian optimization.

    :param f: Function to be maximized.
    :param pbounds: Dictionary with parameters names as keys and a tuple with minimum and maximum values.=
    :param random_state: If the value is an integer, it is used as the seed for creating a \
        numpy.random.RandomState. Otherwise, the random state provided it is used. \
        When set to None, an unseeded random state is generated.
    :param verbose: The level of verbosity.
    :param bounds_transformer: If provided, the transformation is applied to the bounds.
    """

    def __init__(self, f, pbounds, random_state=None, verbose=2, bounds_transformer=None):
        self._random_state = ensure_rng(random_state)

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self._space = TargetSpace(f, pbounds, random_state)
        self._queue = Queue()
        self._last_skipped_params = None

        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

        self._verbose = verbose
        self._bounds_transformer = bounds_transformer
        if self._bounds_transformer is not None:
            if isinstance(self._bounds_transformer, DomainTransformer):
                self._bounds_transformer.initialize(self._space)
            else:
                raise TypeError('The transformer must be an instance of DomainTransformer')

        super(BayesianOptimization, self).__init__(events=OptimizationEvent)

    @property
    def last_skipped(self) -> Dict[str, float]:
        return self._last_skipped_params

    @property
    def space(self) -> TargetSpace:
        return self._space

    @property
    def max(self):
        return self._space.max()

    @property
    def res(self):
        return self._space.res()

    def register(self, x: Union[np.ndarray, Dict[str, float]], y: float):
        """
        Expect observation with known target
        """
        self._space.register(x, y)
        self.dispatch(OptimizationEvent.STEP)

    def probe(self, params: Dict[str, float], lazy=True):
        """
        Evaluates the function on the given points. Useful to guide the optimizer.

        :param params: The parameters where the optimizer will evaluate the function.
        :param lazy:  If True, the optimizer will evaluate the points when calling :func:`maximize`. \
            Otherwise, it will evaluate it at the moment.
        """
        if lazy:
            self._queue.add(params)
        else:
            result = self._space.probe(params)
            if result is not None:
                self.dispatch(OptimizationEvent.STEP)
            else:
                self._last_skipped_params = params
                self.dispatch(OptimizationEvent.SKIP)

    def suggest(self, utility_function: UtilityFunction) -> Dict[str, float]:
        """
        Most promising point to probe next
        """
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        # Scikit-Learn GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)

        # Finding argmax of the acquisition function.
        # noinspection PyArgumentList
        suggestion = acq_max(
            ac=utility_function.utility,
            gp=self._gp,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            random_state=self._random_state
        )
        return self._space.array_to_params(suggestion)

    def _prime_queue(self, init_points: int):
        """
        Make sure there's something in the queue at the very beginning.
        """
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)

        for _ in range(init_points):
            array_ = self._space.random_sample()
            self._queue.add(self._space.array_to_params(array_))

    def _prime_subscriptions(self):
        if not any([subs for subs in self._events.values()]):
            _logger = _get_default_logger(self._verbose)
            for event in OptimizationEvent.__members__.values():
                self.subscribe(event, _logger)

    def maximize(self, init_points: int = 5, n_iter: int = 25,
                 acq='ucb', kappa=2.576, kappa_decay=1, kappa_decay_delay=0,
                 xi=0.0, **gp_params):
        """
        Probes the target space to find the parameters that yield the maximum
        value for the given function.

        :param init_points: Number of iterations before the explorations starts the exploration for the maximum.
        :param n_iter: Number of iterations where the method attempts to find the maximum value.
        :param acq: The acquisition method used. \
            ``ucb`` stands for the Upper Confidence Bounds method. \
            ``ei`` is the Expected Improvement method. \
            ``poi`` is the Probability Of Improvement criterion.
        :param kappa: Parameter to indicate how closed are the next parameters sampled. \
            Higher value = favors spaces that are least explored. \
            Lower value = favors spaces where the regression function is the highest.
        :param kappa_decay: `kappa` is multiplied by this factor every iteration.
        :param kappa_decay_delay: Number of iterations that must have passed before applying the decay to `kappa`.
        :param xi: [unused yet].
        """
        self._prime_subscriptions()
        self.dispatch(OptimizationEvent.START)
        self._prime_queue(init_points)
        self.set_gp_params(**gp_params)

        util = UtilityFunction(acq, kappa, xi, kappa_decay, kappa_decay_delay)
        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
            except StopIteration:
                util.update_params()
                x_probe = self.suggest(util)
                iteration += 1

            self.probe(x_probe, lazy=False)

            if self._bounds_transformer:
                self.set_bounds(self._bounds_transformer.transform(self._space))

        self.dispatch(OptimizationEvent.END)

    def set_bounds(self, new_bounds: Dict[str, Tuple[float, float]]):
        """
        A method that allows changing the lower and upper searching bounds

        :param new_bounds: A dictionary with the parameter name and its new bounds.
        """
        self._space.set_bounds(new_bounds)

    def set_gp_params(self, **params):
        """
        Set parameters to the internal Gaussian Process Regressor
        """
        self._gp.set_params(**params)
