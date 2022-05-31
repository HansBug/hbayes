from typing import List, Dict, Union, Tuple, Optional

import numpy as np

from .util import ensure_rng


def _hashable(x: np.ndarray) -> Tuple[float, ...]:
    """
    Ensure that a point is hashable by a python dict.
    """
    return tuple(map(float, x))


class FuncFailed(Exception):
    """
    Overview:
        Exception class thrown then ``target_func`` is failed. \
        The result (including x and y) will be skipped and ignored after that.
    """
    pass


class TargetSpace(object):
    """
    Overview:
        Holds the param-space coordinates (X) and target values (Y) \
        Allows for constant-time appends while ensuring no duplicates are added.

    Examples:
        >>> from bayes_opt.target_space import TargetSpace
        >>>
        >>> def target_func(p1, p2):
        ...     return p1 + p2
        >>>
        >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        >>> space = TargetSpace(target_func, pbounds, random_state=0)
        >>> x = space.random_sample()
        >>> y = space.probe(x)
        >>> print(x, y)
        [ 0.5488135  71.80374727] 72.35256077479684
        >>> print(space.max())
        {'target': 72.35256077479684, 'params': {'p1': 0.5488135039273248, 'p2': 71.80374727086952}}
    """

    def __init__(self, target_func, pbounds, random_state=None):
        """
        Constructor of :class:`TargetSpace`.

        :param target_func: Function to be maximized.
        :param pbounds: Dictionary with parameters names as keys and a tuple with minimum and maximum values.
        :param random_state: Optionally specify a seed for a random number generator
        """
        self.random_state = ensure_rng(random_state)

        # The function to be optimized
        self.target_func = target_func

        # Get the name of the parameters
        self._keys: List[str] = sorted(pbounds)
        # Create an array with parameters bounds
        self._bounds = np.array([pbounds[k] for k in self._keys], dtype=np.float64)

        # pre-allocated memory for X and Y points
        self._params = np.empty(shape=(0, self.dim))
        self._target = np.empty(shape=0)

        # keep track of unique points we have seen so far
        self._cache: Dict[Tuple[float, ...], float] = {}

    def __contains__(self, x) -> bool:
        return _hashable(x) in self._cache

    def __len__(self) -> int:
        return len(self._target)

    @property
    def empty(self) -> bool:
        return len(self) == 0

    @property
    def params(self) -> np.ndarray:
        return self._params

    @property
    def target(self) -> np.ndarray:
        return self._target

    @property
    def dim(self) -> int:
        return len(self._keys)

    @property
    def keys(self) -> List[str]:
        return self._keys

    @property
    def bounds(self) -> np.ndarray:
        return self._bounds

    def params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        try:
            assert set(params) == set(self.keys)
        except AssertionError:
            raise ValueError(
                "Parameters' keys ({}) do ".format(sorted(params)) +
                "not match the expected set of keys ({}).".format(self.keys)
            )
        return np.asarray([params[key] for key in self.keys])

    def array_to_params(self, x: np.ndarray) -> Dict[str, float]:
        try:
            assert len(x) == len(self.keys)
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
            )
        return dict(zip(self.keys, x))

    def _as_array(self, x: Union[np.ndarray, Dict[str, float]]) -> np.ndarray:
        if isinstance(x, dict):
            x = self.params_to_array(x)
        else:
            x = np.asarray(x, dtype=float)

        x = x.ravel()
        if x.size == self.dim:
            return x
        else:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
            )

    def register(self, x, y):
        # noinspection PyShadowingNames
        """
        Append a point and its target value to the known data.

        :param x: A single point, with len(x) == self.dim.
        :param y: Target function value.
        :raises KeyError: If the point is not unique.

        .. note::
            Runs in ammortized constant time.

        Examples:
            >>> import numpy as np
            >>> from bayes_opt.target_space import TargetSpace
            >>>
            >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
            >>> space = TargetSpace(lambda p1, p2: p1 + p2, pbounds)
            >>> len(space)
            0
            >>> x_ = np.array([0, 0])
            >>> y = 1
            >>> space.register(x_, y)
            [[0. 0.]] [1.]
            >>> len(space)
            1
        """
        x = self._as_array(x)
        if x in self:
            raise KeyError('Data point {} is not unique'.format(x))

        # Insert data into unique dictionary
        self._cache[_hashable(x.ravel())] = y

        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._target = np.concatenate([self._target, [y]])

    def probe(self, x: Union[np.ndarray, Dict[str, float]]) -> Optional[float]:
        """
        Evaluates a single point x, to obtain the value y and then records them as observations.

        .. note::
            * If ``x`` has been previously seen returns a cached value of y.
            * If :class:`FuncFailed` is raised from ``target_func``, this pair of x and y will be ignored, with \
                return value of ``None``.

        :param x: A single point, with ``len(params) == self.dim``.
        :returns: Target function value.
        """
        x = self._as_array(x)

        try:
            return self._cache[_hashable(x)]
        except KeyError:
            x = dict(zip(self._keys, x))
            try:
                target = self.target_func(**x)
            except FuncFailed:
                return None
            else:
                self.register(x, target)
                return target

    def random_sample(self) -> np.ndarray:
        """
        Creates random points within the bounds of the space.

        :returns: [dim] array points with dimensions corresponding to `self._keys`

        Examples:
            >>> from bayes_opt.target_space import TargetSpace
            >>>
            >>> target_func = lambda p1, p2: p1 + p2
            >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
            >>> space = TargetSpace(target_func, pbounds, random_state=0)
            >>> space.random_sample()
            array([ 0.5488135 , 71.80374727])
        """
        # TODO: support integer, category, and basic scipy.optimize constraints
        data = np.empty((1, self.dim))
        for col, (lower, upper) in enumerate(self._bounds):
            data.T[col] = self.random_state.uniform(lower, upper, size=1)
        return data.ravel()

    def max(self):
        """
        Get maximum target value found and corresponding parameters.
        """
        try:
            # noinspection PyArgumentList
            return {
                'target': self.target.max(),
                'params': dict(zip(self.keys, self.params[self.target.argmax()])),
            }
        except ValueError:
            return None

    def res(self):
        """
        Get all target values found and corresponding parameters.
        """
        params = [dict(zip(self.keys, p)) for p in self.params]
        return [
            {"target": target, "params": param}
            for target, param in zip(self.target, params)
        ]

    def set_bounds(self, new_bounds: Dict[str, Tuple[float, float]]):
        """
        A method that allows changing the lower and upper searching bounds

        :param new_bounds: A dictionary with the parameter name and its new bounds
        """
        for row, key in enumerate(self.keys):
            if key in new_bounds:
                self._bounds[row] = new_bounds[key]
