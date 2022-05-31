from typing import Type, Optional, Dict

import numpy as np

from .target_space import TargetSpace


class TransformerCore:
    """
    Overview:
        Base class of transformer cores.
    """

    # noinspection PyUnusedLocal
    def __init__(self, space: TargetSpace, *args, **kwargs):
        pass

    def transform(self, space: TargetSpace) -> Dict[str, np.ndarray]:
        raise NotImplementedError  # pragma: no cover


class DomainTransformer:
    """
    Overview:
        Base class of transformer.

    .. note::
        State pattern is used, its main service should be wrapped in :class:`TransformerCore` object.
    """

    def __init__(self, core_class: Type[TransformerCore], *args, **kwargs):
        self._core_class = core_class
        self._args = args
        self._kwargs = kwargs
        self._core: Optional[TransformerCore] = None

    def initialize(self, space: TargetSpace):
        self._core = self._core_class(space, *self._args, **self._kwargs)

    def _check_initialization(self):
        if self._core is None:
            raise SyntaxError('Transformer not initialized yet.')  # pragma: no cover

    def transform(self, space: TargetSpace) -> Dict[str, np.ndarray]:
        self._check_initialization()
        return self._core.transform(space)

    def __getattr__(self, item):
        self._check_initialization()  # pragma: no cover
        return getattr(self._core, item)  # pragma: no cover


def _create_bounds(parameters: dict, bounds: np.array) -> Dict[str, np.ndarray]:
    return {param: bounds[i, :] for i, param in enumerate(parameters)}


def _trim(new_bounds: np.array, global_bounds: np.array) -> np.array:
    for i, variable in enumerate(new_bounds):
        if variable[0] < global_bounds[i, 0]:
            variable[0] = global_bounds[i, 0]
        if variable[1] > global_bounds[i, 1]:
            variable[1] = global_bounds[i, 1]

    return new_bounds


class SDRCore(TransformerCore):
    """
    Overview:
        Service core of :class:`SequentialDomainReductionTransformer`.
    """

    def __init__(self, space: TargetSpace, gamma_osc: float = 0.7, gamma_pan: float = 1.0, eta: float = 0.9):
        TransformerCore.__init__(self, space)
        self.gamma_osc = gamma_osc
        self.gamma_pan = gamma_pan
        self.eta = eta

        self.original_bounds = np.copy(space.bounds)
        self.bounds = [self.original_bounds]

        self.previous_optimal = np.mean(space.bounds, axis=1)
        self.current_optimal = np.mean(space.bounds, axis=1)
        self.r = space.bounds[:, 1] - space.bounds[:, 0]

        self.previous_d = 2.0 * (self.current_optimal - self.previous_optimal) / self.r
        self.current_d = 2.0 * (self.current_optimal - self.previous_optimal) / self.r

        self.c = self.current_d * self.previous_d
        self.c_hat = np.sqrt(np.abs(self.c)) * np.sign(self.c)

        self.gamma = 0.5 * (self.gamma_pan * (1.0 + self.c_hat) + self.gamma_osc * (1.0 - self.c_hat))
        self.contraction_rate = self.eta + np.abs(self.current_d) * (self.gamma - self.eta)

        self.r = self.contraction_rate * self.r

    def _update(self, space: TargetSpace):
        # setting the previous
        self.previous_optimal = self.current_optimal
        self.previous_d = self.current_d

        self.current_optimal = space.params[np.argmax(space.target)]
        self.current_d = 2.0 * (self.current_optimal - self.previous_optimal) / self.r

        self.c = self.current_d * self.previous_d
        self.c_hat = np.sqrt(np.abs(self.c)) * np.sign(self.c)

        self.gamma = 0.5 * (self.gamma_pan * (1.0 + self.c_hat) + self.gamma_osc * (1.0 - self.c_hat))
        self.contraction_rate = self.eta + np.abs(self.current_d) * (self.gamma - self.eta)

        self.r = self.contraction_rate * self.r

    def transform(self, space: TargetSpace) -> Dict[str, np.ndarray]:
        self._update(space)

        new_bounds = np.array([
            self.current_optimal - 0.5 * self.r,
            self.current_optimal + 0.5 * self.r
        ]).T

        _trim(new_bounds, self.original_bounds)
        self.bounds.append(new_bounds)
        return _create_bounds(space.keys, new_bounds)


class SequentialDomainReductionTransformer(DomainTransformer):
    """
    Overview:
        A sequential domain reduction transformer based on the work by \
        Stander, N. and Craig, K: "On the robustness of a simple domain reduction scheme  \
        for simulation‐based optimization"

    .. note::
        Its main service is wrapped into :class:`SDRCore`.
    """

    def __init__(self, gamma_osc: float = 0.7, gamma_pan: float = 1.0, eta: float = 0.9):
        DomainTransformer.__init__(self, SDRCore, gamma_osc, gamma_pan, eta)
