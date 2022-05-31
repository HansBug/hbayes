import pytest

from hbayes import OptimizationEvent
from hbayes.logger import StepTracker


@pytest.mark.unittest
def test_tracker():
    class MockInstance:
        def __init__(self, max_target=1, max_params=[1, 1]):
            self._max_target = max_target
            self._max_params = max_params

        @property
        def max(self):
            return {"target": self._max_target, "params": self._max_params}

    tracker = StepTracker()
    assert tracker._iterations == 0
    assert tracker._previous_max is None
    assert tracker._previous_max_params is None

    test_instance = MockInstance()
    tracker._update_tracker("other_event", test_instance)
    assert tracker._iterations == 0
    assert tracker._previous_max is None
    assert tracker._previous_max_params is None

    tracker._update_tracker(OptimizationEvent.STEP, test_instance)
    assert tracker._iterations == 1
    assert tracker._previous_max == 1
    assert tracker._previous_max_params == [1, 1]

    new_instance = MockInstance(max_target=7, max_params=[7, 7])
    tracker._update_tracker(OptimizationEvent.STEP, new_instance)
    assert tracker._iterations == 2
    assert tracker._previous_max == 7
    assert tracker._previous_max_params == [7, 7]

    other_instance = MockInstance(max_target=2, max_params=[2, 2])
    tracker._update_tracker(OptimizationEvent.STEP, other_instance)
    assert tracker._iterations == 3
    assert tracker._previous_max == 7
    assert tracker._previous_max_params == [7, 7]

    tracker._time_metrics()
    start_time = tracker._start_time
    previous_time = tracker._previous_time

    tracker._time_metrics()
    assert start_time == tracker._start_time
    assert previous_time < tracker._previous_time
