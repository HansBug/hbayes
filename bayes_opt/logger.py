from __future__ import print_function

import json
import os
from datetime import datetime

from .event import OptimizationEvent


class StepTracker(object):
    def __init__(self):
        self._iterations = 0

        self._previous_max = None
        self._previous_max_params = None

        self._start_time = None
        self._previous_time = None

    def _update_tracker(self, event, instance):
        if event in {OptimizationEvent.STEP, OptimizationEvent.SKIP}:
            self._iterations += 1

            current_max = instance.max
            if current_max and (self._previous_max is None or current_max["target"] > self._previous_max):
                self._previous_max = current_max["target"]
                self._previous_max_params = current_max["params"]

    def _time_metrics(self):
        now = datetime.now()
        if self._start_time is None:
            self._start_time = now
        if self._previous_time is None:
            self._previous_time = now

        time_elapsed = now - self._start_time
        time_delta = now - self._previous_time

        self._previous_time = now
        return (
            now.strftime("%Y-%m-%d %H:%M:%S"),
            time_elapsed.total_seconds(),
            time_delta.total_seconds()
        )


def _get_default_logger(verbose):
    return ScreenLogger(verbose=verbose)


class ScreenLogger(StepTracker):
    _default_cell_size = 9
    _default_precision = 4

    def __init__(self, verbose=2):
        self._verbose = verbose
        self._header_length = None
        super(ScreenLogger, self).__init__()

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, v):
        self._verbose = v

    def _format_number(self, x):
        if isinstance(x, int):
            s = "{x:< {s}}".format(
                x=x,
                s=self._default_cell_size,
            )
        else:
            s = "{x:< {s}.{p}}".format(
                x=x,
                s=self._default_cell_size,
                p=self._default_precision,
            )

        if len(s) > self._default_cell_size:
            if "." in s:
                return s[:self._default_cell_size]
            else:
                return s[:self._default_cell_size - 3] + "..."
        return s

    def _format_key(self, key):
        s = "{key:^{s}}".format(
            key=key,
            s=self._default_cell_size
        )
        if len(s) > self._default_cell_size:
            return s[:self._default_cell_size - 3] + "..."
        return s

    def _row(self, target, keys, params):
        cells = [
            self._format_number(self._iterations + 1),
            self._format_number(target) if isinstance(target, (int, float)) else self._format_key(target),
        ]
        for key in keys:
            cells.append(self._format_number(params[key]))
        return "| " + " | ".join(cells) + " |"

    def _step(self, instance):
        res = instance.res[-1]
        return self._row(res['target'], instance.space.keys, res['params'])

    def _skip(self, instance):
        return self._row('<skipped>', instance.space.keys, instance.last_skipped)

    def _header(self, instance):
        cells = [
            self._format_key("iter"),
            self._format_key("target"),
        ]
        for key in instance.space.keys:
            cells.append(self._format_key(key))

        line = "| " + " | ".join(cells) + " |"
        self._header_length = len(line)
        return line + "\n" + ("-" * self._header_length)

    def _is_new_max(self, instance):
        if instance.max:
            if self._previous_max is None:
                self._previous_max = instance.max["target"]
            return instance.max["target"] > self._previous_max
        else:
            return False

    def update(self, event, instance):
        if event == OptimizationEvent.START:
            line = self._header(instance) + "\n"
        elif event == OptimizationEvent.STEP:
            is_new_max = self._is_new_max(instance)
            if self._verbose == 1 and not is_new_max:
                line = ""
            else:
                line = self._step(instance) + "\n"
        elif event == OptimizationEvent.SKIP:
            line = self._skip(instance) + "\n"
        elif event == OptimizationEvent.END:
            line = "=" * self._header_length + "\n"
        else:
            raise ValueError(f'Unknown event - {event}.')  # pragma: no cover

        if self._verbose:
            print(line, end="")
        self._update_tracker(event, instance)


class JSONLogger(StepTracker):
    def __init__(self, path, reset=True):
        self._path = path if path[-5:] == ".json" else path + ".json"
        if reset:
            try:
                os.remove(self._path)
            except OSError:
                pass
        super(JSONLogger, self).__init__()

    def update(self, event, instance):
        if event == OptimizationEvent.STEP:
            data = dict(instance.res[-1])

            now, time_elapsed, time_delta = self._time_metrics()
            data["datetime"] = {
                "datetime": now,
                "elapsed": time_elapsed,
                "delta": time_delta,
            }

            with open(self._path, "a") as f:
                f.write(json.dumps(data) + "\n")

        self._update_tracker(event, instance)
