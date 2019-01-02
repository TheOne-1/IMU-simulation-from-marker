import numpy as np


class DelayClass:
    def __init__(self, n_sensors):
        self._n_sensors = n_sensors
        self._delays = np.zeros([n_sensors])

    def get_delays(self):
        return self._delays


class RandomDelayClass(DelayClass):
    def __init__(self, n_sensors, sampling_fre, delay_ms):
        self._delay_ms = delay_ms
        delay_sample = 1e-3 * sampling_fre * delay_ms
        super().__init__(n_sensors)
        self._delay_sample = delay_sample
        delays = np.random.rand(n_sensors)   # rand returns a number between 0 and 1
        # minus 0.5 and times 4 to make a -2*delay_sample to 2*delay_sample even distribution
        delays = 4 * delay_sample * (delays - 0.5)
        self._delays = np.round(delays).astype(int)

    def get_delay_ms(self):
        return self._delay_ms


