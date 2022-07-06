import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv1D, Conv1DTranspose, Dense, Input, MaxPool1D
import matplotlib.pyplot as plt


def moving_average(x, w):
    return np.concatenate([np.zeros(w - 1), np.convolve(x, np.ones(w), 'valid') / w])


def construct_model(name):
    model = Sequential(name=name)
    model.add(Conv1D(filters=8,
                     kernel_size=3,
                     activation='linear',
                     input_shape=(1, 256)))
    model.add(MaxPool1D(pool_size=4))

    model.add(Conv1D(filters=8,
                     kernel_size=3,
                     activation='linear'))
    model.add(MaxPool1D(pool_size=4))
    print(model.summary())


class TimeSeriesData(object):
    def __init__(self, data):
        self._data = data.to_numpy()
        self._chunks = None

    def split(self, input_length, output_length, step_length):
        self._chunks = []
        rows, cols = self._data.shape
        total_length = input_length + output_length
        inputs = []
        outputs = []
        for start in range(0, rows - total_length, step_length):
            input_data = self._data[start:start+input_length]
            output_data = self._data[start+input_length:start+total_length]
            inputs.append(input_data)
            outputs.append(output_data)
        return np.stack(inputs), np.stack(outputs)


def construct_multivariate_timeseries(start, stop, step):
    freq_multiplier = 0.2
    noise_freq_multiplier = 2
    smoothing_filter_window = 20
    t = np.arange(start, stop, step)
    X = np.sin((t * freq_multiplier % 360) * np.pi / 180)
    y = np.sin((t * freq_multiplier % 360 - 30) * np.pi / 180)
    noise = np.sin((t * noise_freq_multiplier % 360) * np.pi / 180)
    noise = np.random.random(len(t))
    trend = np.linspace(1, 10, len(t))
    trend_noise = np.sin((t * 0.02 % 360) * np.pi / 180)
    signal = (0.3 * X) + (0.1 * noise) + (0.4 * trend) + (1 * trend_noise)
    smoothed = moving_average(signal, smoothing_filter_window)
    return pd.DataFrame({'X': X,
                         'y': y,
                         'noise': noise,
                         'signal': signal,
                         'smoothed': smoothed
                         })

def main():
    start, stop, step = 0, 50000, 10
    mv_ts = construct_multivariate_timeseries(start, stop, step)
    signals_figure = plt.figure(1)
    ax = plt.subplot(3, 1, 1)
    plt.plot(mv_ts['signal'])

    ax = plt.subplot(3, 1, 2)
    plt.plot(mv_ts['smoothed'])

    ts = TimeSeriesData(mv_ts)
    inputs, outputs = ts.split(100, 20, 3)

    n = 20
    indices = np.random.randint(0, 100, n)

    samples_figure = plt.figure(2)
    for ind, sample_index in enumerate(indices):
        ax = plt.subplot(1, n, ind+1)
        input_data, output_data = inputs[ind, :, 3], outputs[ind, :, 3]
        output_data_axis = range(len(input_data), len(input_data) + len(output_data))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.plot(input_data)
        plt.plot(output_data_axis, output_data)

    signals_figure.show()
    samples_figure.show()
    plt.show()
    # construct_model('deneme')


def numpy_reshape():
    arr1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(f"Array 1: {arr1}")
    arr2 = arr1.reshape((3, 3))
    print(f"Array 2: {arr2}")
    arr3 = arr2.reshape((9,))
    print(f"Array 3: {arr3}")
    arr4 = arr2.reshape((9, 1))
    print(f"Array 4: {arr4}")
    arr5 = arr4.reshape(-1, 1)
    print(f"Array 5: {arr5}")
    arr6 = arr3.reshape(1, -1)
    print(f"Array 6: {arr6}")


def keras_input_shape():
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling2D, Dense
    i = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu', )(i)
    x = MaxPooling2D((2, 2), )(x)
    model = Model(i, x)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    print(model.summary())


if __name__ == '__main__':
    main()
    # numpy_reshape()
    # keras_input_shape()
