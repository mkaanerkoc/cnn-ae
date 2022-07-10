import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv1D, Conv1DTranspose, Dense, Input, MaxPool1D
import matplotlib.pyplot as plt
from tcnae import tcn_ae
from stockstats import wrap, unwrap
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


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


class FinancialTimeSeriesData(object):
    def __init__(self, symbol):
        self._price_ohlc = wrap(pd.read_csv(f'data/{symbol}_30_prices.csv'))
        self._data = pd.DataFrame()
        # self._data['close'] = self._price_ohlc['close']
        self._data['close_4_sma'] = self._price_ohlc['close_4_sma']
        self._data['close_16_ema'] = self._price_ohlc['close_16_ema']
        self._data['close_32_ema'] = self._price_ohlc['close_32_ema']
        self._data['close_64_ema'] = self._price_ohlc['close_64_ema']

    def split(self, input_length, output_length, step_length):
        inputs = []  # input = output in Autoencoders
        rows, cols = self._data.shape
        total_length = input_length + output_length
        for start in range(0, rows - total_length, step_length):
            input_data = self._data[start:start + input_length]
            inputs.append(input_data)
        return np.stack(inputs), np.stack(inputs)


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
    signal = (0.3 * X) + (0.1 * noise) + (0.001 * trend) + (1 * trend_noise)
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
        ax = plt.subplot(2, n, ind+1)
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


def tcn_study():
    TRAIN_TCN = False
    scaler = MinMaxScaler()
    start, stop, step = 0, 50000, 10
    # mv_ts = construct_multivariate_timeseries(start, stop, step)
    eth_ts = FinancialTimeSeriesData('ETHUSDT')
    btc_ts = FinancialTimeSeriesData('BTCUSDT')

    train_X, train_y = eth_ts.split(input_length=84, output_length=0, step_length=8)
    test_X, test_y = btc_ts.split(input_length=84, output_length=0, step_length=8)
    sample, window, dim = train_X.shape
    print(sample, window, dim)
    for i in range(sample):
        train_X[i, :, :] = scaler.fit_transform(train_X[i, :, :])
        test_X[i, :, :] = scaler.fit_transform(test_X[i, :, :])

    n = 24
    indices = np.random.randint(0, 100, n)

    tcn_ae_model = tcn_ae.TCNAE(ts_dimension=dim,
                                nb_filters=32,
                                kernel_size=3,
                                name='tcn_model')  # Use the parameters specified in the paper

    #
    # Train TCN-AE for 10 epochs. For a better accuracy
    # on the test case, increase the epochs to epochs=40
    # The training takes about 3-4 minutes for 10 epochs,
    # and 15 minutes for 40 epochs (on Google CoLab, with GPU enabled)
    #
    if TRAIN_TCN:
        tcn_ae_model.fit(train_X, train_X,
                         batch_size=32,
                         epochs=50,
                         verbose=1,
                         save=True)
    else:
        tcn_ae_model.load()

    samples_figure = plt.figure(2)
    for ind, sample_index in enumerate(indices):
        input_data, output_data = train_X[ind, :, :], train_y[ind, :, :]
        #output_data_axis = range(len(input_data), len(input_data) + len(output_data))
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # plt.plot(input_data)
        #plt.plot(output_data_axis, output_data)

    n = 6
    indices = np.random.randint(0, 100, n)
    predictions = tcn_ae_model.predict_test(test_X[indices, :, :])

    for ind, sample_index in enumerate(indices):
        ax = plt.subplot(2, n, ind + 1)
        plt.plot(test_X[sample_index, :, :])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, ind + 1 + n)
        plt.plot(predictions[ind, :, :])

    samples_figure.show()
    plt.show()


if __name__ == '__main__':
    tcn_study()
