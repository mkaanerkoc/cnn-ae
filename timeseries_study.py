import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, Conv1DTranspose, Dense, Input, MaxPool1D
import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


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


def construct_signals():
    start, stop, step = 0, 10000, 10
    freq_multiplier = 0.1
    noise_freq_multiplier = 2
    print('yow')
    t = np.arange(start, stop, step)
    X = np.sin((t * freq_multiplier % 360) * np.pi / 180)
    Y = np.sin((t * freq_multiplier % 360 - 30) * np.pi / 180)
    noise = 0.1 * np.sin((t * noise_freq_multiplier % 360) * np.pi / 180)
    signal = X + noise
    smoothed = np.concatenate([np.zeros(9), moving_average(signal, 10)])
    return X, Y, noise, signal, smoothed


def main():
    X, Y, noise, signal, smoothed = construct_signals()
    construct_model('deneme')


def numpy_reshape():
    arr1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(f"Array 1: {arr1}")
    arr2 = arr1.reshape((3, 3))
    print(f"Array 2: {arr2}")
    arr3 = arr2.reshape((9, ))
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
    # main()
    # numpy_reshape()
    keras_input_shape()
