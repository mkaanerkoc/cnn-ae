import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, Input
from keras.datasets import mnist
from keras.losses import CategoricalCrossentropy
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from keras import backend as K
import matplotlib.pyplot as plt

tensorboard_callback = TensorBoard(log_dir="./logs")


def preprocess_data(data):
    return data / 255.0


def add_noise(data, noise_factor=0.4):
    noisy_array = data + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=data.shape
    )
    return np.clip(noisy_array, 0.0, 1.0)


def load_data():
    print('data loaded')
    return mnist.load_data()


class Classifier:
    batch_size = 64
    filter_count = 12
    epochs = 65
    kernel_size = (3, 3)

    def __init__(self, name, input_shape):
        self._name = name
        self._input_shape = input_shape
        self._model = None

    @staticmethod
    def prepare_data():
        train_data, test_data = mnist.load_data()
        train_X, train_y = train_data
        test_X, test_y = test_data
        train_X = Classifier._preprocess_data(train_X)
        train_y = to_categorical(train_y)
        test_X = Classifier._preprocess_data(test_X)
        return train_X, train_y, test_X, test_y

    def construct(self):
        model = Sequential(name=self._name)
        model.add(Conv2D(self.filter_count, self.kernel_size, activation='linear', input_shape=self._input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(self.filter_count, self.kernel_size, activation='linear'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))
        print(model.summary())
        self._model = model
        return model

    def fit(self, X, y):
        self._model.compile(loss=CategoricalCrossentropy(),
                            optimizer='adam',
                            metrics=['accuracy'])

        self._model.fit(X, y,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        callbacks=[tensorboard_callback], )

        self._model.save(f'models/{self._name}')

    def infer(self, input_img):
        results = self._model.predict(input_img)
        return np.argmax(results)

    def load(self):
        self._model = load_model(f"models/{self._name}")
        return self._model

    @staticmethod
    def _preprocess_data(data):
        return data / 255.0


class Denoiser:
    batch_size = 128
    filter_count = 16
    epochs = 25
    kernel_shape = (3, 3)

    def __init__(self, name, input_shape):
        self._name = name
        self._input_shape = input_shape
        self._model = None

    def infer(self, input_img):
        return self._model.predict(input_img)

    @staticmethod
    def prepare_data():
        train_data, test_data = mnist.load_data()
        train_X, train_y = train_data
        train_X = Denoiser._preprocess_data(train_X)
        test_X, test_y = test_data
        return train_X, train_y, test_X, test_y

    def construct(self):
        model = Sequential(name=self._name)
        model.add(Input(shape=self._input_shape))
        model.add(Conv2D(self.filter_count, self.kernel_shape, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(self.filter_count, self.kernel_shape, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2DTranspose(self.filter_count, self.kernel_shape, strides=2, activation='relu', padding='same'))
        model.add(Conv2DTranspose(self.filter_count, self.kernel_shape, strides=2, activation='relu', padding='same'))
        model.add(Conv2D(1, self.kernel_shape, activation='sigmoid', padding='same'))
        self._model = model
        print(model.summary())
        return model

    def fit(self, X, y):
        self._model.compile(optimizer='adam',
                            loss='binary_crossentropy',
                            metrics=['accuracy'])
        K.set_learning_phase(0)
        self._model.fit(x=X,
                        y=y,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        shuffle=True,
                        callbacks=[tensorboard_callback], )
        self._model.save(f"models/{self._name}")

    def load(self):
        self._model = load_model(f"models/{self._name}")
        return self._model

    @staticmethod
    def _preprocess_data(data):
        return data / 255.0


TRAIN_CLASSIFIER = False
TRAIN_DENOISER = False


def main():
    classifier = Classifier('classifier', input_shape=(28, 28, 1))
    denoiser = Denoiser('denoiser', input_shape=(28, 28, 1))
    if TRAIN_CLASSIFIER:
        train_X, train_y, test_X, test_y = classifier.prepare_data()
        classifier.construct()
        classifier.fit(train_X, train_y)
    else:
        classifier.load()

    if TRAIN_DENOISER:
        train_X, train_y, test_X, test_y = denoiser.prepare_data()
        denoiser.construct()
        denoiser.fit(train_X, train_X)
    else:
        denoiser.load()

    # predictions & denoising
    train, test = mnist.load_data()
    test_X, test_y = test

    # preprocess
    test_X = preprocess_data(test_X)

    # add normal noise
    noisy_test_X = add_noise(test_X)
    number_of_samples = 16
    indices = np.random.randint(0, len(test_X), number_of_samples)
    for index, data_ind in enumerate(indices):
        # infer
        digit = classifier.infer(test_X[data_ind].reshape(1, 28, 28, 1))
        noisy_digit = classifier.infer(noisy_test_X[data_ind].reshape(1, 28, 28, 1))
        cleared_digit_img = denoiser.infer(noisy_test_X[data_ind].reshape(1, 28, 28, 1))
        cleared_digit = classifier.infer(cleared_digit_img)
        print(digit, noisy_digit)

        # visualize
        ax = plt.subplot(3, number_of_samples, index + 1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title(digit)
        plt.imshow(test_X[data_ind].reshape(28, 28))
        plt.gray()

        ax = plt.subplot(3, number_of_samples, index + 1 + number_of_samples)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title(noisy_digit)
        plt.imshow(noisy_test_X[data_ind].reshape(28, 28))
        plt.gray()

        ax = plt.subplot(3, number_of_samples, index + 1 + 2 * number_of_samples)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title(cleared_digit)
        plt.imshow(cleared_digit_img.reshape(28, 28))
        plt.gray()

    plt.show()


if __name__ == '__main__':
    main()

