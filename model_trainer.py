"""
## Resources that I used for this script

* Custom Generator Creation: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
* Clearing pepper noise: https://stackoverflow.com/questions/41552892/adaptive-threshold-output-unclear

"""


import sys
import os
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl
from keras.datasets import mnist

from grid_detector import GridDetector


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

os.system("color")


def progressBar(message, filename):
    print("\033[A" + "\033[K" + f"{message}:\t{filename}...")


class digitRecognitionModel():
    def __init__(self, input_shape, save_path, categories=10):
        self.input_shape = input_shape
        self.save_path = save_path
        self.categories = categories

    @staticmethod
    def resize_and_rescale(shape):
        inputs = tf.keras.Input(shape=shape)
        m = tfl.experimental.preprocessing.Resizing(56, 56)(inputs)
        out = tfl.experimental.preprocessing.Rescaling(1. / 255)(m)

        model = tf.keras.Model(inputs=inputs, outputs=out)

        return model

    def divideTestDev(self, X_test, y_test):
        test_dev_shuffle = np.arange(0, X_test.shape[0])
        np.random.shuffle(test_dev_shuffle)

        test_dev_shuffle = tf.Variable(test_dev_shuffle, dtype=tf.int64)

        X_test = tf.gather(X_test, test_dev_shuffle)
        y_test = tf.gather(y_test, test_dev_shuffle)

        X_val, y_val = X_test[0:X_test.shape[0] //
                              2], y_test[0:y_test.shape[0] // 2]
        X_test, y_test = X_test[X_test.shape[0] //
                                2:], y_test[y_test.shape[0] // 2:]

        return (X_val, y_val), (X_test, y_test)

    def createEmptyModel(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        # Block 1
        x = tfl.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
        x = tfl.BatchNormalization()(x)
        x = tfl.Conv2D(32, (3, 3), padding="valid", activation="relu")(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.MaxPool2D(pool_size=(2, 2))(x)

        # Block 2
        x = tfl.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.Conv2D(64, (3, 3), padding="valid", activation="relu")(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.MaxPool2D(pool_size=(2, 2))(x)

        # Final block
        x = tfl.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
        x = tfl.BatchNormalization()(x)

        x = tfl.Flatten()(x)
        x = tfl.Dense(256, activation="relu")(x)
        x = tfl.Dropout(0.3)(x)
        x = tfl.Dense(64, activation="relu")(x)
        x = tfl.Dropout(0.3)(x)

        outputs = tfl.Dense(self.categories, activation='softmax')(x)

        model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name="digit_recognizer")

        return model

    def __preProcessing(self):
        with tf.device('/CPU:0'):
            (X_train, y_train), (X_test, y_test) = mnist.load_data()

            # Refactoring 0 labels for sudoku.
            train_mask = y_train == 0
            test_mask = y_test == 0

            empty_cell = np.zeros((28, 28))

            X_train[train_mask] = empty_cell
            X_test[test_mask] = empty_cell

            # One hot encoding
            y_train = tf.keras.utils.to_categorical(y_train, self.categories)
            y_test = tf.keras.utils.to_categorical(y_test, self.categories)

            rar = digitRecognitionModel.resize_and_rescale((28, 28, 1))

            # Reshaping (from (None, None) to (None, None, 1)) and resizing (expanding by the factor of two) datas
            X_train = tf.reshape(X_train, (X_train.shape[0], 28, 28, 1))
            X_train = rar(X_train)

            X_test = tf.reshape(X_test, (X_test.shape[0], 28, 28, 1))
            X_test = rar(X_test)

            # Shuffling and splitting dev/test data
            (X_val, y_val), (X_test, y_test) = self.divideTestDev(X_test, y_test)

            return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def build(self, from_saved=False, batch_size=128, num_epoch=10):
        self.model = self.createEmptyModel()
        if from_saved:
            self.model.load_weights(self.save_path)
        else:
            (X_train, y_train), (X_val, y_val), (X_test,
                                                 y_test) = self.__preProcessing()

            self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                               loss="categorical_crossentropy", metrics=["accuracy"])

            print("X_train: {}".format(X_train.shape))
            print("y_train: {}".format(y_train.shape))
            print("X_val: {}".format(X_val.shape))
            print("y_val: {}".format(y_val.shape))
            print("X_test: {}".format(X_test.shape))
            print("y_test: {}".format(y_test.shape))

            self.model.summary()

            _ = self.model.fit(X_train, y_train,
                               batch_size=batch_size,
                               epochs=num_epoch,
                               verbose=1,
                               validation_data=(X_val, y_val))

            score = self.model.evaluate(X_test, y_test, verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])

            self.model.save_weights(self.save_path)
            print("Saved model to disk")

        return self.model

    def test(self, X_test, y_test):
        score = self.model.evaluate(X_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])


class SudokuFreak():
    def __init__(self, digit_recognizer, detector, image_shape=(400, 400)):
        self.model = digit_recognizer
        self.detector = detector
        self.image_shape = image_shape

    def extractGrid(self, image_folder, filename, save_folder=None):
        self.detector.path = image_folder
        self.detector.loadImage(filename)
        self.detector.preProcessing()
        contours, _ = self.detector.findContours()
        biggestContour = self.detector.findBiggestContour(contours)
        grid = self.detector.birdEyeView(
            biggestContour["contour"], biggestContour["area"])

        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            grid_image = Image.fromarray(grid)
            grid_path = os.path.join(save_folder, filename)
            self.progressBar("Saving", grid_path)
            print()
            grid_image.save(grid_path)

        self.grid = grid
        return grid

    def encodeLabel(self, label_folder, filename, save_folder=None):
        read_path = os.path.join(label_folder, filename)
        with open(read_path, "r") as label_file:
            matrix = list(map(str.strip, label_file.read().split("\n")))[2:-1]

        matrix = np.array(list(map(str.split, matrix)))
        matrix = matrix.astype(int)

        matrix = tf.keras.utils.to_categorical(matrix.reshape(
            81, ), num_classes=10).reshape((9, 9, 10))

        if save_folder is not None:
            write_path = os.path.join(
                save_folder, filename.split('.')[0] + ".npy")
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            progressBar("Saving", write_path)
            print()
            np.save(write_path, matrix)

        self.label = matrix
        return matrix

    def decodeLabel(self, label):
        if tf.is_tensor(label):
            label = label.numpy()

        new_label = np.zeros((9, 9))
        for i in range(9):
            for j in range(9):
                res = np.where(label[i][j] == 1)
                new_label[i][j] = res[0]
        return new_label

    def predict(self, pad=4):
        image_width, image_height = self.image_shape
        horizontal_step = image_width // 9
        vertical_step = image_height // 9

        digit_width = horizontal_step - 2 * pad
        digit_height = vertical_step - 2 * pad

        segment_shape = (digit_height, digit_width, 1)

        # Get rid of colors
        gray = cv2.cvtColor(self.grid, cv2.COLOR_BGR2GRAY)

        # Blur the image for smoothing
        blurred = cv2.GaussianBlur(gray, (3, 3), 1)

        # ??
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(
            blurred, cv2.MORPH_OPEN, kernel, iterations=1)

        # gaussian with binary inverse thresholding
        threshold = cv2.adaptiveThreshold(
            morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 1, 9, 2)

        # Clearing pepper noise from adaptive thresholding
        preprocessed_grid = cv2.medianBlur(threshold, 5)

        segments = np.zeros((81, ) + segment_shape)
        probs = np.zeros((9, 9))

        rar = digitRecognitionModel.resize_and_rescale(segment_shape)
        for i in range(9):
            for j in range(9):
                segment = preprocessed_grid[i * vertical_step + pad: (
                    i+1) * vertical_step - pad, j * horizontal_step + pad: (j+1) * horizontal_step - pad]

                # cv2.imshow("test", segment)
                # cv2.waitKey()

                segment = segment.reshape(segment.shape + (1, ))
                segments[i * 9 + j, :, :, :] = segment

        segments = rar(segments)
        prediction = self.model.predict(segments)
        probs = np.round(np.amax(prediction, axis=-1).reshape(9, 9), 2)

        prediction = np.argmax(prediction, axis=-1).reshape(9, 9)
        print(prediction)
        print(probs)

        cv2.imshow("median blur", preprocessed_grid)

        cv2.waitKey()


if __name__ == "__main__":
    digit_recognizer = digitRecognitionModel(
        (56, 56, 1), "models/model_digit_recognition.h5")
    model = digit_recognizer.build(from_saved=True)

    SUDOKU_IMAGES = "datasets_rearranged/sudoku_images/images/"
    SUDOKU_LABELS = "datasets_rearranged/sudoku_images/labels/"
    detector = GridDetector(SUDOKU_IMAGES)

    train_images = os.listdir(SUDOKU_IMAGES)
    sudoku_freak = SudokuFreak(model, detector)

    file = np.random.choice(train_images).split(".")[0]
    print(file)

    sudoku_freak.extractGrid(filename="image1006" + ".jpg",
                             image_folder=SUDOKU_IMAGES)
    sudoku_freak.encodeLabel(filename="image1006" + ".dat",
                             label_folder=SUDOKU_LABELS)
    sudoku_freak.predict()
