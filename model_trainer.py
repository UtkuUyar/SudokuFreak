"""
TODOS:
1.  Train a model for digit recognition: Input shape = (56, 56, 1), Output shape = (11, )
2.  Preprocess train and test data for sudoku images by using GridDetector. and
    Reformat the labels. (value -> [probabilty of this square being a digit, c], where c represents probabilities for each digit.)
3.  Use transfer learning methodologies and train model with sudoku image data.
"""
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import shape
import tensorflow as tf
import tensorflow.keras.layers as tfl
from keras.datasets import mnist

from grid_detector import GridDetector

os.system("color")


def progressBar(message, filename):
    print("\033[A" + "\033[K" + f"{message}:\t{filename}...")


def divideTestDev(X_test, y_test):
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


def reformatLabels(save=True):
    read_path = "datasets_rearranged/{}/labels/"
    write_path = "datasets_rearranged/preprocessed/{}/labels/"
    for subset in ["train", "test"]:
        file_path = read_path.format(subset)
        save_path = write_path.format(subset)
        for label_filename in os.listdir(file_path):
            with open(os.path.join(file_path, label_filename), "r") as label_file:
                matrix = list(
                    map(str.strip, label_file.read().split("\n")))[2:-1]

            matrix = np.array(list(map(str.split, matrix)))
            matrix = matrix.astype(int)

            matrix = tf.keras.utils.to_categorical(matrix.reshape(
                81, ), num_classes=10).reshape((9, 9, 10))
            matrix[:, :, 0] = (matrix[:, :, 0] < 1).astype(int)

            if save:
                path = os.path.join(save_path, label_filename.split('.')[0])
                progressBar("Saving", path + ".npy")
                np.save(path, matrix)

    print()


def extractGrids(detector, images_map):
    for subset in ["train", "test"]:
        detector.path = f"datasets_rearranged/{subset}/images/"
        for filename in images_map[subset]:
            detector.loadImage(filename)
            preprocessed = detector.preProcessing()

            contours, _ = detector.findContours()
            biggestContour = detector.findBiggestContour(contours)
            grid = detector.birdEyeView(
                biggestContour["contour"], biggestContour["area"])

            grid = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)

            grid_image = Image.fromarray(grid)
            grid_path = f"datasets_rearranged/preprocessed/{subset}/images/{filename}"
            progressBar("Saving", grid_path)
            grid_image.save(grid_path)

    print()


class digitRecognitionModel():
    def __init__(self, input_shape, save_path, categories=10):
        self.input_shape = input_shape
        self.save_path = save_path
        self.categories = categories

    def createEmptyModel(self):
        return tf.keras.Sequential(
            [
                tfl.Conv2D(16, (3, 3), input_shape=self.input_shape,
                           activation="relu"),
                tfl.BatchNormalization(),
                tfl.MaxPool2D(pool_size=(2, 2)),
                tfl.Conv2D(32, (3, 3), input_shape=self.input_shape,
                           activation="relu"),
                tfl.BatchNormalization(),
                tfl.MaxPool2D(pool_size=(2, 2)),
                tfl.Conv2D(64, (3, 3), input_shape=self.input_shape,
                           activation="relu"),
                tfl.BatchNormalization(),
                tfl.MaxPool2D(pool_size=(2, 2)),
                tfl.Dropout(0.3),
                tfl.Flatten(),
                tfl.Dense(128, activation="relu"),
                tfl.Dropout(0.3),
                tfl.Dense(self.categories, activation='softmax')
            ]
        )

    def __preProcessing(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # One hot encoding
        y_train = tf.keras.utils.to_categorical(y_train, self.categories)
        y_test = tf.keras.utils.to_categorical(y_test, self.categories)

        # Reshaping (from (None, None) to (None, None, 1)) and resizing (expanding by the factor of two) datas
        X_train = tf.reshape(X_train, (X_train.shape[0], 28, 28, 1))
        X_train = tf.image.resize(X_train, [56, 56])

        X_test = tf.reshape(X_test, (X_test.shape[0], 28, 28, 1))
        X_test = tf.image.resize(X_test, [56, 56])

        # Shuffling and splitting dev/test data
        (X_val, y_val), (X_test, y_test) = divideTestDev(X_test, y_test)

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

            # model.summary()
            # 1.3. Train and test the model.

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


if __name__ == "__main__":
    # Part 1: Train a model for digit recognition. Use MNIST dataset.
    digit_recognizer = digitRecognitionModel(
        (56, 56, 1), "models/model_digit_recognition.h5")
    model = digit_recognizer.build(from_saved=True)

    # Part 2: Preprocessing with GridDetector
    SUDOKU_DATASET_TRAIN = "datasets_rearranged/train/images/"
    SUDOKU_DATASET_TEST = "datasets_rearranged/test/images/"
    detector = GridDetector(SUDOKU_DATASET_TRAIN)

    train_images = os.listdir(SUDOKU_DATASET_TRAIN)
    test_images = os.listdir(SUDOKU_DATASET_TEST)

    images_map = {"train": train_images, "test": test_images}

    # 2.1. Extract sudoku grids from dataset and reformat labels
    extractGrids(detector, images_map)
    reformatLabels(save=True)
    X_train, X_test = [], []
    y_train, y_test = [], []

    # 2.2 Construct matrices
    PREPROCESSED_TRAIN = "datasets_rearranged/preprocessed/{}/{}/"
    PREPROCESSED_TEST = "datasets_rearranged/preprocessed/{}/{}/"
    print("Loading saved data")
    for subset in ["train", "test"]:
        for filename in os.listdir(PREPROCESSED_TRAIN.format(subset, "images")):
            label_filename = os.path.join(
                PREPROCESSED_TRAIN.format(subset, "labels"), filename.split(".")[0] + ".npy")
            image_filename = os.path.join(PREPROCESSED_TRAIN.format(subset, "images"),
                                          filename)

            image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
            label = np.load(label_filename)

            progressBar("Loading", filename + " and " +
                        filename.split(".")[0] + ".npy")
            if subset == "train":
                X_train.append(image)
                y_train.append(label)
            else:
                X_test.append(image)
                y_test.append(label)

    print()

    X_train = tf.convert_to_tensor(X_train)
    X_test = tf.convert_to_tensor(X_test)
    y_train = tf.convert_to_tensor(y_train)
    y_test = tf.convert_to_tensor(y_test)

    # 2.3 Split test/dev sets
    X_train = tf.reshape(X_train, (list(X_train.shape) + [1]))
    X_test = tf.reshape(X_test, (list(X_test.shape) + [1]))

    (X_val, y_val), (X_test, y_test) = divideTestDev(X_test, y_test)

    print("X_train: {}".format(X_train.shape))
    print("X_val: {}".format(X_val.shape))
    print("X_test: {}".format(X_test.shape))
    print("y_train: {}".format(y_train.shape))
    print("y_val: {}".format(y_val.shape))
    print("y_test: {}".format(y_test.shape))

    # 2.4 Construct train, dev, test tensor datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dev_dataset = train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    # 2.5. Apply Data augmentation for datasets.
    augmentation = tf.keras.Sequential([
        tfl.experimental.preprocessing.RandomFlip(
            "horizontal_and_vertical"),
        tfl.experimental.preprocessing.RandomRotation(0.2),
    ])
    batch_size = 16

    aug_train_dataset = train_dataset.map(
        lambda x, y: (augmentation(x, training=True), y))
    aug_train_dataset = aug_train_dataset.batch(batch_size)

    # Part 3: Transfer Learning
