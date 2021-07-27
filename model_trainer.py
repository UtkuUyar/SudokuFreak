"""
TODOS:
1.  Train a model for digit recognition: Input shape = (56, 56, 1), Output shape = (11, )
2.  Preprocess train and test data for sudoku images by using GridDetector.
3.  Reformat the labels. (value -> [probabilty of this square being a digit, c], where c represents probabilities for each digit.)
4.  Use transfer learning methodologies and train model with sudoku image data.
"""
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl
from keras.datasets import mnist

from grid_detector import GridDetector

os.system("color")


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
        test_dev_shuffle = np.arange(0, X_test.shape[0])
        np.random.shuffle(test_dev_shuffle)

        test_dev_shuffle = tf.Variable(test_dev_shuffle, dtype=tf.int64)

        X_test = tf.gather(X_test, test_dev_shuffle)
        y_test = tf.gather(y_test, test_dev_shuffle)

        X_val, y_val = X_test[0:X_test.shape[0] //
                              2], y_test[0:y_test.shape[0] // 2]
        X_test, y_test = X_test[X_test.shape[0] //
                                2:], y_test[y_test.shape[0] // 2:]

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
    try:
        for subset in ["train", "test"]:
            detector.path = f"datasets_rearranged/{subset}/images/"
            for filename in images_map[subset]:
                detector.loadImage(filename)
                preprocessed = detector.preProcessing()

                contours, _ = detector.findContours()
                biggestContour = detector.findBiggestContour(contours)
                grid = detector.birdEyeView(
                    biggestContour["contour"], biggestContour["area"])

                grid_image = Image.fromarray(grid)
                grid_path = f"datasets_rearranged/preprocessed/{subset}/images/{filename}"
                print("\033[A" + "\033[K" + f"Saving {grid_path}...")
                grid_image.save(grid_path)

    except:
        print(filename)
