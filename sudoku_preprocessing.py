"""
## Resources that I used for this script

* Clearing pepper noise: https://stackoverflow.com/questions/41552892/adaptive-threshold-output-unclear

"""
from model_trainer import digitRecognitionModel, progressBar
from PIL import Image
from skimage import io as IO

import cv2
import tensorflow as tf
import numpy as np
import os


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
