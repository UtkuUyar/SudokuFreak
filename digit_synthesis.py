import os
import sys
import cv2
import numpy as np
import imutils

from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import rotate

os.system("color")

MAIN_PATH = os.getcwd()
FONT_FOLDERS = os.path.join(MAIN_PATH, "library/fonts/")
SAVE_FOLDER_IMAGE = os.path.join(
    MAIN_PATH, "datasets_rearranged/digits/images/")
SAVE_FOLDER_LABEL = os.path.join(
    MAIN_PATH, "datasets_rearranged/digits/labels/")


def progressBar(message, filename):
    print("\033[A" + "\033[K" + f"{message}:\t{filename}...")


def showRandomDigits(X):
    indices = np.arange(0, X.shape[0])
    indices = np.random.choice(indices, 100)

    random_images = X[indices]

    display = np.concatenate([np.concatenate(
        random_images[i * 10:(i+1) * 10], axis=0) for i in range(10)], axis=1)

    cv2.imshow("Random 100 digits", display)
    cv2.waitKey()


def findFonts():
    paths = []
    for folder in os.listdir(FONT_FOLDERS):
        parent = os.path.join(FONT_FOLDERS, folder)
        for filename in os.listdir(parent):
            if filename.split('.')[1] == "ttf":
                paths.append(os.path.join(parent, filename))

    return paths


def createDigits(start=0, end=9):
    fonts = findFonts()
    total_images = (end - start + 1) * len(fonts)
    X = np.zeros((total_images, 56, 56))
    y = np.zeros((total_images, ))

    index = 0
    for digit in range(start, end+1):
        for font_path in fonts:
            img = Image.new("L", (56, 56))
            font = ImageFont.truetype(font_path, 28)

            if digit != start:
                drawer = ImageDraw.Draw(img)
                digit_width, digit_height = drawer.textsize(str(digit), font)
                drawer.text(((56 - digit_width) / 2,
                             (56 - digit_height) / 2), str(digit), fill=255, font=font)

            array = np.asarray_chkfinite(img, dtype=np.uint8)
            X[index, :, :] = array
            y[index] = digit

            index += 1

    return X, y


class Augmenter:
    @staticmethod
    def flip(images, labels, directions=["ud", "lr"]):
        for d in directions:
            assert d in ["ud", "lr"], "Direction can take values of: ud|lr"

        total_categories = len(directions) + 1
        new_images = np.zeros((total_categories, ) + images.shape)
        new_images[0, :, :, :] = images

        for i, direction in enumerate(directions, start=1):
            if direction == "ud":
                new_images[i, :, :, :] = np.flip(images, axis=1)
            else:
                new_images[i, :, :, :] = np.flip(images, axis=2)

        return np.concatenate(new_images, axis=0), np.concatenate([labels] * total_categories)

    @staticmethod
    def shift(images, labels, amount=10, directions=["left", "right", "up", "down"], overflow="fill-blank"):
        for d in directions:
            assert d in ["left", "right", "up",
                         "down"], "Direction can take values of: left, right, up, down"

        total_categories = len(directions) + 1
        new_images = np.zeros((total_categories, ) + images.shape)
        new_images[0, :, :, :] = images

        width, height = images.shape[1:]

        for i, direction in enumerate(directions, start=1):
            if direction == "left":
                left_slice = images[:, :, :amount]
                right_slice = images[:, :, amount:]

                new_images[i, :, :, :-amount] = right_slice

                if overflow == "fill-blank":
                    new_images[i, :, :, -amount:] = np.zeros((width, amount))
                elif overflow == "roll":
                    new_images[i, :, :, -amount:] = left_slice

            elif direction == "right":
                left_slice = images[:, :, :-amount]
                right_slice = images[:, :, -amount:]

                new_images[i, :, :, amount:] = left_slice

                if overflow == "fill-blank":
                    new_images[i, :, :, :amount] = np.zeros((width, amount))
                elif overflow == "roll":
                    new_images[i, :, :, amount:] = right_slice

            elif direction == "up":
                lower_slice = images[:, amount:, :]
                upper_slice = images[:, :amount, :]

                new_images[i, :, :-amount, :] = lower_slice

                if overflow == "fill-blank":
                    new_images[i, :, -amount:, :] = np.zeros((amount, height))
                elif overflow == "roll":
                    new_images[i, :, -amount:, :] = upper_slice
            else:
                lower_slice = images[:, -amount:, :]
                upper_slice = images[:, :-amount, :]

                new_images[i, :, amount:, :] = upper_slice

                if overflow == "fill-blank":
                    new_images[i, :, :amount, :] = np.zeros((amount, height))
                elif overflow == "roll":
                    new_images[i, :, :amount, :] = lower_slice

        return np.concatenate(new_images, axis=0), np.concatenate([labels] * total_categories)

    @staticmethod
    def rotate(images, labels, angle=20, both_ways=True):

        total_categories = 2 + int(both_ways)
        new_images = np.zeros((total_categories, ) + images.shape)
        new_images[0, :, :, :] = images

        angles = [angle] + (both_ways * [-angle])

        for i, ang in enumerate(angles, start=1):
            rotated_images = rotate(
                images, ang, axes=(2, 1), reshape=False, order=0)
            mask = rotated_images < 0
            rotated_images[mask] = 0
            new_images[i, :, :, :] = rotated_images

        return np.concatenate(new_images, axis=0), np.concatenate([labels] * total_categories)


if __name__ == "__main__":
    X, y = createDigits(start=0, end=9)
    pure_image_count = X.shape[0]
    pure_font_count = int(pure_image_count / 10)

    showRandomDigits(X)
    print()

    print("\033[A" + "\033[K" + "Augmenting... (1/3)")
    images, labels = Augmenter.shift(X, y)
    showRandomDigits(images)

    print("\033[A" + "\033[K" + "Augmenting... (2/3)")
    images, labels = Augmenter.flip(images, labels)
    showRandomDigits(images)

    print("\033[A" + "\033[K" + "Augmenting... (3/3)")
    images, labels = Augmenter.rotate(images, labels)
    showRandomDigits(images)

    print()

    digit_str_map = {0: "empty",
                     1: "one",
                     2: "two",
                     3: "three",
                     4: "four",
                     5: "five",
                     6: "six",
                     7: "seven",
                     8: "eight",
                     9: "nine"}

    for index in range(images.shape[0]):
        image_array = images[index, :, :]
        label = labels[index]
        # digit_font_batch

        digit = digit_str_map[label]
        filename_base = "{}_{}".format(digit, index)

        image_filename = os.path.join(
            SAVE_FOLDER_IMAGE, filename_base + ".jpg")
        label_filename = os.path.join(
            SAVE_FOLDER_LABEL, filename_base + ".npy")

        cv2.imwrite(image_filename, image_array)

        np.save(label_filename, label)

        progressBar("Saved", filename_base)

    print()
