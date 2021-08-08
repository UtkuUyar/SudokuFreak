"""
## Resources that I used for this script

* Custom Generator Creation: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

"""

from digit_synthesis import showRandomDigits
from skimage import io as IO
import tensorflow.keras.layers as tfl
import tensorflow as tf
import numpy as np
import os


DIGITS_DATASET_IMAGES = "datasets_rearranged/digits/images/"
DIGITS_DATASET_LABELS = "datasets_rearranged/digits/labels/"


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

os.system("color")


def progressBar(message, filename):
    print("\033[A" + "\033[K" + f"{message}:\t{filename}...")


class DataGen(tf.keras.utils.Sequence):
    def __init__(self, ID_list, labels, batch_size=128, dimensions=(56, 56), n_channels=1, n_classes=10, shuffle=True):
        super().__init__()
        self.dimensions = dimensions
        self.channels = n_channels
        self.classes = n_classes
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.labels = labels
        self.ID_list = ID_list
        # Setting up indexes
        self.on_epoch_end()

    # Overwritten
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ID_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __batch_generation(self, batch_ID_list):
        X = np.zeros((self.batch_size, ) + self.dimensions +
                     (self.channels, ), dtype=np.uint8)
        y = np.zeros((self.batch_size, ), dtype=int)

        for i, ID in enumerate(batch_ID_list):
            path = os.path.join(DIGITS_DATASET_IMAGES, ID + ".jpg")
            image_array = IO.imread(path, as_gray=True)

            X[i, :, :] = image_array.reshape(
                image_array.shape + (self.channels, ))
            y[i] = self.labels[ID]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.classes)

    # Overwritten
    def __len__(self):
        return len(self.ID_list) // self.batch_size

    # Overwritten
    def __getitem__(self, index):
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]
        batch_ID_list = [self.ID_list[i] for i in indexes]

        X, y = self.__batch_generation(batch_ID_list)

        return X, y


class digitRecognitionModel():
    def __init__(self, input_shape, classes=10):
        self.input_shape = input_shape
        self.classes = classes

    @staticmethod
    def divideTestDev(test_dev_indexes, shuffle=False):
        if shuffle:
            np.random.shuffle(test_dev_indexes)

        test_indexes = test_dev_indexes[:len(test_dev_indexes) // 2]
        validation_indexes = test_dev_indexes[len(test_dev_indexes) // 2:]

        return test_indexes, validation_indexes

    def createEmptyModel(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        # Block 1
        x = tfl.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
        x = tfl.Conv2D(32, (3, 3), padding="valid", activation="relu")(x)
        x = tfl.MaxPool2D(pool_size=(2, 2))(x)

        # Block 2
        x = tfl.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
        x = tfl.Conv2D(64, (3, 3), padding="valid", activation="relu")(x)
        x = tfl.MaxPool2D(pool_size=(2, 2))(x)

        # Final block
        x = tfl.Conv2D(128, (3, 3), padding="same", activation="relu")(x)

        x = tfl.Flatten()(x)
        x = tfl.Dense(256, activation="relu")(x)
        x = tfl.Dropout(0.3)(x)
        x = tfl.Dense(64, activation="relu")(x)
        x = tfl.Dropout(0.3)(x)

        outputs = tfl.Dense(self.classes, activation='softmax')(x)

        model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name="digit_recognizer")

        return model

    def build(self, from_saved=False):
        self.model = self.createEmptyModel()
        if from_saved:
            self.model.load_weights(self.save_path)
        else:
            self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                               loss="categorical_crossentropy", metrics=["accuracy"])

        return self.model


if __name__ == "__main__":

    params = {'dimensions': (56, 56),
              'batch_size': 128,
              'n_classes': 10,
              'n_channels': 1,
              'shuffle': True}

    all_IDs = []
    train_test_IDs = {"train": None, "test": None, "validation": None}
    labels = dict()
    # Read all digit data and shuffle them into training and test sets.
    # Train: 80%, Test: 10%, Validation: 10%
    print("Loading dataset information...")
    for filename in os.listdir(DIGITS_DATASET_IMAGES):
        ID = filename.split('.')[0]
        all_IDs.append(ID)
        labels[ID] = int(np.load(os.path.join(
            DIGITS_DATASET_LABELS, ID + ".npy")).item())

    print("Constructing generators...")
    all_IDs = np.array(all_IDs, dtype=str)
    data_count = len(all_IDs)
    indexes = np.arange(data_count)
    np.random.shuffle(indexes)

    train_indexes = indexes[:int(data_count * 8/10)]
    test_dev_indexes = indexes[int(data_count * 8/10):]

    test_indexes, validation_indexes = digitRecognitionModel.divideTestDev(
        test_dev_indexes)

    train_test_IDs["train"] = all_IDs[train_indexes]
    train_test_IDs["test"] = all_IDs[test_indexes]
    train_test_IDs["validation"] = all_IDs[validation_indexes]

    training_generator = DataGen(train_test_IDs['train'], labels, **params)
    test_generator = DataGen(train_test_IDs['test'], labels, **params)
    val_generator = DataGen(train_test_IDs['validation'], labels, **params)

    print("Visualization")
    for key, value in train_test_IDs.items():
        choices = np.random.choice(value, 100)
        display = np.zeros((100, 56, 56))
        for i, ID in enumerate(choices):
            path = os.path.join(
                DIGITS_DATASET_IMAGES, ID + ".jpg")
            display[i, :, :] = IO.imread(path, as_gray=True)

        showRandomDigits(display)

    print("Constructing the model...")
    digit_recognizer = digitRecognitionModel(
        params["dimensions"] + (params["n_channels"], ))
    model = digit_recognizer.build(from_saved=False)

    model.summary()

    print("Training the model...")
    model.fit(x=training_generator, validation_data=val_generator, epochs=8)

    print("Testing the model...")
    score = model.evaluate(test_generator, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print("Saving the model...")
    model.save("models/model_digit_recognition.h5")

    tf.keras.backend.clear_session()
