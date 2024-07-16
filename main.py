import os
import random
import tensorflow as tf
tf.keras.backend.clear_session()

import numpy as np
import matplotlib.pyplot as plt
import time

from classification_models.keras import Classifiers
from keras.src.layers import RandomFlip
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.applications import MobileNet
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.optimizers import Adam
from model_type import ModelType


def load_dataset(validation_split=0.2, dec_factor=10):
    # Load the CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    print("shape: ", train_images.shape)

    # Reduce the number of images by a factor of dec_factor
    train_images = train_images[::dec_factor]  # Take every Nth image
    train_labels = train_labels[::dec_factor]  # Corresponding labels
    test_images = test_images[::dec_factor]
    test_labels = test_labels[::dec_factor]

    # Split the training data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=validation_split, random_state=42)

    # Normalize pixel values to be between 0 and 1
    train_images, val_images, test_images = train_images / 255.0, val_images / 255.0, test_images / 255.0

    # Convert labels to one-hot encoding
    train_labels = to_categorical(train_labels, 10)
    val_labels = to_categorical(val_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


def create_model(base_model :ModelType, learning_rate=None,
                 pretrain_model_weights_freezed=True,
                 use_dropout_layer=False,
                 dropout_rate=0.5,
                 use_random_flip=False):
    image_size = 128

    if base_model == ModelType.MOBILE_NET:
        base_model = MobileNet(input_shape=(image_size, image_size, 3),
                               include_top=False,
                               weights='imagenet')
    elif base_model == ModelType.RESNET18:
        ResNet18, preprocess_input = Classifiers.get('resnet18')
        base_model = ResNet18(input_shape=(image_size, image_size, 3),
                         include_top=False,
                         weights='imagenet')
    else:
        raise ValueError("Unknown ModelType")



    base_model.trainable = not pretrain_model_weights_freezed

    model = models.Sequential()

    if use_random_flip:
        model.add(RandomFlip("horizontal_and_vertical", seed=42, input_shape=train_images.shape[1:]))

    model.add(Resizing(image_size, image_size, interpolation="nearest", input_shape=train_images.shape[1:]),)
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())

    if use_dropout_layer:
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(10, activation='softmax'))

    # Specify the learning rate

    # Instantiate the Adam optimizer with the default learning rate
    optimizer = Adam(learning_rate=learning_rate) if learning_rate is not None else Adam()

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def plot_train_vs_val_accuracy(history):
    # Plot training & validation accuracy values
    # =========== FILL IN THIS CODE SECTION

    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(train_accuracy) + 1)

    # Creating the first subplot for accuracy
    plt.figure(figsize=(12, 6))  # Adjust size if needed
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.plot(epochs, train_accuracy, 'g', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.xticks(range(1, len(epochs) + 1))  # Setting ticks for integer epochs
    plt.ylabel('Accuracy')
    plt.legend()

    # Accessing training history for loss plot
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Creating the second subplot for loss
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.plot(epochs, train_loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'y', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.xticks(range(1, len(epochs) + 1))  # Setting ticks for integer epochs
    plt.ylabel('Loss')
    plt.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the combined plot
    plt.show(block=False)

    plt.pause(10)
    plt.close()

    # ===========
    return

def train_and_test(model):
    # Do the actual training
    start_time = time.time()

    history = model.fit(train_images, train_labels, epochs=5, validation_data=(val_images, val_labels))

    end_time = time.time()
    execution_time = end_time - start_time

    print(f'Training time: {execution_time:.1f} seconds')

    # Evaluate
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")

    plot_train_vs_val_accuracy(history)

# Set the random seeds
os.environ['PYTHONHASHSEED'] = str(42)  # This variable influences the hash function's behavior in Python 3.3 and later.
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Load the dataset
train_images, train_labels, val_images, val_labels, test_images, test_labels = load_dataset()

# Create the backbone model that will be used to train
print("********************************\n"
      "************ 1 + 2 *************\n"
      "********************************")

# 2.A
# Do you think that training for more epochs will improve the results? Explain.
# I think that the training for more epochs will improve the results but minorly and not in drastic way

# 2.B
# Do you think that increasing the dataset will improve the results? Explain
# yes think so, because in this way the model will learn better the patterns

model = create_model(ModelType.MOBILE_NET)
train_and_test(model)




print("********************************\n"
      "************ 3.A ***************\n"
      "********************************")
model = create_model(ModelType.MOBILE_NET,
                     pretrain_model_weights_freezed=False)
train_and_test(model)

print("********************************\n"
      "************ 3.B ***************\n"
      "********************************")
model = create_model(ModelType.MOBILE_NET,
                     learning_rate=0.01)
train_and_test(model)

print("********************************\n"
      "************ 3.C ***************\n"
      "********************************")
model = create_model(ModelType.MOBILE_NET,
                     use_dropout_layer=True, dropout_rate=0.5)
train_and_test(model)

print("********************************\n"
      "************ 3.D ***************\n"
      "********************************")
model = create_model(ModelType.MOBILE_NET,
                     use_random_flip=True)
train_and_test(model)


print("********************************\n"
      "************ 3.E ***************\n"
      "********************************")
model = create_model(ModelType.RESNET18,
                     use_random_flip=True)
train_and_test(model)