# importing
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, RandomFlip
from tensorflow.keras import regularizers
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import gc
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras import backend as K
from utlis.load_patchs import extract_labels
import numpy as np
def custom_f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)  # Round predictions to nearest integer
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)  # True positives
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)  # False positives
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)  # False negatives

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return K.mean(f1)


def custom_f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)  # Round predictions to nearest integer
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)  # True positives
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)  # False positives
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)  # False negatives

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return K.mean(f1)

def train_and_evaluate_models(models_list, train_dataset, validation_dataset, epochs, batch_size, final_model):
    """
    Trains a list of models and evaluates their validation accuracy.

    Parameters:
    models_list (list): A list of TensorFlow/Keras models to be trained.
    train_dataset (tf.data.Dataset): The dataset to use for training.
    validation_dataset (tf.data.Dataset): The dataset to use for validation.
    epochs (int): The number of epochs to train each model. Default is 10.
    batch_size (int): The batch size to use during training. Default is 32.
    final_model (bool): If True, the function will return the best model instead of clearing it from memory.

    Returns:
    dict: A dictionary with model names as keys and their validation accuracy as values.
    or
    model: The best model trained if final_model is set to True.
    """
    model_performance = {}
    length = len(models_list)
    for i, model in enumerate(models_list):
        print("*" * 90)
        print(f"Training model {i + 1}/{length}...")

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.01),
                      loss=SparseCategoricalCrossentropy(from_logits=True),
                      metrics=["accuracy"])

        # Train the model
        history = model.fit(train_dataset,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=validation_dataset)

        # Store the validation accuracy of the final epoch
        val_accuracy = history.history['val_accuracy'][-1]
        model_performance[f"build_model_{i + 1}"] = val_accuracy
        print(f"Model {i + 1} validation accuracy: {val_accuracy}")
        print("*" * 90)
        if not final_model:
            # Clear the session to free memory
            tf.keras.backend.clear_session()
            # Delete the model objects and force garbage collection
            del model
            gc.collect()
        else:

            # Return the best model if final_model is True
            return model, history

    # Find the best model based on validation accuracy
    best_model_name = max(model_performance, key=model_performance.get)
    best_model_val_accuracy = model_performance[best_model_name]

    print(f"The best model is {best_model_name} with a val accuracy of {best_model_val_accuracy}")

    return model_performance, best_model_name


def predict(prediction_model, test_dataset):
    images, labels = extract_labels(test_dataset)
    predictions = np.argmax(prediction_model.predict(images),axis=1)
    model_acc=accuracy_score(predictions,labels)
    model_f1=f1_score(predictions,labels,average='macro')
    return model_acc, model_f1


def model_1(image_shape, class_len=10):
    garabage_clf = Sequential([
        InputLayer(input_shape=image_shape),  # 128x128x3 images with 3 color channels
        Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="relu"),
        MaxPool2D(pool_size=(2, 2)),
        Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="relu"),
        MaxPool2D(pool_size=(2, 2)),
        Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="relu"),
        MaxPool2D(pool_size=(2, 2)),
        Flatten(),
        Dense(units=128, activation="relu"),
        Dense(units=class_len, activation="linear")
    ])

    return garabage_clf


def model_2(image_shape, class_len=10):
    garabage_clf = Sequential([
        InputLayer(input_shape=image_shape),  # 128x128x3 images with 3 color channels
        Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="relu"),
        MaxPool2D(pool_size=(2, 2)),
        Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="relu"),
        MaxPool2D(pool_size=(2, 2)),
        Flatten(),
        Dense(units=128, activation="relu"),
        Dense(units=class_len, activation="linear")
    ])

    return garabage_clf


# def model_3(image_shape, class_len=10):
#     garabage_clf = Sequential([
#         InputLayer(input_shape=image_shape),  # 128x128x3 images with 3 color channels
#         Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="relu"),
#         MaxPool2D(pool_size=(2, 2)),
#         Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="relu"),
#         MaxPool2D(pool_size=(2, 2)),
#         Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="relu"),
#         MaxPool2D(pool_size=(2, 2)),
#         Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="relu"),
#         MaxPool2D(pool_size=(2, 2)),
#         Flatten(),
#         Dense(units=256, activation="relu"),
#         Dense(units=class_len, activation="linear")
#     ])
#
#     return garabage_clf


# def build_model_4(image_shape, classes_names):
#     garabage_clf = Sequential([
#         InputLayer(input_shape=image_shape),  # 128x128x3 images with 3 color channels
#         Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="relu",
#                kernel_regularizer=regularizers.l1_l2(l1=0.01)),
#         MaxPool2D(pool_size=(2, 2)),
#         Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="relu",
#                kernel_regularizer=regularizers.l1_l2(l1=0.01)),
#         MaxPool2D(pool_size=(2, 2)),
#         Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="relu",
#                kernel_regularizer=regularizers.l1_l2(l1=0.01)),
#         MaxPool2D(pool_size=(2, 2)),
#         Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="relu",
#                kernel_regularizer=regularizers.l1_l2(l1=0.01)),
#         MaxPool2D(pool_size=(2, 2)),
#         Flatten(),
#         Dense(units=32, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)),
#         Dense(units=len(classes_names), activation="linear")
#     ])
#
#     return garabage_clf
#
#
# def build_model_5(image_shape, classes_names):
#     garabage_clf = Sequential([
#         InputLayer(input_shape=image_shape),  # 128x128x3 images with 3 color channels
#         Conv2D(filters=4, kernel_size=(2, 2), strides=(1, 1), padding='valid', activation="relu",
#                kernel_regularizer=regularizers.l1_l2(l1=0.0001)),
#         MaxPool2D(pool_size=(2, 2)),
#         Conv2D(filters=8, kernel_size=(2, 2), strides=(1, 1), padding='valid', activation="relu"),
#         MaxPool2D(pool_size=(2, 2)),
#         Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), padding='valid', activation="relu",
#                kernel_regularizer=regularizers.l1_l2(l1=0.0001)),
#         MaxPool2D(pool_size=(2, 2)),
#         Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), padding='valid', activation="relu"),
#         MaxPool2D(pool_size=(2, 2)),
#         Flatten(),
#         Dense(units=len(classes_names), activation="linear")
#     ])
#
#     return garabage_clf
#
#
# def build_model_6(image_shape, classes_names):
#     garabage_clf = Sequential([
#         InputLayer(input_shape=image_shape),  # 128x128x3 images with 3 color channels
#         Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="relu"),
#         MaxPool2D(pool_size=(2, 2)),
#         Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="relu"),
#         MaxPool2D(pool_size=(2, 2)),
#         Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="relu"),
#         MaxPool2D(pool_size=(2, 2)),
#         Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="relu"),
#         MaxPool2D(pool_size=(2, 2)),
#         Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="relu"),
#         MaxPool2D(pool_size=(2, 2)),
#         Flatten(),
#         Dense(units=512, activation="relu"),
#         Dense(units=len(classes_names), activation="linear")
#     ])
#
#     return garabage_clf
#
#
# def build_model_7(image_shape, classes_names):
#     garabage_clf = Sequential([
#         InputLayer(input_shape=image_shape),  # 128x128x3 images with 3 color channels
#         # RandomFlip("horizontal_and_vertical"),
#         # RandomRotation(0.4),
#         Conv2D(filters=4, kernel_size=(2, 2), strides=(1, 1), padding='same', activation="relu"),
#         MaxPool2D(pool_size=(2, 2)),
#
#         Conv2D(filters=8, kernel_size=(2, 2), strides=(1, 1), padding='same', activation="relu"),
#         MaxPool2D(pool_size=(2, 2)),
#
#         Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), padding='same', activation="relu"),
#         MaxPool2D(pool_size=(2, 2)),
#
#         Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding='same', activation="relu"),
#         MaxPool2D(pool_size=(2, 2)),
#
#         Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), padding='same', activation="relu"),
#         MaxPool2D(pool_size=(2, 2)),
#
#         Flatten(),
#
#         Dense(units=len(classes_names), activation="linear")
#     ])
#
#     return garabage_clf
#
#
# def build_model_8(image_shape, classes_names):
#     garabage_clf = Sequential([
#         InputLayer(input_shape=image_shape),  # 128x128x3 images with 3 color channels
#         RandomFlip("horizontal_and_vertical"),
#         Conv2D(filters=4, kernel_size=(2, 2), strides=(1, 1), padding='same', activation="relu"),
#         MaxPool2D(pool_size=(2, 2)),
#         Conv2D(filters=8, kernel_size=(2, 2), strides=(1, 1), padding='same', activation="relu"),
#         MaxPool2D(pool_size=(2, 2)),
#         Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), padding='same', activation="relu"),
#         MaxPool2D(pool_size=(2, 2)),
#         Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), padding='same', activation="relu"),
#         MaxPool2D(pool_size=(2, 2)),
#         Flatten(),
#
#         Dense(units=len(classes_names), activation="linear")
#     ])
#
#     return garabage_clf
