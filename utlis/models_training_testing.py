# importing
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, RandomFlip, Dropout, \
    BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import gc
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras import backend as K
from tensorflow.keras.applications import MobileNetV2,VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

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

def train_and_evaluate_models(models_list, train_dataset, validation_dataset, epochs, batch_size, final_model,l_r=.001,mome=.9):
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
        model.compile(SGD(learning_rate=l_r, momentum=mome),
                      loss=SparseCategoricalCrossentropy(),
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


def model_2(image_shape, class_len=10):
    base_model = MobileNetV2(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
    base_model.trainable=False

    clf = Sequential([
        InputLayer(input_shape=image_shape),  # 128x128x3 images with 3 color channels
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dense(class_len, activation='softmax')
    ])

    return clf

def model_3(image_shape, class_len=10):
    base_model = VGG16(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
    base_model.trainable=False

    clf = Sequential([
        InputLayer(input_shape=image_shape),  # 128x128x3 images with 3 color channels
        base_model,
        GlobalAveragePooling2D(),
        Dense(class_len, activation='softmax')
    ])

    return clf




def model_1():
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model



