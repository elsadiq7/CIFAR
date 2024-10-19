import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

classes={0: 'airplane',
 1: 'automobile',
 2: 'bird',
 3: 'cat',
 4: 'deer',
 5: 'dog',
 6: 'frog',
 7: 'horse',
 8: 'ship',
 9: 'truck'}




def show_images(images_patch, labels):
    """
    Display a grid of images with their corresponding labels.

    Parameters:
    - images_patch (list or array-like): A collection of image arrays to display.
    - labels (list or array-like): Labels corresponding to each image in `images_patch`.
    """

    # Calculate the number of images per row and column
    images_num = int(np.sqrt(len(images_patch)))

    # Create a grid of subplots with `images_num` rows and `images_num` columns
    fig, ax = plt.subplots(images_num, images_num, figsize=(images_num *5, images_num *5))

    count = 0  # Initialize a counter to keep track of the current image index

    # Loop over the rows of the subplot grid
    for i in range(images_num):
        # Loop over the columns of the subplot grid
        for j in range(images_num):
            if count < len(images_patch):
                # Display the image at the current index in the subplot
                # images_patch[count]=
                ax[i][j].imshow(tf.image.resize(images_patch[count], [64, 64], method=tf.image.ResizeMethod.BILINEAR))

                # Set the title of the subplot to the label of the current image
                ax[i][j].set_title(classes[int(labels[count])], fontweight='bold')

                # Hide the axis for the current subplot
                ax[i][j].set_axis_off()

                count += 1  # Increment the counter for the next image

    # Set a title for the entire grid of images
    fig.suptitle('Grid of Train Images', fontsize=30, fontweight='bold')

    # Adjust layout to prevent overlap and ensure proper spacing
    plt.tight_layout()

    # Display the grid of images
    plt.show()



def plot_accuracy(epoch, history, string="Validation"):
    """
    Plots the accuracy and loss for both training and validation over epochs.

    Parameters:
    - epoch (int): The total number of epochs.
    - history (History object): The history object returned by the model training process (e.g., from Keras).
    - string (str, optional): Custom label for the validation metrics. Defaults to "Validation".

    This function creates two subplots: one for accuracy and one for loss, comparing
    training and validation metrics over the number of epochs.
    """

    # Generate an array of epoch numbers (e.g., [1, 2, ..., epoch])
    epoch = np.arange(1, epoch + 1)

    # Extract the accuracy and loss data from the history object
    train_accuracy = history.history['accuracy']          # Training accuracy
    val_accuracy = history.history['val_accuracy']        # Validation accuracy
    train_loss = history.history['loss']                  # Training loss
    val_loss = history.history['val_loss']                # Validation loss (fixed any typo if present)

    # Create a figure with two subplots (1 row, 2 columns)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Accuracy Plot
    ax[0].plot(epoch, train_accuracy, label="Train Accuracy", color="b", linestyle="--", linewidth=2, marker='o', alpha=0.7)
    ax[0].plot(epoch, val_accuracy, label=f"{string} Accuracy", color="g", linestyle=":", linewidth=2, marker='s', alpha=0.7)
    ax[0].set_title("Accuracy")
    ax[0].legend()
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].grid(True)  # Add a grid for better readability

    # Loss Plot
    ax[1].plot(epoch, train_loss, label="Train Loss", color="r", linestyle="--", linewidth=2, marker='o', alpha=0.7)
    ax[1].plot(epoch, val_loss, label=f"{string} Loss", color="m", linestyle=":", linewidth=2, marker='s', alpha=0.7)
    ax[1].set_title("Loss")
    ax[1].legend()
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].grid(True)  # Add a grid for better readability
    # Adjust layout for better spacing between subplots
    plt.tight_layout()

    # Display the plots
    plt.show()