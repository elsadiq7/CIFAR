
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
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


import numpy as np
import matplotlib.pyplot as plt

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
    epochs = np.arange(1, epoch + 1)

    # Extract the accuracy and loss data from the history object
    train_accuracy = history.history['accuracy']          # Training accuracy
    val_accuracy = history.history.get('val_accuracy')    # Validation accuracy (use .get() in case there's no validation)
    train_loss = history.history['loss']                  # Training loss
    val_loss = history.history.get('val_loss')            # Validation loss (use .get() for safety)

    # Create a figure with two subplots (1 row, 2 columns)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Accuracy Plot
    ax[0].plot(epochs, train_accuracy, label="Train Accuracy", color="b", linestyle="--", linewidth=2, marker='o', alpha=0.7)
    if val_accuracy:
        ax[0].plot(epochs, val_accuracy, label=f"{string} Accuracy", color="g", linestyle=":", linewidth=2, marker='s', alpha=0.7)
    ax[0].set_title("Accuracy")
    ax[0].legend()
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].grid(True)  # Add a grid for better readability
    ax[0].set_ylim([0, 1])

    # Loss Plot
    ax[1].plot(epochs, train_loss, label="Train Loss", color="r", linestyle="--", linewidth=2, marker='o', alpha=0.7)
    if val_loss:
        ax[1].plot(epochs, val_loss, label=f"{string} Loss", color="m", linestyle=":", linewidth=2, marker='s', alpha=0.7)
    ax[1].set_title("Loss")
    ax[1].legend()
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].grid(True)  # Add a grid for better readability
    ax[1].set_ylim([0, max(val_loss)+1])

    # Adjust layout for better spacing between subplots
    plt.tight_layout()

    # Display the plots
    plt.show()


def plot_combined_confusion_heat_map(y_train_act, y_train_pred, y_test_act, y_test_pred, str6=""):
    # Confusion matrix for training data
    clf_matrix_train = confusion_matrix(y_train_act, y_train_pred, normalize='true') * 100
    clf_matrix_str_train = np.round(clf_matrix_train, 1).astype(str)

    # Confusion matrix for testing data
    clf_matrix_test = confusion_matrix(y_test_act, y_test_pred, normalize='true') * 100
    clf_matrix_str_test = np.round(clf_matrix_test, 1).astype(str)

    size = 15
    plt.rcParams.update({'font.size': size, 'font.family': 'Times New Roman'})

    # Create a figure for both heatmaps with a narrower width
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.3})

    # Set a common color bar scale by combining both matrices
    vmin = min(clf_matrix_train.min(), clf_matrix_test.min())
    vmax = max(clf_matrix_train.max(), clf_matrix_test.max())

    # Plot for training heatmap
    ax_train = sns.heatmap(clf_matrix_train, annot=clf_matrix_str_train, annot_kws={'weight': "bold"}, fmt='s',
                           cmap='YlGnBu', vmin=vmin, vmax=vmax, cbar=False, ax=axes[0])
    ax_train.set_xlabel('Predicted Label', size=size + 2, fontweight='bold')
    ax_train.set_ylabel('True Label', size=size + 2, fontweight='bold')
    ax_train.tick_params(axis='both', which='major', labelsize=size + 1)
    ax_train.set_xticklabels(ax_train.get_xticklabels(), fontweight='bold')
    ax_train.set_yticklabels(ax_train.get_yticklabels(), fontweight='bold')
    # Plot for testing heatmap
    ax_test = sns.heatmap(clf_matrix_test, annot=clf_matrix_str_test, annot_kws={'weight': "bold"}, fmt='s',
                          cmap='YlGnBu', vmin=vmin, vmax=vmax, cbar=False, ax=axes[1])
    ax_test.set_xlabel('Predicted Label', size=size + 2, fontweight='bold')
    ax_test.set_ylabel('True Label', size=size + 2, fontweight='bold')
    ax_test.tick_params(axis='both', which='major', labelsize=size + 1)
    # Making tick labels bold
    ax_test.set_xticklabels(ax_test.get_xticklabels(), fontweight='bold')
    ax_test.set_yticklabels(ax_test.get_yticklabels(), fontweight='bold')
    # Add a single color bar for both heatmaps
    cbar = fig.colorbar(ax_train.collections[0], ax=axes, location='right', pad=0.02, shrink=1)
    cbar.ax.tick_params(labelsize=size + 1)
    cbar.set_label('Accuracy(%)', size=size + 1, fontweight='bold')
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontweight='bold')

    # Add subplot labels a and b
    fig.text(0.26, .9, 'train', fontsize=size + 3, fontweight='bold')  # Label for training heatmap
    fig.text(0.63, .9, 'test', fontsize=size + 3, fontweight='bold')  # Label for testing heatmap
    fig.text(0.4, .9, 'Confusion matrix', fontsize=size + 3, fontweight='bold')  # Label for testing heatmap


    # Save the figure with high DPI
    try:
        os.mkdir("images")
    except:
        print("the folder is founded")
    plt.savefig(f"images/combined_{str6}_heat_map.pdf", format='pdf', bbox_inches='tight', dpi=1024)
    print(f"images/combined_{str6}_heat_map.pdf")
    plt.show()