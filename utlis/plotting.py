import numpy as np
import matplotlib.pyplot as plt
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
    fig, ax = plt.subplots(images_num, images_num, figsize=(images_num * 3, images_num * 3))

    count = 0  # Initialize a counter to keep track of the current image index

    # Loop over the rows of the subplot grid
    for i in range(images_num):
        # Loop over the columns of the subplot grid
        for j in range(images_num):
            if count < len(images_patch):
                # Display the image at the current index in the subplot
                ax[i][j].imshow(images_patch[count])

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