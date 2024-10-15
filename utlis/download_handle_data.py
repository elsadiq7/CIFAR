import requests
import tarfile
import pickle
import numpy as np
import imageio
import os


def download_data(url):
    """
    Downloads a .tar.gz file from the specified URL and extracts its contents.

    Parameters:
    - url (str): The URL of the .tar.gz file to download.

    """
    # Local filename to save the downloaded .tar.gz file
    local_filename = "data/file.tar.gz"

    # Create a directory named 'data' if it doesn't exist
    try:
        os.mkdir('data')
    except FileExistsError:
        print('Directory already exists')

    # Check if the file already exists to avoid unnecessary downloads
    if os.path.exists(local_filename):
        print('File already exists')
    else:
        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Open a file in binary write mode to save the .tar.gz file
            with open(local_filename, 'wb') as file:
                file.write(response.content)
            print("File downloaded successfully!")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")


def extract_cifar10_data(directory):
    """
    Extracts the contents of the CIFAR-10 .tar.gz file into the specified directory.

    Parameters:
    - directory (str): The directory where the contents will be extracted.
    """
    local_filename = "data/file.tar.gz"  # The name of the .tar.gz file to be extracted

    # Check if the .tar.gz file exists before trying to extract it
    if not os.path.exists(local_filename):
        print(f"The file '{local_filename}' does not exist. Please download it first.")
        return

    # Ensure the extraction directory exists
    os.makedirs(directory, exist_ok=True)

    # Extract the contents of the .tar.gz file
    with tarfile.open(local_filename, 'r:gz') as tar:
        tar.extractall(path=directory)  # Extract to the specified directory
    print(f"Contents extracted to: {directory}")


class ConvertDataToImages:
    """
    A class to convert CIFAR-10 dataset batches from binary format into images
    and save them in a structured directory format.

    Attributes:
        labels_dict (dict): A dictionary mapping label indices to label names.
        batch_data (list): A list to hold image data from a batch.
        batch_labels (list): A list to hold the corresponding labels for the images.
        batch_filenames (list): A list to hold the filenames associated with each image.
        labels_names (list): A list to hold the names of labels from metadata.
    """

    def __init__(self):
        """Initializes the ConvertDataToImages class."""
        self.labels_dict = {}  # Dictionary for label mapping
        self.batch_data = []  # List for storing batch image data
        self.batch_labels = []  # List for storing labels corresponding to batch data
        self.batch_filenames = []  # List for storing filenames for each image
        self.labels_names = []  # List for storing label names

    def load_label_names(self, meta_file_path):
        """
        Load metadata for CIFAR-10 dataset.

        Parameters:
            meta_file_path (str): The file path to the metadata file.
        """
        with open(meta_file_path, 'rb') as f:
            # Load metadata containing label names
            meta = pickle.load(f, encoding='latin1')
        self.labels_names = meta['label_names']

    def create_images_folder(self):
        """
        Create a directory structure for storing images based on their labels.

        Creates a main 'images' directory and subdirectories for each label.
        If the directories already exist, it catches the exception and prints a message.
        """
        try:
            os.mkdir('data/images')  # Create main images directory
            # Create a subdirectory for each label
            for filename in self.labels_names:
                os.mkdir(f'data/images/{filename}')
        except FileExistsError:
            print('Files already exist. Skipping directory creation.')

    def create_image_labels_to_string(self):
        """
        Create a mapping from label indices to label names.

        This method populates the labels_dict with label indices as keys
        and corresponding label names as values.
        """
        count = 0
        for filename in self.labels_names:
            self.labels_dict[count] = filename  # Map index to label name
            count += 1

    def load_batch(self, file_path):
        """
        Load a batch of CIFAR-10 data.

        Parameters:
            file_path (str): The file path to the CIFAR-10 batch file.
        """
        with open(file_path, 'rb') as f:
            # Load the batch and extract data, labels, and filenames
            batch = pickle.load(f, encoding='latin1')
        self.batch_data = batch['data']
        self.batch_labels = batch['labels']
        self.batch_filenames = batch['filenames']

    def convert_data_to_images(self, names_paths, end_file):
        """
        Convert CIFAR-10 data to images and save them to disk.

        Parameters:
            names_paths (str): The file path to save the names and labels of images.
            end_file (int): An indicator to manage the format of the output file.
        """
        names_labels = []  # List to hold image names and labels for saving to file
        for i in range(len(self.batch_data)):
            # Extract RGB channels and reshape them to 32x32
            R_channel = self.batch_data[i][0:1024].reshape(32, 32)  # Red channel
            G_channel = self.batch_data[i][1024:2048].reshape(32, 32)  # Green channel
            B_channel = self.batch_data[i][2048:3072].reshape(32, 32)  # Blue channel

            # Stack the channels to form a complete image
            image = np.stack((R_channel, G_channel, B_channel), axis=-1)
            # Construct the image file path
            image_filename = f"data/images/{self.labels_dict[self.batch_labels[i]]}/{self.batch_filenames[i]}"
            # Save the image using imageio
            imageio.imwrite(image_filename, image)

            # Prepare labels for writing to the text file
            if end_file == 0:
                names_labels.append(f"{self.batch_filenames[i]} {self.batch_labels[i]}\n")
            elif (end_file == 1) and (i == len(self.batch_data) - 1):
                names_labels.append(f"{self.batch_filenames[i]} {self.batch_labels[i]}")
            else:
                names_labels.append(f"{self.batch_filenames[i]} {self.batch_labels[i]}\n")

        # Write names and labels to the specified text file
        with open(names_paths, 'a') as f:
            f.writelines(names_labels)

    def run(self, batch_paths, names_paths, end_file):
        """
        Execute the entire process of loading data, creating directories,
        and converting data to images.

        Parameters:
            batch_paths (str): The file path to the CIFAR-10 batch file.
            names_paths (str): The file path to save the names and labels of images.
            end_file (int): An indicator to manage the format of the output file.

        Returns:
            dict: A dictionary mapping label indices to label names.
        """
        self.load_label_names('data/cifar-10-batches-py/batches.meta')  # Load label names
        self.create_images_folder()  # Create folder structure
        self.create_image_labels_to_string()  # Create label mapping
        self.load_batch(batch_paths)  # Load batch data
        self.convert_data_to_images(names_paths, end_file)  # Convert and save images
        return self.labels_dict  # Return the label dictionary
