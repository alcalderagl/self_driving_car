import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cv2
from skimage.feature import hog
import glob
import os
import pandas as pd

from tqdm import tqdm



class Utils:
    
    def __init__(self, raw_path = None, processed_path = None):
      self.raw_path = raw_path
      self.processed_path = processed_path
      self.pickle_file = None
      if raw_path and processed_path:
        self.pickle_file = processed_path + "hog_pedestrians_dataset.pkl"
      pass

    def read_images(self, path_images = []):
        """
        Reads multiple images from the provided file paths.

        Parameters:
        path_images (list): List of file paths to the images to be read.

        Returns:
        images (list): List of loaded images as NumPy arrays.
        """
        images = []
        
        # Iterate over each image path and read the image
        for path_image in path_images:
            # Load image using OpenCV
            images.append(cv2.imread(path_image))
        # Return the list of images
        return images

    def convert_BGR2GRAY(self, color_images = []):
        """
        Converts a list of BGR images to grayscale.

        Parameters:
        color_images (list): List of BGR images as NumPy arrays.

        Returns:
        images (list): List of grayscale images as NumPy arrays.
        """
        images = []
        
        # Convert each BGR image to grayscale
        for color_image in color_images:
            images.append(cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY))
            
        # Return the list of images
        return images


    def extract_hog(self, image):
        """
        Extracts Histogram of Oriented Gradients (HOG) features from a grayscale image.

        Parameters:
        image (numpy.ndarray): A single grayscale image (2D array). 
        
        Returns:
        tuple: 
            - features (ndarray): 1D array of HOG features.
            - hog_image (ndarray): Visualization of the HOG (used for debugging or plotting).
        """
        
        return hog(
            image,
            orientations = 11,
            pixels_per_cell = (16,16),
            cells_per_block = (2,2),
            transform_sqrt = False,
            visualize = True, # Returns a visualization of the HOG
            feature_vector=True # Output will be a 1D feature vector
        )

    def process_hog_imgs(self, df):
        """
        Processes a DataFrame of image paths and extracts HOG features for each image.

        Parameters:
        df (pandas.DataFrame): DataFrame containing a 'filename' column with image paths.

        Returns:
        None
        """
        
        # Initialize a new column to store HOG features
        df["features"] = None
        
        # Iterate over each row in the DataFrame with a progress bar
        for i, row in tqdm(df.iterrows(), total=len(df)):
            img_path = row["filename"]
            
            # Read image using matplotlib (note: mpimg.imread returns float32 images)
            img_color = mpimg.imread(img_path)
            
            # Convert the color image to grayscale using OpenCV
            img_gray = cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)
            
            # If image is not loaded properly, skip it
            if img_gray is None:
                print(f"coud not read {img_path}")
                continue
            
            # Extract HOG features from the grayscale image
            features, _ = self.extract_hog(img_gray)
            # Store features in the 'features' column of the DataFrame
            df.at[i, "features"] = features
        
        # Save the updated DataFrame to a .pkl file for efficient loading later
        df.to_pickle(self.pickle_file)
        
    
    def is_pedestrian(self, x):
        """
        Converts a binary label to a descriptive string.

        Parameters:
        x (int): The label value (1 for pedestrian, 0 for not pedestrian).

        Returns:
        str: "pedestrian" if x is 1, otherwise "not pedestrian".
        """
        return "pedestrian" if x == 1 else "not pedestrian"

    def show_predictions(self, image_ids, y_true, y_pred, n=10, nrows=5, ncols=2, figsize=(12,8)):
        """
        Displays a set of image predictions with true and predicted labels.

        Parameters:
        image_ids (list): List of image file paths to display.
        y_true (list or array): Ground truth labels corresponding to the images.
        y_pred (list or array): Predicted labels from the classifier.
        n (int): Number of images to show. Default is 10.
        nrows (int): Number of rows in the plot grid. Default is 2.
        ncols (int): Number of columns in the plot grid. Default is 5.
        figsize (tuple): Size of the figure to display. Default is (12, 8).

        Returns:
        None
        """
        images = []
        labels = []
        
        # Loop through the first `n` predictions
        for i in range(n):
            image_path = image_ids[i]

            
            # Load image from file and convert from BGR to RGB for proper visualization
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            images.append(img)
            
            # Format label with true and predicted values
            labels.append(f"Image: {os.path.basename(image_path)}\nTrue: {self.is_pedestrian(y_true[i])}\nPred: {self.is_pedestrian(y_pred[i])}")
            
        # Use helper function to plot images in a grid with labels
        self.plot_images(images=images, labels=labels, nrows=nrows, ncols=ncols, figsize=figsize)
            


    def plot_images(self, images=[], labels=[], nrows=1, ncols=2, figsize=(12,8), cmap="gray", hideAxis=True):
        """
        Plots a list of images in a grid layout using Matplotlib.

        Parameters:
        images (list): List of images to be plotted.
        labels (list): List of labels for each image. Default is an empty list.
        nrows (int): Number of rows in the grid. Default is 1.
        ncols (int): Number of columns in the grid. Default is 2.
        figsize (tuple): Size of the figure in inches. Default is (12, 8).
        cmap (str): Colormap to be used for displaying the images. Default is "gray".

        Returns:
        None
        """
        if len(images) > 1:
            # set the subplots
            fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
            # Iterate over each axis defined in M*N as flat
            for i, ax in enumerate(axs.flat):
                
                # validates that the dimensions of axs fits with images array parameter
                if i < len(images):
                    # plot the image
                    ax.imshow(images[i], cmap=cmap)
                    # if there are labels defined then set a title
                    if len(labels) > 0:
                        ax.set_title(labels[i])
                    # dont show the axis ticks
                    if hideAxis:
                        ax.axis("off")
                else:
                    # dont show an empty plot
                    ax.axis("off")
        else:
            plt.figure(figsize=figsize)
            plt.imshow(images[0], cmap=cmap)
            if len(labels) > 0:
                # Display width and height in the title
                plt.title(labels[0])  
            if hideAxis:
                plt.xticks([])
                plt.yticks([])
        # fits the plots
        plt.tight_layout()
        plt.show()