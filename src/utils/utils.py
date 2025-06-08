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


    def __init__(self):
        # Diccionario con nombres de clase GTSRB
        self.id_to_name = {
            0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)", 2: "Speed limit (50km/h)", 3: "Speed limit (60km/h)",
            4: "Speed limit (70km/h)", 5: "Speed limit (80km/h)", 6: "End of speed limit (80km/h)", 7: "Speed limit (100km/h)",
            8: "Speed limit (120km/h)", 9: "No passing", 10: "No passing for vehicles > 3.5 tons", 11: "Right-of-way at intersection",
            12: "Priority road", 13: "Yield", 14: "Stop", 15: "No vehicles", 16: "Vehicles > 3.5 tons prohibited",
            17: "No entry", 18: "General caution", 19: "Dangerous curve left", 20: "Dangerous curve right",
            21: "Double curve", 22: "Bumpy road", 23: "Slippery road", 24: "Road narrows on the right", 25: "Road work",
            26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing", 29: "Bicycles crossing",
            30: "Beware of ice/snow", 31: "Wild animals crossing", 32: "End of all restrictions", 33: "Turn right ahead",
            34: "Turn left ahead", 35: "Ahead only", 36: "Go straight or right", 37: "Go straight or left",
            38: "Keep right", 39: "Keep left", 40: "Roundabout mandatory", 41: "End of no passing", 42: "End of no passing >3.5t"
        }

    def show_predictions(self, image_ids, y_true, y_pred, n=10, nrows=2, ncols=5, figsize=(12, 6)):
        images = []
        labels = []

        for i in range(n):
            image_path = image_ids[i]
            if not os.path.exists(image_path):
                print(f"Imagen no encontrada: {image_path}")
                continue

            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

            true_label = self.id_to_name.get(y_true[i], f"Clase {y_true[i]}")
            pred_label = self.id_to_name.get(y_pred[i], f"Clase {y_pred[i]}")
            labels.append(f"True: {true_label}\nPred: {pred_label}")

        self.plot_images(images=images, labels=labels, nrows=nrows, ncols=ncols, figsize=figsize)

    def plot_images(self, images=[], labels=[], nrows=1, ncols=2, figsize=(12, 8), cmap="gray", hideAxis=True):
        if len(images) > 1:
            fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
            for i, ax in enumerate(axs.flat):
                if i < len(images):
                    ax.imshow(images[i], cmap=cmap)
                    if len(labels) > 0:
                        ax.set_title(labels[i], fontsize=8)
                    if hideAxis:
                        ax.axis("off")
                else:
                    ax.axis("off")
        else:
            plt.figure(figsize=figsize)
            plt.imshow(images[0], cmap=cmap)
            if len(labels) > 0:
                plt.title(labels[0])
            if hideAxis:
                plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.show()
