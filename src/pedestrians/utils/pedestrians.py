import os
import glob
import pandas as pd
import cv2
from lxml import etree

class Pedestrians:
    
    def __init__(self, image_file_path, xml_file_path):
        """
        Initializes the pedestrian dataset processor.

        Purpose:
        This method initializes the object by collecting image and XML annotation file paths,
        sets the destination path for the output dataset CSV, and processes the data to extract
        pedestrian crops and labels.

        Parameters:
        image_file_path (str): Path to the directory or glob pattern where image files are located.
        xml_file_path (str): Path to the directory or glob pattern where XML annotation files are located.

        Returns:
        None
        """
        
        # Get lists of image and XML file paths (based on input patterns or directories)
        img_paths, xml_paths = self._get_file_paths(image_file_path=image_file_path, xml_file_path=xml_file_path)
        # Set the path where the resulting dataset CSV will be saved
        self.dataset_name = "data/pedestrians_dataset.csv"
        # Directory to save cropped pedestrian images
        self.output_dir = "data/pedestrian_crops"
        # Process the image and annotation files to extract pedestrian crops and labels
        self._get_pedestrian_images(img_paths=img_paths, xml_paths=xml_paths)
        

    def _get_file_paths(self, image_file_path, xml_file_path):
        """
        Retrieves and sorts image and XML annotation file paths.

        Parameters:
        image_file_path (str): Glob pattern or directory path to image files (e.g., "images/*.jpg").
        xml_file_path (str): Glob pattern or directory path to XML annotation files (e.g., "annotations/*.xml").

        Returns:
        tuple: A tuple containing two lists:
            - image_files (list): Sorted list of image file paths.
            - xml_files (list): Sorted list of XML annotation file paths.
        """
        image_files = sorted(glob.glob(image_file_path))
        xml_files = sorted(glob.glob(xml_file_path))
        
        return image_files, xml_files
    
    def _get_pedestrian_images(self, img_paths, xml_paths):
        """
        Extracts pedestrian crops from images based on XML annotations and saves them.

        Parameters:
        img_paths (list): List of image file paths.
        xml_paths (list): List of XML annotation file paths.

        Returns:
        None
        """
        
        dataset_entries = []
        
        # Loop through image and XML file pairs
        for (img_path, xml_path) in zip(img_paths, xml_paths):
            # Parse the XML annotation file
            tree = etree.parse(xml_path)
            root = tree.getroot()
            
            # Load the corresponding image
            image = cv2.imread(img_path)
            image_filename = root.find("filename").text
            
            
            # Loop through all <object> elements in the XML
            for idx, obj in enumerate(root.findall("object")):
                # label is a tag to identify if it is a self-person or person
                label = obj.find("name").text
                
                # Extract bounding box coordinates
                bbox = obj.find("bndbox")
                xmin = int(float(bbox.find("xmin").text))
                ymin = int(float(bbox.find("ymin").text))
                xmax = int(float(bbox.find("xmax").text))
                ymax = int(float(bbox.find("ymax").text))
                
                # crop the region from the image
                crop = image[ymin: ymax, xmin: xmax]
                # resize it to 64x64
                crop = cv2.resize(crop, (64,64))
                
                # Construct a unique filename for the cropped image
                crop_filename = f"{os.path.splitext(image_filename)[0]}_{idx}.jpg"
                crop_filename = os.path.join(self.output_dir, crop_filename)
                
                # Save the cropped image to disk
                cv2.imwrite(crop_filename, crop)
                
                # Add an entry to the dataset (label: 1 = person, 0 = other)
                dataset_entries.append({
                    "filename": crop_filename,
                    "label": 1 if label == "person" else 0
                })
        
        # Create a DataFrame with all image paths and labels and save as CSV
        df = pd.DataFrame(dataset_entries)
        df.to_csv(self.dataset_name, index=False)
        print("Pedestrians dataset generated")