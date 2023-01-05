import albumentations as A
import math
import numpy as np
import os
import random
import yaml

from pathlib import Path
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow import keras


class Dataset(keras.utils.Sequence):
    def __init__(self, set_type: str, path: str = "", seeding: bool = False):
        # load parameters from config
        with open(path + "./config.yaml", "r") as stream:
            config = yaml.safe_load(stream)
        self.set_type = set_type
        self.MAX_IMAGES = config["max_number_images"]
        self.SUBIMAGES = config["subimages"]
        self.SAVE_TRAINING_IMAGES = config["save_training_images"]
        self.UPSCALING_FACTOR = config["model_upscaling_factor"]
        self.LR_IMG_SIZE = tuple(config["lr_size"])
        self.HR_IMG_SIZE = tuple(config["hr_size"])
        self.BATCH_SIZE = config["batch_size"]
        self.LR_IMG_FOLDER = config["input_lr_folder"]
        self.HR_IMG_FOLDER = config["input_hr_folder"]

        if seeding:
            random.seed(config["seed"])

        if self.SAVE_TRAINING_IMAGES:
            # prepare folder
            Path("datasets/debugImages/LR/" + self.set_type + "/").mkdir(parents=True, exist_ok=True)
            Path("datasets/debugImages/HR/" + self.set_type + "/").mkdir(parents=True, exist_ok=True)

        # prepare transformation for scale
        self.transform = A.ToFloat(max_value=255)

        self.initialize_dataset(config)

        # number of LR images should be the same as number of HR images
        assert len(self.lr_images) == len(
            self.hr_images), "Number of images are different for LR and HR."

    def __len__(self):
        return math.ceil(len(self.hr_images) / self.BATCH_SIZE)

    def __getitem__(self, idx):
        batch_lr_images = self.lr_images[idx * self.BATCH_SIZE:(idx + 1) * self.BATCH_SIZE]
        batch_hr_images = self.hr_images[idx * self.BATCH_SIZE:(idx + 1) * self.BATCH_SIZE]

        return (batch_lr_images, batch_hr_images)

    def initialize_dataset(self, config):
        self.lr_images = np.empty((0, self.LR_IMG_SIZE[1], self.LR_IMG_SIZE[0], 3))
        self.hr_images = np.empty((0, self.HR_IMG_SIZE[1], self.HR_IMG_SIZE[0], 3))

        # get image names from the folders
        self.hr_image_filenames = np.sort([x for x in os.listdir(
            self.HR_IMG_FOLDER) if x.endswith(".png")])
        self.lr_image_filenames = np.sort([x for x in os.listdir(
            self.LR_IMG_FOLDER) if x.endswith(".png")])

        # take only a specified number of images for datasets
        if(self.MAX_IMAGES != 0):
            self.hr_image_filenames = self.hr_image_filenames[:self.MAX_IMAGES]
            self.lr_image_filenames = self.lr_image_filenames[:self.MAX_IMAGES]

        # Define dataset splits
        training_size = int(len(self.lr_image_filenames) * (config["training_split"] / 100))

        validation_size = int(len(self.lr_image_filenames) * (config["validation_split"] / 100))

        test_size = int(len(self.lr_image_filenames) * (config["test_split"] / 100))

        # initialize images according to splits
        if self.set_type == "train":
            self.hr_image_filenames = self.hr_image_filenames[:training_size]
            self.lr_image_filenames = self.lr_image_filenames[:training_size]
        elif self.set_type == "val":
            self.hr_image_filenames = self.hr_image_filenames[training_size:training_size + validation_size]
            self.lr_image_filenames = self.lr_image_filenames[training_size:training_size + validation_size]
        else:
            self.hr_image_filenames = self.hr_image_filenames[-test_size:]
            self.lr_image_filenames = self.lr_image_filenames[-test_size:]

        for im1, im2 in zip(self.lr_image_filenames, self.hr_image_filenames):
            # get LR und HR image
            lr_image = Image.open(os.path.join(self.LR_IMG_FOLDER, im1))
            hr_image = Image.open(os.path.join(self.HR_IMG_FOLDER, im2))

            # get random subimage(s) from the actual LR and HR image
            for i in range(self.SUBIMAGES):
                # get max coordinates from which we can crop the image
                max_width = lr_image.width - self.LR_IMG_SIZE[0]
                max_height = lr_image.height - self.LR_IMG_SIZE[1]
                x_rand = random.randint(0, max_width) if max_width >= 0 else 0
                y_rand = random.randint(0, max_height) if max_height >= 0 else 0

                # cropping
                hr_cropped = hr_image.crop(
                    (x_rand * 2, y_rand * 2, x_rand * 2 + self.HR_IMG_SIZE[0],
                     y_rand * 2 + self.HR_IMG_SIZE[1]))
                lr_cropped = lr_image.crop(
                    (x_rand, y_rand, x_rand + self.LR_IMG_SIZE[0], y_rand + self.LR_IMG_SIZE[1]))

                self.hr_images = np.append(self.hr_images, [self.transform(
                    image=np.array(hr_cropped))["image"]], axis=0)
                self.lr_images = np.append(self.lr_images, [self.transform(
                    image=np.array(lr_cropped))["image"]], axis=0)

                if self.SAVE_TRAINING_IMAGES:  # for debugging training images
                    lr_cropped.save(
                        "datasets/debugImages/LR/" + self.set_type + "/" + lr_image.filename.split(".png")[0].split("/")
                        [-1] + ".png")
                    hr_cropped.save(
                        "datasets/debugImages/HR/" + self.set_type + "/" + hr_image.filename.split(".png")[0].split("/")
                        [-1] + ".png")
