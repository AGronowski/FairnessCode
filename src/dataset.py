import torch
import torchvision
from torchvision import transforms
from skimage import io, transform
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
import argparse
import matplotlib.pyplot as plt
import os
from PIL import Image

torch.manual_seed(2020)
np.random.seed(2020)

def getimages(csv_file,root_dir,image_size,transform):
    frame = pd.read_csv(csv_file)

    N = len(frame)
    images = torch.zeros(N, 3, image_size, image_size)
    for n in range(N):
        img_name = os.path.join(root_dir,
                                frame.iloc[n, 1])
        image = io.imread(img_name)  # numpy.ndarray
        image = torchvision.transforms.functional.to_pil_image(image)  # PIL image

        if transform:
            image = transform(image)

        images[n] = image  # shape (28,28)
    return images

class Celeba_dataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # self.targets = self.frame.iloc[:,2]
        # self.sensitives = self.frame.iloc[:,3]
        # self.images = images

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir,
                                self.frame.iloc[index, 1])
        image = io.imread(img_name) #numpy.ndarray
        image = torchvision.transforms.functional.to_pil_image(image) #PIL image

        if self.transform:
            image = self.transform(image)

        target = self.frame.iloc[index,2] #target is age
        sensitive = self.frame.iloc[index,3] #sensitive is gender

        return image, target, sensitive

class Eyepacs_dataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # self.targets = self.frame.iloc[:,2]
        # self.sensitives = self.frame.iloc[:, 4]
        # self.images = images


    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir,
                                self.frame.iloc[index, 1])
        image = io.imread(img_name) #numpy.ndarray
        image = torchvision.transforms.functional.to_pil_image(image) #PIL image

        if self.transform:
            image = self.transform(image)

        target = self.frame.iloc[index,2] #target is diabetic_retinopathy
        sensitive = self.frame.iloc[index,4] #sensitive is ita_dark

        return image, target, sensitive



def get_eyepacs(debugging):
    root_dir = "../data/eyepacs_small"

    image_size = 256

    transform = transforms.Compose([transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])


    if debugging:
        trainset = Eyepacs_dataset('../data/eyepacs_debugging.csv',root_dir,
                                  transform)
        testset = Eyepacs_dataset('../data/eyepacs_debugging.csv',root_dir,
                                  transform)
    else:
        csv_file = '../data/eyepacs_control_train_jpeg.csv'
        trainset = Eyepacs_dataset(csv_file,root_dir,
                                  transform)
        testset = Eyepacs_dataset('../data/eyepacs_test_dr_ita_jpeg.csv',root_dir,
                                      transform)

    return trainset, testset

def get_celeba(debugging,dataset):
    root_dir = '../data/celeba_small'
    image_size = 64
    transform = transforms.Compose([transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

    if debugging:

        csv_file = '../data/celeba_debugging.csv'
        trainset = Celeba_dataset(csv_file,root_dir,
                                  transform)
        testset = Celeba_dataset(csv_file,root_dir,transform)
    else:
        if dataset == 0: #gender
            trainset = Celeba_dataset('../data/celeba_gender_train_jpg.csv',root_dir,
                                      transform)
        elif dataset == 1 : #race
            trainset = Celeba_dataset('../data/celeba_skincolor_train_jpg.csv',root_dir,
                                      transform)
        else:
            print("error")
            return False
        testset = Celeba_dataset('../data/celeba_balanced_combo_test_jpg.csv',root_dir,
                                  transform)


    return trainset, testset

# get_celeba()
# csv_file = '../data/celeba_gender_train_jpg.csv'
# frame = pd.read_csv(csv_file)
# root_dir = '../data/celeba'
# img_name =os.path.join(root_dir,
#              frame.iloc[1, 1])
# image = io.imread(img_name)

from shutil import copyfile
from sys import exit
from progressbar import progressbar

'''
Change .png to .jpeg
'''

# csv_file = '../data/eyepacs_control_train.csv'
# frame = pd.read_csv(csv_file)
# root_dir = '../../eyepacs/train'
#
# dir = '../../eyepacs_small'
# from progressbar import progressbar
#
# for i in progressbar(range(len(frame))):
#     img_name = frame.iloc[i, 1]
#     name_beginning = img_name[:-3]
#     name = name_beginning + 'jpeg'
#     frame.iloc[i, 1] = name
#
# frame.to_csv("eyepacs_control_train_jpeg.csv",index=False)

'''
Copy image into another folder
'''

# csv_file = '../data/eyepacs_control_train_jpeg.csv'
# frame = pd.read_csv(csv_file)
# root_dir = '../../eyepacs/test'
# dir = '../../eyepacs_small'
#
# for i in progressbar(range(len(frame))):
#     img_path = os.path.join(root_dir,
#                             frame.iloc[i, 1])
#     output_path = os.path.join(dir,
#                           frame.iloc[i, 1])
#
#     try:
#         copyfile(img_path, output_path)
#         print(img_path)
#     except:
#         print("Unexpected error:")
