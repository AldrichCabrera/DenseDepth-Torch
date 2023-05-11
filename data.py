#!/usr/bin/python
import torch
import numpy as np
import pandas as pd
import csv
import random

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from PIL import Image
from io import BytesIO
from itertools import permutations

_check_pil = lambda x: isinstance(x, Image.Image)
_check_np_img = lambda x: isinstance(x, np.ndarray)

class RandomHorizontalFlip(object):
    def __call__(self, sample):

        img, depth = sample["image"], sample["depth"]

        if not _check_pil(img):
            raise TypeError("Expected PIL type. Got {}".format(type(img)))
        if not _check_pil(depth):
            raise TypeError("Expected PIL type. Got {}".format(type(depth)))

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {"image": img, "depth": depth}

class RandomChannelSwap(object):
    def __init__(self, probability):
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample["image"], sample["depth"]

        if not _check_pil(image):
            raise TypeError("Expected PIL type. Got {}".format(type(image)))
        if not _check_pil(depth):
            raise TypeError("Expected PIL type. Got {}".format(type(depth)))

        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[..., list(self.indices[random.randint(0, len(self.indices) - 1)])])

        return {"image": image, "depth": depth}

class depthDatasetMemory(Dataset):
    def __init__(self, nyu2_train, transform=None):
        self.nyu_dataset = nyu2_train
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.nyu_dataset[idx]
        image = Image.open("./"+sample[0])
        depth = Image.open("./"+sample[1])
        sample = {"image": image, "depth": depth}
        
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.nyu_dataset)

class ToTensor(object):
    def __init__(self, is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, depth = sample["image"], sample["depth"]
        image = self.to_tensor(image)
        #depth = depth.resize((160, 120))
        depth = depth.resize((320, 240))

        if self.is_test:
            depth = self.to_tensor(depth).float() / 1000
        else:
            depth = self.to_tensor(depth).float() * 1000

        # put in expected range
        depth = torch.clamp(depth, 10, 1000)

        return {"image": image, "depth": depth}

    def to_tensor(self, pic):
        if not (_check_pil(pic) or _check_np_img(pic)):
            raise TypeError("pic should be PIL Image or ndarray. Got {}".format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == "I":
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == "I;16":
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == "YCbCr":
            nchannel = 3
        elif pic.mode == "I;16":
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

def getNoTransform(is_test=False):
    return transforms.Compose([ToTensor(is_test=is_test)])

def getDefaultTrainTransform():
    return transforms.Compose([RandomHorizontalFlip(), RandomChannelSwap(0.5), ToTensor()])

def getTrainingTestingData(path, batch_size):
    train_path = 'espada_train.csv'
    #train_path = './data/nyu2_train.csv'
    test_path  = 'espada_test.csv'
    #test_path  = './data/nyu2_test.csv'

    with open(train_path, newline='') as file_csv:
        lector_csv = csv.reader(file_csv, delimiter=',', quotechar='"')
        nyu2_train = [fila for fila in lector_csv]

    #print("nyu2_train", nyu2_train)
    print("LENnyu2_train", len(nyu2_train))

    with open(test_path, newline='') as file_csv:
        lector_csv = csv.reader(file_csv, delimiter=',', quotechar='"')
        nyu2_test = [fila for fila in lector_csv]

    #print("nyu2_test", nyu2_test)
    print("LENnyu2_test", len(nyu2_test))

    transformed_training = depthDatasetMemory(nyu2_train, transform=getDefaultTrainTransform())
    transformed_testing = depthDatasetMemory(nyu2_test, transform=getNoTransform())

    return DataLoader(transformed_training, batch_size, shuffle=True), DataLoader(transformed_testing, 
        batch_size, shuffle=False)

def load_testloader(path, batch_size=1):
    data, nyu2_train = loadZipToMem(path)
    transformed_testing = depthDatasetMemory(data, nyu2_train, transform=getNoTransform())
    
    return DataLoader(transformed_testing, batch_size, shuffle=False)
