import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import scipy
import SimpleITK as sitk
from timeit import default_timer
import torch.nn.functional as F
import torchvision.transforms as transforms
import random
import torchvision.transforms as T
def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

class RandomAugment(object):
    def __init__(self, output_size=None):
        self.output_size = output_size

    def __call__(self, sample):
        src, tar = sample['src'], sample['tar']

        if random.random() > 0.5:
            src, tar = random_rot_flip(src, tar)
            src = torch.from_numpy(src.astype(np.float32))
            tar = torch.from_numpy(tar.astype(np.float32))

            # print(src.shape, tar.shape, '    flip')

        elif random.random() > 0.5:
            src, tar = random_rotate(src, tar)
            src = torch.from_numpy(src.astype(np.float32))
            tar = torch.from_numpy(tar.astype(np.float32))

            # print(src.shape, tar.shape, '    rotate')

        # x, y = src.shape
        # if x != self.output_size[0] or y != self.output_size[1]:
        #     src = zoom(src, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
        #     tar = zoom(tar, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        sample = {'src': src, 'tar': tar}
        return sample



class Dataset_Only_Train_DiFuS_ICLR(Dataset):
    def __init__(self, Config=None, fixed_src=-1):
        self.test_type = Config["DataSet"]["test_type"]
        self.fixed_src = fixed_src
        if "Plant" in self.test_type:
            if Config["general"]["server"] == "My":
                path = "/home/nellie/data/DealData/For_Nivetha/plants_traindata.mat"
            elif Config["general"]["server"] == "Ri":
                path = "/scratch/bsw3ac/nellie/data/For_Nivetha/plants_traindata.mat"

            data = scipy.io.loadmat(path)

            self.SdefList = data['SdefList']
            self.src = data['src']
            self.tar = data['tar']
            self.SRGBdefList = data['SRGBdefList']
            self.srcRGB = data['srcRGB']
            self.tarRGB = data['tarRGB']
            self.src_time = data['src_time']
            self.tar_time = data['tar_time']
            self.text_data = None
            
            # (62, 1, 128, 128) (62, 1, 128, 128) (62, 1, 128, 128, 3) (62, 1, 128, 128, 3) (62, 1) (62, 1) (62, 7, 1, 128, 128) (62, 7, 1, 128, 128, 3)
            # print(self.src.shape, self.tar.shape, self.srcRGB.shape, self.tarRGB.shape, self.src_time.shape, self.tar_time.shape, self.SdefList.shape, self.SRGBdefList.shape)

        elif "OASIS" in self.test_type:
            if Config["general"]["server"] == "My":
                path = "/home/nellie/data/DealData/For_Nivetha/OASIS32D_traindata.mat"
            elif Config["general"]["server"] == "Ri":
                path = "/scratch/bsw3ac/nellie/data/For_Nivetha/OASIS32D_traindata.mat"
            data = scipy.io.loadmat(path)

            self.src = data['src']
            self.tar = data['tar']
            self.SdefList = data['SdefList']
            self.text_data = data['text_data']
            self.textid = data['textid']
            

    def __len__(self):
        return self.src.shape[0]
    
    def __getitem__(self, idx):
        if self.fixed_src != -1:
            idx = self.fixed_src
        if "Plant" in self.test_type:
            src = self.src[idx]; tar = self.tar[idx]
            srcRGB = self.srcRGB[idx]; tarRGB = self.tarRGB[idx]
            src_time = self.src_time[idx]; tar_time = self.tar_time[idx]
            SdefList = self.SdefList[idx]; SRGBdefList = self.SRGBdefList[idx]
            
            # (1, 128, 128) (1, 128, 128) (1, 128, 128, 3) (1, 128, 128, 3) (1,) (1,) (7, 1, 128, 128) (7, 1, 128, 128, 3)
            # print(src.shape, tar.shape, srcRGB.shape, tarRGB.shape, src_time.shape, tar_time.shape, SdefList.shape, SRGBdefList.shape)

            sample = {'src': src, 'tar': tar, 'srcRGB': srcRGB, 'tarRGB': tarRGB, 'src_time': src_time, 'tar_time': tar_time, 'SdefList': SdefList, 'SRGBdefList': SRGBdefList}
            return sample
        elif "OASIS" in self.test_type:
            src = self.src[idx]; tar = self.tar[idx]
            SdefList = self.SdefList[idx]; text_data = self.text_data[idx]; textidx = self.textid[idx]

            sample = {'src': src, 'tar': tar, 'SdefList': SdefList, 'text_data': text_data, 'textidx': textidx}
            return sample





class Dataset_Only_Test_DiFuS_ICLR(Dataset):
    def __init__(self, Config=None, fixed_src=-1):
        self.test_type = Config["DataSet"]["test_type"]
        self.fixed_src = fixed_src

        if "Plant" in self.test_type:
            if Config["general"]["server"] == "My":
                path = "/home/nellie/data/DealData/For_Nivetha/plants_testdata.mat"
            elif Config["general"]["server"] == "Ri":
                path = "/scratch/bsw3ac/nellie/data/For_Nivetha/plants_testdata.mat"
            data = scipy.io.loadmat(path)

            self.SdefList = data['SdefList']
            self.src = data['src']
            self.tar = data['tar']
            self.SRGBdefList = data['SRGBdefList']
            self.srcRGB = data['srcRGB']
            self.tarRGB = data['tarRGB']
            self.src_time = data['src_time']
            self.tar_time = data['tar_time']
            self.text_data = None


            self.transform_size1 = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                # T.RandomRotation(degrees=30),
                T.RandomRotation(degrees=90),
                T.RandomRotation(degrees=180),
                T.RandomRotation(degrees=270),
                # T.RandomResizedCrop(size=(128, 128), scale=(1.0, 1.0), ratio=(1.0, 1.0)),
                # T.ColorJitter(brightness=0.2, contrast=0.2)
            ])
            
            # (62, 1, 128, 128) (62, 1, 128, 128) (62, 1, 128, 128, 3) (62, 1, 128, 128, 3) (62, 1) (62, 1) (62, 7, 1, 128, 128) (62, 7, 1, 128, 128, 3)
            # print(self.src.shape, self.tar.shape, self.srcRGB.shape, self.tarRGB.shape, self.src_time.shape, self.tar_time.shape, self.SdefList.shape, self.SRGBdefList.shape)

        elif "OASIS" in self.test_type:
            if Config["general"]["server"] == "My":
                path = "/home/nellie/data/DealData/For_Nivetha/OASIS32D_testdata.mat"
            elif Config["general"]["server"] == "Ri":
                path = "/scratch/bsw3ac/nellie/data/For_Nivetha/OASIS32D_testdata.mat"

            data = scipy.io.loadmat(path)

            self.src = data['src']
            self.tar = data['tar']
            self.SdefList = data['SdefList']
            self.text_data = data['text_data']
            self.textid = data['textid']

            self.transform_size1 = None
            
    def __len__(self):
        if self.fixed_src != -1: #Fixe_src for CDM
            # return 16*5
            return 100
            # return 200
            # return 500
        # return 32

        
        return self.src.shape[0]
        # return int((self.src.shape[0]// 16) * 16)   #Metrics

    def __getitem__(self, idx):
        if self.fixed_src != -1:
            idx = self.fixed_src
        if "Plant" in self.test_type:
            src = self.src[idx]; tar = self.tar[idx]
            srcRGB = self.srcRGB[idx]; tarRGB = self.tarRGB[idx]
            src_time = self.src_time[idx]; tar_time = self.tar_time[idx]
            SdefList = self.SdefList[idx]; SRGBdefList = self.SRGBdefList[idx]
            
            # (1, 128, 128) (1, 128, 128) (1, 128, 128, 3) (1, 128, 128, 3) (1,) (1,) (7, 1, 128, 128) (7, 1, 128, 128, 3)
            # print(src.shape, tar.shape, srcRGB.shape, tarRGB.shape, src_time.shape, tar_time.shape, SdefList.shape, SRGBdefList.shape)

            sample = {'src': src, 'tar': tar, 'srcRGB': srcRGB, 'tarRGB': tarRGB, 'src_time': src_time, 'tar_time': tar_time, 'SdefList': SdefList, 'SRGBdefList': SRGBdefList}
            return sample
        elif "OASIS" in self.test_type:
            src = self.src[idx]; tar = self.tar[idx]
            SdefList = self.SdefList[idx]; text_data = self.text_data[idx]; textidx = self.textid[idx]

            sample = {'src': src, 'tar': tar, 'SdefList': SdefList, 'text_data': text_data, 'textidx': textidx}
            return sample



