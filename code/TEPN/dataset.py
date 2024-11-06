import os
import cv2
import torch
import scipy.io as scio
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class ToTensor(object):
    def __call__(self, sample, mu=0.5, sigma=0.5):
        input, label_cn2, label_integral, gt = sample['input'], sample['label_cn2'], sample['label_integral'], sample[
            'gt']
        input = np.ascontiguousarray(input).transpose((0,1,4,2,3))
        gt = np.ascontiguousarray(gt).transpose(0,3,1,2)
        label_cn2 = np.ascontiguousarray(label_cn2).transpose((2,0,1))
        label_cn2 = torch.from_numpy(label_cn2).float() * 1e9 # physics
        # label_cn2 = torch.from_numpy(label_cn2).float() * 5e12 # Antarctica
        label_integral = np.ascontiguousarray(label_integral).transpose((2,0,1))
        label_integral = torch.from_numpy(label_integral).float()*5e7 # pysics
        # label_integral = torch.from_numpy(label_integral).float() * 2.5e11 # Antarctica

        sample['input'] = (torch.from_numpy(input).float() / 255.0 - mu) / sigma
        sample['gt'] = (torch.from_numpy(gt).float() / 255.0 - mu) / sigma
        sample['label_cn2'] = label_cn2
        sample['label_integral'] = label_integral
        return sample


class VideoFrameDataset(Dataset):
    def __init__(self, para,flag):
        if flag == 0:
            self.root_path = para.data_root
        else:
            self.root_path = para.data_test_root

        self.temporal_len = para.time_length
        self.flist = os.listdir(self.root_path)

        self.transform = transforms.Compose([ToTensor()])

    def __getitem__(self, idx):
        fille_name = self.flist[idx]
        input_path = os.path.join(self.root_path, fille_name, 'input')
        labels_path = os.path.join(self.root_path, fille_name, 'truth')
        video_list = os.listdir(input_path)
        video_list.sort()
        lables_list = os.listdir(labels_path)
        lables_list.sort()
        gt = np.zeros((5,320,480,3))
        for gt_index in range(5):
            gt[gt_index] = cv2.imread(os.path.join(labels_path, lables_list[gt_index]))
        label_cn2 = scio.loadmat(os.path.join(labels_path, lables_list[5]))
        label_cn2 = label_cn2['Cn2_Mat']
        label_integral = scio.loadmat(os.path.join(labels_path, lables_list[6]))
        label_integral = label_integral['integral3dMat']
        vid = np.zeros((5,self.temporal_len,320,480,3))

        for vid_index in range(5):
            video = cv2.VideoCapture(os.path.join(input_path, video_list[vid_index]))
            for frame_index in range(self.temporal_len):
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                vid[vid_index,frame_index,:,:,:]=video.read()[1]
            video.release()

        sample = {'input': vid, 'label_cn2': label_cn2, 'label_integral': label_integral, 'gt': gt}
        sample = self.transform(sample)

        return sample
    def __len__(self):
        return int(len(self.flist))

def create_dataset(dataset, para):
    dataloader = DataLoader(dataset, batch_size=para.batch_size, shuffle=para.shuffle, drop_last=True)
    return dataloader

def create_dataset_v(dataset, para):
    dataloader = DataLoader(dataset, batch_size=para.batch_size, shuffle=False, drop_last=True)
    return dataloader

