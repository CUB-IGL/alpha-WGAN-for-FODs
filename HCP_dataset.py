import csv
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from skimage.transform import resize
from nilearn import surface
import nibabel as nib
from skimage import exposure
import random

class HCPdataset(Dataset):
    def __init__(self, train=True,imgtype = 'fod',lmax=6,is_flip=False,augmentation=False):
        root='/media/nas/HCP_S1200_3T/'
        self.augmentation = augmentation
        self.root = root
        self.lmax=lmax
        list_img = [dI for dI in os.listdir(self.root) if os.path.isdir(os.path.join(self.root,dI))]
        list_img.sort()
        self.imglist = list_img
        self.is_flip = is_flip
        self.imgtype = imgtype
        
    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        path = os.path.join(self.root,self.imglist[index])
        
        if self.imgtype=='dwi':
            img = nib.load(os.path.join(self.root,self.imglist[index]+'/T1w/Diffusion/data.nii.gz'))        
            A = np.zeros((145, 174, 145, 288))
            A = img.get_data()
            
            bvals=np.loadtxt(os.path.join(self.root,self.imglist[index]+'/T1w/Diffusion/bvals'))
            bvecs=np.loadtxt(os.path.join(self.root,self.imglist[index]+'/T1w/Diffusion/bvecs'))
            bval_range=[900,1100]
            shell=np.where((bvals>bval_range[0]) & (bvals<bval_range[1]))
            bvecs=bvecs[:,shell]
            img=img[:,:,:,shell]
            
        if self.imgtype=='fod':
            rootf=self.root+self.imglist[index]+'/T1w/Diffusion/'
            nums=self.lmax
            while os.path.isfile(rootf+'wm.mif')==False:
                index=random.randint(0, self.__len__()-1)
                rootf=self.root+self.imglist[index]+'/T1w/Diffusion/'
            if os.path.isfile(rootf+'wm.nii')==False:
                os.system('mrconvert '+rootf+'wm.mif '+rootf+'wm.nii')
            img = nib.load(os.path.join(rootf+'wm.nii'))
            img=img.get_data()
            img=img[:,:,:,:nums]
            img = np.transpose(img, (3, 0, 1, 2))
        
        if self.is_flip:
            img = np.swapaxes(img,1,2)
            img = np.flip(img,1)
            img =np.flip(img,2)
        
        if self.augmentation:
            random_n = torch.rand(1)
            random_i = 0.3*torch.rand(1)[0]+0.7
            if random_n[0] > 0.5:
                img = np.flip(img,0)

        imageout = torch.from_numpy(img).float()       
        return imageout