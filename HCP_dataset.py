import csv
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from skimage import transform
from nilearn import surface
import nibabel as nib
from skimage import exposure
import random

class HCPdataset(Dataset):
    def __init__(self, train=True,imgtype = 'fod',n_vol=45,res=64,augmentation=False,gen_T1=False,z64=False):
        if gen_T1==False:
            root='/sc-scratch/sc-scratch-synmri/HCP_data_'+str(int(res))+'/'
        else:
            root='/sc-scratch/sc-scratch-synmri/HCP_data_'+str(int(res))+'_T1/'
        if z64==True:
            root='/sc-scratch/sc-scratch-synmri/HCP_data_128_64/'
        self.augmentation = augmentation
        self.root = root
        self.n_vol=n_vol
        list_img = [dI for dI in os.listdir(self.root)]
        list_img = [val for val in list_img if not val.endswith("_f.nii")]
        if augmentation == False:
            list_img = [val for val in list_img if not val.startswith("1_")]
            list_img = [val for val in list_img if not val.startswith("2_")]
            list_img = [val for val in list_img if not val.startswith("3_")]
            list_img = [val for val in list_img if not val.startswith("4_")]
        list_img.sort()
        self.imglist = list_img
        self.imgtype = imgtype
        self.res=res
        
    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        path = os.path.join(self.root,self.imglist[index])
        
        if self.imgtype=='dwi':
            img = nib.load(os.path.join(self.root,self.imglist[index]))        
            A = np.zeros((145, 174, 145, 288))
            A = img.get_data()
            
            bvals=np.loadtxt(os.path.join(self.root,self.imglist[index]+'/T1w/Diffusion/bvals'))
            bvecs=np.loadtxt(os.path.join(self.root,self.imglist[index]+'/T1w/Diffusion/bvecs'))
            bval_range=[900,1100]
            shell=np.where((bvals>bval_range[0]) & (bvals<bval_range[1]))
            bvecs=bvecs[:,shell]
            img=img[:,:,:,shell]
        
        root=self.root
        res=self.res
        n_vol=self.n_vol
        img = nib.load(os.path.join(root+self.imglist[index]))
        img=img.get_fdata()
        img=img[:,:,:,:n_vol]

        img = np.transpose(img, (3, 0, 1, 2))        
        imageout = torch.from_numpy(img.copy()).float()       
        return imageout
