import csv
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from skimage.transform import resize
from nilearn import surface
import nibabel as nib
from skimage import exposure
import matplotlib.pyplot as plt

index=5
root='/media/nas/HCP_S1200_3T/'
list_img = [dI for dI in os.listdir(root) if os.path.isdir(os.path.join(root,dI))]
list_img.sort()
rootf=root+list_img[index]+'/T1w/Diffusion/'
lmax=2
nums=0
for i in range(0,lmax):
    nums+=2*i+1
nums=6
if os.path.isfile(rootf+'wm.nii'):
    img = nib.load(os.path.join(rootf+'wm.nii'))
else:
    if os.path.isfile(rootf+'wm.mif'): 
        os.system('mrconvert '+rootf+'wm.mif '+rootf+'wm.nii')
        img = np.array(nib.load(os.path.join(rootf+'wm.nii')))
Affine=img.affine
img=img.get_data()
img=img[:,:,:,:nums]

# img=
nib.save(nib.Nifti1Image(img,affine=Affine), rootf+'wm_test.nii')
# def __len__(self):
#         return len(self.imglist)

# def __getitem__(self, index):
#         path = os.path.join(self.root,self.imglist[index])
        
#         img = nib.load(os.path.join(self.root,self.imglist[index]+'/T1w/Diffusion/data.nii.gz'))
        
#         A = np.zeros((145, 174, 145, 288))
#         A = img.get_data()
        
#         bvals=np.loadtxt(os.path.join(self.root,self.imglist[index]+'/T1w/Diffusion/bvals'))
#         bvecs=np.loadtxt(os.path.join(self.root,self.imglist[index]+'/T1w/Diffusion/bvecs'))
#         bval_range=[900,1100]
#         shell=np.where((bvals>bval_range[0]) & (bvals<bval_range[1]))
#         bvecs=bvecs[:,shell]
#         img=img[:,:,:,shell]
        
