#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 14:51:19 2022

@author: adt
"""

import os
import numpy as np
root='/media/nas/HCP_S1200_3T/'
count_ex=0
count_ex2=0
count_nex=0
list_img = [dI for dI in os.listdir(root) if os.path.isdir(os.path.join(root,dI))]
list_img.sort()
c_del=0
exist=np.ones((np.size(list_img)),dtype=bool)
for i in range(np.size(list_img)):
    exist[i]=os.path.isfile(root+list_img[i]+'/T1w/Diffusion/wm.mif')
c=np.where(exist==False)[0]
for i in range(np.size(c)):
    del list_img[c[i]-c_del]
    c_del+=1
# exist=np.ones((np.size(list_img)),dtype=bool)
for i in range(np.size(list_img)):
    if os.path.isfile(root+list_img[i]+'/T1w/Diffusion/wm.nii'):
        count_ex+=1
    if os.path.isfile(root+list_img[i]+'/T1w/Diffusion/wm.mif'):
        count_ex2+=1
    else:
        count_nex+=1
