#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 15:59:11 2022

@author: adt
"""
f_name='FODs_low_res'

import os
import numpy as np
import shutil
root='/media/nas/HCP_S1200_3T/'
list_img = [dI for dI in os.listdir(root) if os.path.isdir(os.path.join(root,dI))]
list_img.sort()
if not os.path.exists('/media/nas/FODs_low_res'):
    os.mkdir('/media/nas/FODs_low_res')
for i in range(100):
    if os.path.isfile(root+list_img[i]+'/T1w/Diffusion/wm.nii')==True:
        shutil.copyfile(root+list_img[i]+'/T1w/Diffusion/wm.nii', '/media/nas/FODs_low_res/'+list_img[i]+'.nii')
shutil.make_archive(f_name, 'zip', f_name)

