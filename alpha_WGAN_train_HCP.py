#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 17:28:38 2022

@author: adt
"""

import numpy as np
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
import nibabel as nib
from torch.utils.data.dataset import Dataset
from torch.utils.data import dataloader
from nilearn import plotting
from HCP_dataset import *
from Model_alphaWGAN import *
from conv4d import *
from datetime import datetime
import matplotlib.pyplot as plt
torch.cuda.empty_cache()

BATCH_SIZE=2
gpu = True
workers = 4

l_max=45

LAMBDA= 10
_eps = 1e-15

Use_BRATS=False
Use_ATLAS = False
Use_ADNI=False
Use_HCP=True

#setting latent variable sizes
latent_dim = 5000

if Use_HCP==True:
    #'fod' or 'dwi'
    trainset = HCPdataset(imgtype='fod', lmax=l_max)
    train_loader = torch.utils.data.DataLoader(trainset,batch_size = BATCH_SIZE, shuffle=True,
                                               num_workers=workers)
    
    
def inf_train_gen(data_loader):
    while True:
        for _,images in enumerate(data_loader):
            yield images
            
G = Generator(noise = latent_dim,lmax=l_max)
CD = Code_Discriminator(code_size = latent_dim ,num_units = 4096)
D = Discriminator(channel=l_max*32*4, is_dis=True,lmax=l_max)
E = Discriminator(out_class = latent_dim,is_dis=False,lmax=l_max)

if gpu==True:
    G.cuda()
    D.cuda()
    CD.cuda()
    E.cuda()

g_optimizer = optim.Adam(G.parameters(), lr=0.0002)
d_optimizer = optim.Adam(D.parameters(), lr=0.0002)
e_optimizer = optim.Adam(E.parameters(), lr = 0.0002)
cd_optimizer = optim.Adam(CD.parameters(), lr = 0.0002)

def calc_gradient_penalty(model, x, x_gen, w=10):
    """WGAN-GP gradient penalty"""
    assert x.size()==x_gen.size(), "real and sampled sizes do not match"
    alpha_size = tuple((len(x), *(1,)*(x.dim()-1)))
    alpha_t = torch.cuda.FloatTensor if x.is_cuda else torch.Tensor
    alpha = alpha_t(*alpha_size).uniform_()
    x_hat = x.data*alpha + x_gen.data*(1-alpha)
    x_hat = Variable(x_hat, requires_grad=True)

    def eps_norm(x):
        x = x.view(len(x), -1)
        return (x*x+_eps).sum(-1).sqrt()
    def bi_penalty(x):
        return (x-1)**2

    grad_xhat = torch.autograd.grad(model(x_hat).sum(), x_hat, create_graph=True, only_inputs=True)[0]

    penalty = w*bi_penalty(eps_norm(grad_xhat)).mean()
    return penalty

real_y = Variable(torch.ones((BATCH_SIZE, 1)).cuda(non_blocking=True))
fake_y = Variable(torch.zeros((BATCH_SIZE, 1)).cuda(non_blocking=True))

criterion_bce = nn.BCELoss()
criterion_l1 = nn.L1Loss()
criterion_mse = nn.MSELoss()

start_time=datetime.now()
current_time=start_time
g_iter = 1
d_iter = 1
cd_iter =1
TOTAL_ITER = 200000
#torch.autograd.set_detect_anomaly(True)
gen_load = inf_train_gen(train_loader)
for iteration in range(TOTAL_ITER):
    for p in D.parameters():  
        p.requires_grad = False
    for p in CD.parameters():  
        p.requires_grad = False
    for p in E.parameters():  
        p.requires_grad = True
    for p in G.parameters():  
        p.requires_grad = True

    ###############################################
    # Train Encoder - Generator 
    ###############################################
    for iters in range(g_iter):
        G.zero_grad()
        E.zero_grad()
        real_images = gen_load.__next__()
        _batch_size = real_images.size(0)
        with torch.no_grad():
            real_images = Variable(real_images).cuda(non_blocking=True)
            z_rand = Variable(torch.randn((_batch_size,latent_dim))).cuda()
        z_hat = E(real_images).view(_batch_size,-1)
        x_hat = G(z_hat)
        x_rand = G(z_rand)
        c_loss = -CD(z_hat).mean()

        d_real_loss = D(x_hat).mean()
        d_fake_loss = D(x_rand).mean()
        d_loss = -d_fake_loss-d_real_loss
        l1_loss =10* criterion_l1(x_hat,real_images)
        loss1 = l1_loss + c_loss + d_loss
        if iters<g_iter-1:
            loss1.backward()
        else:
            loss1.backward(retain_graph=True)
        e_optimizer.step()
        g_optimizer.step()
        g_optimizer.step()
    
    ###############################################
    # Train D
    ###############################################
    for p in D.parameters():  
        p.requires_grad = True
    for p in CD.parameters():  
        p.requires_grad = False
    for p in E.parameters():  
        p.requires_grad = False
    for p in G.parameters():  
        p.requires_grad = False

    #for iters in range(d_iter):
    d_optimizer.zero_grad()
    real_images = gen_load.__next__()
    _batch_size = real_images.size(0)
    with torch.no_grad():
        z_rand = Variable(torch.randn((_batch_size,latent_dim))).cuda()
        real_images = Variable(real_images).cuda(non_blocking=True)
    z_hat = E(real_images).view(_batch_size,-1)
    x_hat = G(z_hat)
    x_rand = G(z_rand)
    x_loss2 = -2*D(real_images).mean()+D(x_hat).mean()+D(x_rand).mean()
    gradient_penalty_r = calc_gradient_penalty(D,real_images.data, x_rand.data)
    gradient_penalty_h = calc_gradient_penalty(D,real_images.data, x_hat.data)
    loss2 = x_loss2+gradient_penalty_r+gradient_penalty_h
    loss2.backward(retain_graph=True)
    d_optimizer.step()
    

    ###############################################
    # Train CD
    ###############################################
    for p in D.parameters():  
        p.requires_grad = False
    for p in CD.parameters():  
        p.requires_grad = True
    for p in E.parameters():  
        p.requires_grad = False
    for p in G.parameters():  
        p.requires_grad = False

    for iters in range(cd_iter):
        cd_optimizer.zero_grad()
        with torch.no_grad():
            z_rand = Variable(torch.randn((_batch_size,latent_dim))).cuda()
        gradient_penalty_cd = calc_gradient_penalty(CD,z_hat.data, z_rand.data)
        cc_loss = -CD(z_hat).mean()
        loss3 =-CD(z_rand).mean() - cc_loss + gradient_penalty_cd    
        loss3.backward(retain_graph=True)



        cd_optimizer.step()
        
    

    ##############################################
    #Visualization
    ##############################################
    dur_last=datetime.now()-current_time
    current_time = datetime.now()
    if (iteration) % 100 == 0:
        print('[{}/{}]'.format(iteration,TOTAL_ITER))
        print("Current time", current_time)
        print("Duration total: {}".format(current_time-start_time))
        print("Duration last iteration: {}".format(dur_last))
          
    if (iteration) % 2000 == 0:
        print('[{}/{}]'.format(iteration,TOTAL_ITER),
              'D: {:<8.3}'.format(loss2.data.cpu().numpy()), 
              'En_Ge: {:<8.3}'.format(loss1.data.cpu().numpy()),
              'Code: {:<8.3}'.format(loss3.data.cpu().numpy()),
              )
        feat = np.squeeze((0.5*real_images[0,0]+0.5).data.cpu().numpy())
        feat = nib.Nifti1Image(feat,affine = np.eye(4))
        plotting.plot_img(feat,title="X_Real")
        plotting.show()

        feat = np.squeeze((0.5*x_hat[0,0]+0.5).data.cpu().numpy())
        feat = nib.Nifti1Image(feat,affine = np.eye(4))
        plotting.plot_img(feat,title="X_DEC")
        plotting.show()

        feat = np.squeeze((0.5*x_rand[0,0]+0.5).data.cpu().numpy())
        feat = nib.Nifti1Image(feat,affine = np.eye(4))
        plotting.plot_img(feat,title="X_rand")
        plotting.show()

    ###############################################
    # Model Save
    ###############################################
    if (iteration+1)%10000 ==0: 
        torch.save(G.state_dict(),'./check/G_iter'+str(iteration+1)+'.pth')
        torch.save(D.state_dict(),'./check/D_iter'+str(iteration+1)+'.pth')
        torch.save(E.state_dict(),'./check/E_iter'+str(iteration+1)+'.pth')
        torch.save(CD.state_dict(),'./check/CD_iter'+str(iteration+1)+'.pth')
        
## generate and export image
# func = nib.load('/media/nas/HCP_S1200_3T/996782/T1w/Diffusion/wm.nii')
# z_rand = Variable(torch.randn((latent_dim))).cuda()
# x_rand=G(z_rand).cpu().detach().numpy()[0,:,:,:,:]
# x_rand = np.transpose(x_rand, (1,2,3,0))
# x_rand_zero=nib.Nifti1Image(x_rand[:,:,:,0], func.affine)
# x_rand = nib.Nifti1Image(x_rand, func.affine)
# nib.save(x_rand, 'Syn_FOD.nii.gz')
# nib.save(x_rand_zero, 'Syn_FOD_zero')

