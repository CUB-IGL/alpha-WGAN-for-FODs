#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 17:28:38 2022

@author: adt
"""

import numpy as np
import torch
import os
import sys
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
import nibabel as nib
from torch.utils.data.dataset import Dataset
from torch.utils.data import dataloader
from HCP_dataset import *
from Model_alphaWGAN import *
from datetime import datetime
import matplotlib.pyplot as plt
torch.cuda.empty_cache()

BATCH_SIZE=6
gpu = True
gen_T1=False
workers = 6

if len(sys.argv) > 1:
    n_vol  = int(sys.argv[1])
else:
    n_vol = 45

if gen_T1==True:
    n_vol+=1

#l_max	Required volumes
#2	6
#4	15
#6	28
#8	45

LAMBDA= 10
_eps = 1e-15
res=int(32)
learn_rate=0.0001

Use_BRATS=False
Use_ATLAS = False
Use_ADNI=False
Use_HCP=True

#setting latent variable sizes
if len(sys.argv) > 2:
    latent_dim = int(sys.argv[2])
else:
    latent_dim= int(100)

if latent_dim<1000:
    LD=str(latent_dim)
else:
    LD=str(int(latent_dim/1000))+'k'

if len(sys.argv)>3 :
    res=res*int(sys.argv[3])

if Use_HCP==True:
    #'fod' or 'dwi'
    trainset = HCPdataset(imgtype='fod', n_vol=n_vol,augmentation=True, res=res, gen_T1=gen_T1,z64=True)
    train_loader = torch.utils.data.DataLoader(trainset,batch_size = BATCH_SIZE, shuffle=True,
                                               num_workers=workers)
num_run=int(1)
while os.path.isdir('train_'+str(num_run))==True:
    num_run+=1
num_run=str(num_run)
os.system('mkdir train_'+num_run)
os.system('mkdir train_'+num_run+'/img')
os.system('mkdir train_'+num_run+'/checkpoints')
log = open('train_'+num_run+'/log.txt', 'w')
params = open('train_'+num_run+'/params.txt', 'w')

params.write('latent_dimension\t number volumes\t resolution\t learn rate\t lambda\t eps\n')
params.write(str(latent_dim)+'\t'+str(n_vol)+'\t'+str(res)+'\t'+str(learn_rate)+'\t'+str(LAMBDA)+'\t'+str(_eps))
params.close()

log.write('Epoche\t loss1\t l1_loss\t c_loss\t d_Loss\t loss2\t D_real\t D_enc\t D_gen\t grad_pen_r\t grad_pen_h\t loss3\t CD_enc\t CD_gen\t grad_pen_cd'+'\n')
log.close()
os.system('cp alpha_WGAN_train_HCP.py train_'+num_run+'/')
os.system('cp HCP_dataset.py train_'+num_run+'/')
os.system('cp Model_alphaWGAN.py train_'+num_run+'/')

def inf_train_gen(data_loader):
    while True:
        for _,images in enumerate(data_loader):
            yield images
            
G = Generator(noise = latent_dim,lmax=n_vol,channel=200)
CD = Code_Discriminator(code_size = latent_dim ,num_units = 4096)
D = Discriminator(channel=2000, is_dis=True, lmax=n_vol, res=res)
E = Discriminator(out_class = latent_dim, channel = 2000, is_dis=False, lmax=n_vol, res=res)

if gpu==True:
    G.cuda()
    D.cuda()
    CD.cuda()
    E.cuda()

g_optimizer = optim.Adam(G.parameters(), lr = learn_rate, betas=(0.5,0.999))
d_optimizer = optim.Adam(D.parameters(), lr = learn_rate, betas=(0.5,0.999))
e_optimizer = optim.Adam(E.parameters(), lr = learn_rate, betas=(0.5,0.999))
cd_optimizer = optim.Adam(CD.parameters(), lr = learn_rate, betas=(0.5,0.999))

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

criterion_l1 = nn.L1Loss()
criterion_mse = nn.MSELoss()

start_time=datetime.now()
current_time=start_time
g_iter = 1
d_iter = 4
cd_iter =1

TOTAL_ITER = 50000
#torch.autograd.set_detect_anomaly(True)
gen_load = inf_train_gen(train_loader)
#with torch.no_grad():
#   z_exp1 = Variable(torch.randn((latent_dim))).cuda()
#    z_exp2 = Variable(torch.randn((latent_dim))).cuda()
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
        if gen_T1==True:
            real_images[:,0,:,:,:]=real_images[:,0,:,:,:]*0.0002
        _batch_size = real_images.size(0)
        with torch.no_grad():
            real_images = Variable(real_images).cuda(non_blocking=True)
            z_rand = Variable(torch.randn((_batch_size,latent_dim))).cuda()
        z_hat = E(real_images).view(_batch_size,-1)
        x_hat = G(z_rand)
        x_rand = G(z_rand)
        c_loss =-CD(z_hat).mean()
        d_real_loss = D(x_hat).mean()
        d_fake_loss = D(x_rand).mean()
        d_loss = -d_fake_loss-d_real_loss
        l1_loss =50000 * criterion_l1(x_hat,real_images)
        loss1 = l1_loss + c_loss + d_loss
        if iters<g_iter-1:
            loss1.backward()
        else:
            loss1.backward()
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
#    if iteration > 10000:
#        d_iter = 6
#    if iteration > 20000:
#        d_iter = 8
    for iters in range(d_iter):
        d_optimizer.zero_grad()
        real_images = gen_load.__next__()
        if gen_T1==True:
            real_images[:,0,:,:,:]=real_images[:,0,:,:,:]*0.0002
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
        loss2.backward()
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
        loss3.backward()
        cd_optimizer.step()
        
    ##############################################
    #Visualization
    ##############################################
    dur_last=datetime.now()-current_time
    current_time = datetime.now()
    if (iteration) % 100 == 0:
        log = open('train_'+num_run+'/log.txt', 'a')
        log.write(str(iteration)+'\t'+str(loss1.item())+'\t'+str(l1_loss.item())+'\t'+str(c_loss.item())+'\t'+str(d_loss.item())+'\t'+str(loss2.item())+'\t'+str(D(real_images).mean().item())+'\t'+str(D(x_hat).mean().item())+'\t'+str(D(x_rand).mean().item())+'\t'+str(gradient_penalty_r.mean().item())+'\t'+str(gradient_penalty_h.item())+'\t'+str(loss3.item())+'\t'+str(CD(z_hat).mean().item())+'\t'+str(CD(z_rand).mean().item())+'\t'+str(gradient_penalty_cd.item())+'\n')
        log.close()
        print('[{}/{}]'.format(iteration,TOTAL_ITER)+' training '+str(num_run))
        print("Current time", current_time)
        print("Duration total: {}".format(current_time-start_time))
        print("Duration last iteration: {}".format(dur_last))

    if (iteration+1)%10000 ==0: 
        torch.save(D.state_dict(),'train_'+num_run+'/checkpoints/D_iter'+str(iteration+1)+'.pth')
        torch.save(E.state_dict(),'train_'+num_run+'/checkpoints/E_iter'+str(iteration+1)+'.pth')
        torch.save(CD.state_dict(),'train_'+num_run+'/checkpoints/CD_iter'+str(iteration+1)+'.pth')
    
    if (iteration+1)%200 ==0:
        torch.save(G.state_dict(),'train_'+num_run+'/checkpoints/G_iter'+str(iteration+1)+'.pth')

    if (iteration+1)%100 == 0:
        torch.save(G.state_dict(),'train_'+num_run+'/checkpoints/G.pth')
