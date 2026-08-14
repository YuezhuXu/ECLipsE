# -*- coding: utf-8 -*-


import numpy as np
import os  
from scipy.io import savemat
import torch 

trivial_uppers = []
input_size = 4
output_size = 1
lyrs = [2, 5, 10, 20, 30, 50, 75, 100]
#neurons = [20, 40, 60, 80, 100]
neurons = [80,100,120,140,160]
norm_ctrl = [0.4,1.8]



for lyr in lyrs:
    for n in neurons:
        for rd in range(10):
            np.random.seed(rd*77+n*7+lyr*13)
            # for exporitng .mat and .nnet
            weights = [] 
            biases = []
            paras_dct = {}  # for exporting .pth
            net_dims = [input_size]
            net_dims += [n]*(lyr-1)
            net_dims += [output_size]
           
            
            
            trivial_upper = 1;
            
            for i in range(1, len(net_dims)):
                norm_rand = np.random.uniform(norm_ctrl[0],norm_ctrl[1])   
                weight =  np.random.rand(net_dims[i], net_dims[i-1])
                weight = norm_rand * weight / np.linalg.norm(weight,2)
                weights.append(weight)
                paras_dct[str(2*(i-1))+".weight"] = weight
                
                bias = np.random.rand(net_dims[i])
                biases.append(bias)
                paras_dct[str(2*(i-1))+".bias"] = bias
                
                
                trivial_upper *= norm_rand
        
            
            trivial_uppers.append(trivial_upper)
              
            
            fname = r'datasets\\random\\lyr'+str(lyr)+'n'+str(n)+'test'+str(rd+1)+'.mat'
            data = {'weights': np.array(weights, dtype=object)}
            savemat(fname, data)
            