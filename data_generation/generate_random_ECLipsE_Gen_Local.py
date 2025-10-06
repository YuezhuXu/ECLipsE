import numpy as np
import os  
from scipy.io import savemat
import torch 

trivial_uppers = []
input_size = 5
output_size = 2
lyrs = [5, 10, 15, 20, 25]
neurons = [10, 20, 40, 60]
'''
lyrs = [15, 20, 25, 30, 35]
neurons = [30, 50, 70, 90]

lyrs = [30, 40, 50, 60, 70]
neurons = [60, 80, 100, 120]
'''
# Varyig radius
lyrs = [5,30,60]
neurons = [128]

# norm_ctrl = [0.8,2.5]
norm_ctrl =[2,2.5]
# [0.8,2.5] for small (relu) and large sets (elu) with rand 77,9
# [2,2.5] for varying radius with rand 77, 9, leakyrelu


for lyr in lyrs:
    for n in neurons:
    
        np.random.seed(n*77+lyr*9)
        # for exporitng .mat and .nnet
        weights = [] 
        biases = []
        paras_dct = {}  # for exporting .pth
        net_dims = [input_size]
        net_dims += [n]*(lyr-1)
        net_dims += [output_size]
       
        
        #norm_ctrl = norm_ctrl_gen#/np.sqrt(lyr) #(norm_ctrl_gen)**(20/(lyr))
        
        trivial_upper = 1;
        
        for i in range(1, len(net_dims)):
            
            norm_rand = np.random.uniform(norm_ctrl[0],norm_ctrl[1])   
            weight =  np.random.randn(net_dims[i], net_dims[i-1])
            weight = norm_rand * weight / np.linalg.norm(weight,2)
            weights.append(weight)
            paras_dct[str(2*(i-1))+".weight"] = weight
            
            bias = np.random.randn(net_dims[i])
            biases.append(bias)
            paras_dct[str(2*(i-1))+".bias"] = bias
            
            
            trivial_upper *= norm_rand
            
            W_cell = np.empty((1, len(weights)), dtype=object)
            b_cell = np.empty((1, len(biases)), dtype=object)
            for idx, (W, b) in enumerate(zip(weights, biases)):
                W_cell[0, idx] = W
                b_cell[0, idx] = b
        

        
        trivial_uppers.append(trivial_upper)
          
        
        fname = r'..\datasets_ECLipsE_Gen_Local\random\lyr'+str(lyr)+'n'+str(n)+'.mat'
        data = {'weights': W_cell, 'biases': b_cell}
        savemat(fname, data)
        
        
            



