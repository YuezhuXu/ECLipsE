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
            
            
            
            fname_pth = r'datasets\\random\\lyr'+str(lyr)+'n'+str(n)+'test'+str(rd+1)+'.pth'
            paras_torch = {key: torch.tensor(value) for key, value in paras_dct.items()}
            torch.save(paras_torch, fname_pth)
            
            
            trivial_uppers.append(trivial_upper)
              
            
            fname = r'datasets\\random\\lyr'+str(lyr)+'n'+str(n)+'test'+str(rd+1)+'.mat'
            data = {'weights': np.array(weights, dtype=object)}
            savemat(fname, data)
            
            # Random mins, maxes, means, and ranges for demonstration
            mins = np.ones(input_size)#np.random.uniform(-1, 1, input_size)
            maxes = np.ones(input_size)#np.random.uniform(1, 2, input_size)
            means = np.ones(input_size+1)#np.random.uniform(-1, 1, input_size + 1)
            ranges = np.ones(input_size+1)#np.random.uniform(0.5, 1.5, input_size + 1)

            # Save to a .nnet file
            file_path = 'datasets\\random\\lyr'+str(lyr)+'n'+str(n)+'test'+str(rd+1)+'.nnet'
            with open(file_path, 'w') as f:
                # Write header information
                f.write("// Neural Network file format\n")
                f.write(f"{lyr},{input_size},{output_size}\n")
                
                # Write layer sizes
                layer_sizes_str = ",".join(map(str, net_dims))
                f.write(layer_sizes_str + "\n")
                
                # Placeholder for unused line
                f.write("\n")
                
                # Write mins and maxes
                f.write(",".join(map(str, mins)) + "\n")
                f.write(",".join(map(str, maxes)) + "\n")
                
                # Write means and ranges
                f.write(",".join(map(str, means)) + "\n")
                f.write(",".join(map(str, ranges)) + "\n")
                
                # Write weights and biases
                for i in range(lyr):
                    # Write weights
                    for row in weights[i]:
                        f.write(",".join(map(str, row)) + ",\n")
                    
                    # Write biases
                    for bias in biases[i]:
                        f.write(str(bias) + ",\n")




