import pandas as pd
import numpy as np
import math
import random
import sys 
import torch
import torch.nn as nn
import torch.nn.functional as F  # F.mse_loss F.softmax
import torch.optim as optim #optim.sgd
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import TensorDataset, DataLoader  # for batch and split Xtrain Ytrain dataset
import scipy
import scipy.ndimage as nd
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.interpolate import griddata
from scipy import special
from scipy.stats import truncnorm
import scipy.stats as stats
import numba
from numba import jit, cuda
# to measure exec time
from timeit import default_timer as timer   

from locale import format
from dataclasses import dataclass, MISSING
from sklearn import preprocessing #preprocessing.normalize


import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec


from icecream import ic  # for debugging. print variable name
import pickle # for saving variable
import json # for saving variable in text file 


from PC_param import default_parameters_network # To get the parameters
pars = default_parameters_network()




def connectivity_matrix_2x2(Dic_W, N_image, sparce = "_", distance_diag = "_", rand = np.random.randint(0,999)):
    
    cmap1 = 'viridis'
    #cmap = 'plasma'
    cmap2 = 'magma'
    """    W_t = {}
    for name, par in model_T.named_parameters() : # enumerate(trained_model.parameters()):
        W_t[name] = par #getattr(trained_model, "wee)
        if name.startswith("J"):
            W_t.popitem()
            break"""

    W_list = list(Dic_W.items())
    wei_name = W_list[0][0]
    wei_value = W_list[0][1]
    wie_name = W_list[1][0]
    wie_value = W_list[1][1]
    num_fig_st = str(rand)# np.random.randint(0,999))

    fig_m = plt.figure()
    gs = gridspec.GridSpec(1, 2)

    ax_wei = plt.subplot(gs[0,0])
    ax0_wei = ax_wei.matshow(wei_value.detach().numpy() , cmap = cmap1)
                # Adds subplot 'ax' in grid 'gs' at position [x,y]
    ax_wei.set_ylabel(wei_name)
    fig_m.colorbar(ax0_wei,fraction=0.046, pad=0.04, orientation='horizontal').ax.tick_params(labelsize=10)
    fig_m.add_subplot(ax_wei)
    
    ax_wie = plt.subplot(gs[0,1])
    ax0_wie = ax_wie.matshow(wie_value.detach().numpy() , cmap = cmap2)
                # Adds subplot 'ax' in grid 'gs' at position [x,y]
    ax_wie.set_ylabel(wie_name)
    fig_m.colorbar(ax0_wie,fraction=0.046, pad=0.04, orientation='horizontal').ax.tick_params(labelsize=10)
    fig_m.add_subplot(ax_wie)
    
    #fig_m.colorbar(caxes_wei)
    fig_m.suptitle(f'Blend of {N_image} connectivity matrices obtained through averaging',  y= 0.83)#, fontsize=16
    fig_m.savefig(f'C:/Users/knzga/Documents/02_Computational Neuroscience Project/Image/ConnectivityMatrix/Blended_CM/{N_image}Blended_connectivity_matrix_0.002_650_0.001_dd3_{num_fig_st:0>3}.png')
    fig_m.tight_layout()
    fig_m.show()
    
def get_blended_tensor(loaded_images):
    #stack the list for file into 1 tensor
    combined_wie = torch.stack(loaded_images)
    # Compute mean along 7 axis
    mean_tensor = torch.mean(combined_wie, dim = 0)
    return mean_tensor






# Load the variable for wee and wii (eye matrix but a bit blurry : more sparse )
"""file_name = "wee_wii_optimal.pkl"
with open(file_name, "rb") as file:
    loaded_wee_wii = pickle.load(file)"""

# How to load the parameter file, Open the file for reading
"""with open("lagrandian_multiplier_417.txt", "r") as fp:
    # Load the dictionary from the file
    lm_dict = json.load(fp)

print(lm_dict) #['normalization']
"""
file_name = []

file_name.append("WIE_sp0.001_dd3.0_norm6000_247.pkl")
file_name.append("WIE_sp0.001_dd3.0_norm6000_332.pkl")
file_name.append("WIE_sp0.001_dd3.0_norm6000_826.pkl")
file_name.append("WIE_sp0.001_dd3.0_norm6000_338.pkl")
file_name.append("WIE_sp0.001_dd3.0_norm6000_175.pkl")
file_name.append("WIE_sp0.001_dd3.0_norm6000_516.pkl")
file_name.append("WIE_sp0.001_dd3.0_norm6000_74.pkl")
file_name.append("WIE_sp0.001_dd3.0_norm6000_456.pkl")
file_name.append("WIE_sp0.001_dd3.0_norm7000_933.pkl")
file_name.append("WIE_sp0.001_dd3.0_norm7000_563.pkl")
loaded_wie = [0]*len(file_name)
for i in range(len(file_name)):
    with open(file_name[i], "rb") as file:
        loaded_wie[i] = pickle.load(file)


file_name2 = []

file_name2.append("WEI_sp0.001_dd3.0_norm7000_563.pkl")
file_name2.append("WEI_sp0.001_dd3.0_norm6000_456.pkl")
file_name2.append("WEI_sp0.001_dd3.0_norm7000_933.pkl")
file_name2.append("WEI_sp0.001_dd3.0_norm6000_247.pkl")
file_name2.append("WEI_sp0.001_dd3.0_norm6000_332.pkl")
file_name2.append("WEI_sp0.001_dd3.0_norm6000_826.pkl")
file_name2.append("WEI_sp0.001_dd3.0_norm6000_338.pkl")
file_name2.append("WEI_sp0.001_dd3.0_norm6000_175.pkl")
file_name2.append("WEI_sp0.001_dd3.0_norm6000_516.pkl")
file_name2.append("WEI_sp0.001_dd3.0_norm6000_74.pkl")

loaded_wei = [0]*len(file_name2)
for i in range(len(file_name2)):
    with open(file_name2[i], "rb") as file:
        loaded_wei[i] = pickle.load(file)

#stack the list for file into 1 tensor
"""combined_wie = torch.stack(loaded_wie)

# Compute mean along 7 axis
mean_tensor = torch.mean(combined_wie, dim = 0)
plt.matshow(mean_tensor.detach().numpy(), cmap= cmap2)
plt.show()"""

wie_T = get_blended_tensor(loaded_wie)
wei_T = get_blended_tensor(loaded_wei)

Dic_W = {'wei': wei_T, 'wie':wie_T}
N_image = len(loaded_wie)

connectivity_matrix_2x2(Dic_W, N_image)
plt.show()
   