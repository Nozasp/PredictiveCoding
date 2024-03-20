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


""" Plot HeatMap of firing rate function"""
def HeatMap(rE, rI, J=None, toshow = True):
    if J == None:
        J = [.00989, 0.0081, .1, .87, .00081]  # J = dict(Jin=.008, Jee= .2, Jie=.2, Jei=1.4, Jii=6.7)
    if type(J) == dict:
        J = np.round(np.array(list(J.values())), 4)
        

    rE_df = pd.DataFrame(rE.T)  # to get time vs pop
    rI_df = pd.DataFrame(rI.T)
    rE_df.index = rE_df.index + 1
    rI_df.index = rI_df.index + 1
    rE_df.index.name, rI_df.index.name = ["Excitatory Population", "Inhibitory Population"]
    rE_df.columns.name, rI_df.columns.name = ["Time ms", "Time ms"]
    # print(rE_df.loc[[10]])

    # set context for the upcoming plot
    sns.set_context("notebook", font_scale=.8, rc={"lines.linewidth": 2.5, 'font.family': 'Helvetica'})

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(6, 6))

    sns.heatmap(rE_df, ax=axA, cmap="viridis")
    sns.heatmap(rI_df, ax=axB)
    ##J2= {'Jee':  'Jei': , 'Jie': mpy(), 'Jii': mo), 'Jin': umpy()}
    axA.set_title(f"Firing rate in Hz of exc populations over time. Jie: {J[2]}, Jee: {J[0]}, Jin: {J[4]}",
                  fontdict={"fontsize": 10})
    axB.set_title(f"Firing rate in Hz of inh populations over time. Jei: {J[1]}, Jii: {J[3]}",
                  fontdict={"fontsize": 10})
    plt.tight_layout()

    if toshow:
        plt.show()

    else: 
        return fig, (axA, axB)
    #num_fig = random.randint(0, 1000)
    #fig.savefig(f'C:/Users/knzga/Documents/02_Computational Neuroscience Project/Image/OutputNetwork/Heatmap/Heatmap_{num_fig:0>3}.png')

def connectivity_matrix_toBlend(model_T, Learning_Rate, n_epoch, sparce = "_", distance_diag = "_", rand = np.random.randint(0,999)):
    cmap1 = 'viridis'
    #cmap = 'plasma'
    cmap2 = 'magma'
    W_t = {}
    for name, par in model_T.named_parameters() : # enumerate(trained_model.parameters()):
        W_t[name] = par #getattr(trained_model, "wee)
        if name.startswith("J"):
            W_t.popitem()
            break

    W_list = list(W_t.items())
    wei_name = W_list[0][0]
    wei_value = W_list[0][1]
    wie_name = W_list[1][0]
    wie_value = W_list[1][1]
    num_fig_st = str(rand)

    fig_wei = plt.figure()
    axes_wei = fig_wei.add_subplot(111)    
    # using the matshow() function 
    caxes_wei = axes_wei.matshow(wei_value.detach().numpy(), cmap = cmap1)#interpolation ='nearest')
    fig_wei.colorbar(caxes_wei)
    fig_wei.savefig(f'C:/Users/knzga/Documents/02_Computational Neuroscience Project/Image/ConnectivityMatrix/connectivity_matrix_{Learning_Rate}_{n_epoch}_AfterTraining_{sparce}_{distance_diag}_{wei_name}_{num_fig_st:0>3}.png')
    fig_wei.tight_layout()
    fig_wei.show()
    
    
    fig_wie = plt.figure()
    axes_wie = fig_wie.add_subplot(111)
    caxes_wie = axes_wie.matshow(wie_value.detach().numpy(), cmap = cmap2)#interpolation ='nearest')
    fig_wie.colorbar(caxes_wie)
    # Adds subplot 'ax' in grid 'gs' at position [x,y]
    fig_wie.savefig(f'C:/Users/knzga/Documents/02_Computational Neuroscience Project/Image/ConnectivityMatrix/connectivity_matrix_{Learning_Rate}_{n_epoch}_AfterTraining_{sparce}_{distance_diag}_{wie_name}_{num_fig_st:0>3}.png')
    fig_wie.tight_layout()
    fig_wie.show()
    
    #plt.show()

def connectivity_matrix_2x2(model_T, Learning_Rate, n_epoch, sparce = "_", distance_diag = "_", rand = np.random.randint(0,999)):
    cmap1 = 'viridis'
    #cmap = 'plasma'
    cmap2 = 'magma'
    W_t = {}
    for name, par in model_T.named_parameters() : # enumerate(trained_model.parameters()):
        W_t[name] = par #getattr(trained_model, "wee)
        if name.startswith("J"):
            W_t.popitem()
            break

    W_list = list(W_t.items())
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
    fig_m.suptitle(f'Connectivity matrix obtained with LR = {Learning_Rate}, Epochs = {n_epoch} \n , discouraging term ={distance_diag}, sparsity = {sparce}',  y= 0.83)#, fontsize=16
    fig_m.savefig(f'C:/Users/knzga/Documents/02_Computational Neuroscience Project/Image/ConnectivityMatrix/connectivity_matrix_{Learning_Rate}_{n_epoch}_AfterTraining_{sparce}_{distance_diag}_{num_fig_st:0>3}.png')
    fig_m.tight_layout()
    fig_m.show()
    
    
   
    
    #plt.show()

# Make a dictionary a list if I want to access index! dictionary has no indexing (storage with keys)

def plot_connectivity_matrix(model_T, num_fig, Learning_Rate, n_epoch, limit = 2, sparce = "_", distance_diag = "_"):
    labelsize = 10
    subtitle_size = 11
    #cmap = 'viridis'
    #cmap = 'plasma'
    cmap = 'magma'

    W_t = {}
    for name, par in model_T.named_parameters() : # enumerate(trained_model.parameters()):
        W_t[name] = par #getattr(trained_model, "wee)
        if name.startswith("J"):
            W_t.popitem()
            break
        
    #print(len(W_t))

    column = 0
    row = 1
    for i in range(len(W_t)):
        column += 1
        if column == limit +1:
            row +=1
            column = 0 

    if len(W_t) >=limit:
        column = limit
    else:
        column = len(W_t)
    
    ic(row, column)
    fig_m = plt.figure()
    # create figure window

    gs = gridspec.GridSpec(row, column)
    # Creates grid 'gs' of a rows and b columns

    """for r in range(row):
            for c in range(column):
                print(r, c)
    """
    ic(W_t.keys)
    W_list = list(W_t.items())
    
    if row > 0:
        index = 0
        for r in range(row):
            for c in range(column):
                name = W_list[index][0]
                w = W_list[index][1]
                ic(r, c, name)
                ax = plt.subplot(gs[r,c])
                ax0 = ax.matshow(w.detach().numpy() , cmap = cmap)
                # Adds subplot 'ax' in grid 'gs' at position [x,y]
                ax.set_ylabel(name)
                #ax.set_title(f"{name} trained", size = subtitle_size)
                #ax.figure.axes[index].tick_params(axis="both", labelsize= labelsize) 
                
                fig_m.colorbar(ax0,fraction=0.046, pad=0.04, orientation='horizontal').ax.tick_params(labelsize=10)
                fig_m.add_subplot(ax)

                index +=1
                if index == len(W_t):
                    break
                else:
                    continue
                
    num_fig_st = str(num_fig)
    if row == 1:
        fig_m.suptitle(f'Connectivity matrix obtained with LR = {Learning_Rate}, Epochs = {n_epoch}',  y= 0.83)#, fontsize=16
    else: 
        fig_m.suptitle(f'Connectivity matrix obtained with LR = {Learning_Rate}, Epochs = {n_epoch}')#, fontsize=16
    fig_m.savefig(f'C:/Users/knzga/Documents/02_Computational Neuroscience Project/Image/ConnectivityMatrix/connectivity_matrix_{Learning_Rate}_{n_epoch}_AfterTraining_{sparce}_{distance_diag}_{num_fig_st:0>3}.png')
    fig_m.tight_layout()
    fig_m.show()
    #plt.show()




""" Filters gauss and Dog and LoG"""
def gaussian_filter(s, N):
    pop = np.arange(1, N + 1)
    n = 1 / (np.sqrt(2 * np.pi) * N * s)
    gaussW = n * np.exp(-(pop - pop[:, np.newaxis]) ** 2 / (2 * s ** 2))
    gaussW2 = gaussW / np.max(gaussW) # gaussW /(.009 ** 2 / np.max(gaussW))  # 1
    return gaussW2

def dog_filter(sOut, N):
    sIn = sOut / 30
    pop = np.arange(1, N + 1)
    gaussIn = np.exp(-(pop - pop[:, np.newaxis]) ** 2 / (2 * sIn ** 2))
    gaussOut = np.exp(-(pop - pop[:, np.newaxis]) ** 2 / (2 * sOut ** 2))
    dog = gaussOut - gaussIn
    if np.max(dog) == 0 or None:
        print('zero max')
        dog = 0
    else:
        dog = dog / ( np.max(dog)) ###dog /.042 ** 2 /( np.max(dog))### # .0088
    return dog

def LoG_filter(s, N):
    x_lap = np.eye(N)
    lapl_filter = nd.gaussian_laplace(x_lap, sigma=(s, s))
    return lapl_filter

def dLogGaus(s=.61, N=20):
    dig = LoG_filter(s, N) + gaussian_filter(.019 * s, N)
    return dig



def truncated_normal(N, mu=0., sigma= .1):
    sigma = 1/np.sqrt(N)
    lower, upper = 0., np.inf
    pop = np.arange(1, N + 1)
    
    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    #x = np.linspace(0, size, size)

    return X.pdf(pop- pop[:, np.newaxis])#X.rvs(N)

def init_random_matrix(N, mu = 0.):

    lower, upper = 0., np.inf
    sigma = 1/np.sqrt(N)#random.randint(1, 30), random.uniform(0.1, 5)
    X = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    rnd_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            rnd_matrix[i][j] = X.rvs(1)# np.random.uniform(low=0.0, high=1.0, size=1) #X.rvs(1)
    return rnd_matrix 

def init_random_Tgaussian(N, mu =0., sigma = .025, lower = 0., upper = np.inf): #this simga! standard deviation not sigma2
    #lower, upper = 0., np.inf
    #sigma = 1/np.sqrt(N)                                                                  #random.randint(1, 30), random.uniform(0.1, 5)
    X = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

    rnd_tensor = X.rvs(N) 

    return torch.tensor(rnd_tensor)



def make_it_proba(r_e, offset = 1):
    eps = torch.FloatTensor([sys.float_info.epsilon]) #torch.FloatTensor
    #offset = 0.5 #eps #1 # 5
    sigma = 0.025#works well 0.025 #works0.1 #0.01 #0.025
    noisy_term = torch.tensor(init_random_Tgaussian(r_e.shape[1], mu = 1, sigma = sigma, lower = 1))#offset +  #, dtype = torch.float32
    unormalised_prob = r_e + (noisy_term)
    prob_r = unormalised_prob / (unormalised_prob).sum(1, keepdims = True)
    return prob_r 


def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x), axis=1, keepdims=True)

''' Classic Normalization
use:
#preprocessing.normalize(re_numpy, axis= 0)[20,:].sum()
or use:
'''
def normalize(x):
    N = (x - x.min()) / (x.max() - x.min())
    return N


"""
Differentiable function for back propagation

To avoid non-differentiable araising from discontinuity of the function, I "relax" (smoothen) the where() expression by using a sigmoid instead
*   with grad_fn:
*   if I get : > <SumBackward1 object at 0x7f79da0b9520> # differentiable
*   else I get none
"""
def relu_stim(x, stim):
    return torch.nn.functional.relu(1.0 - torch.abs(x - stim),
                                    inplace=False)  # inplace = False to avoid implace operation

def Dirac(A, N=pars["NumN"]):
    y = scipy.signal.unit_impulse(N, idx=(torch.max(torch.argmax(A))))  
    return torch.tensor(y)

def replace_argmax(r):
    # along some dimension (e.g., the last dimension).
    indices = torch.arange(r.shape[-1]).to(r.device)
    return torch.gather(indices, dim=-1, index=torch.argmax(r, dim=-1)).max()



""" Target design: Get the expected stimuli and then create a matrix of 1 where stimuli 0 elsewhere"""

def get_stimuli_input(X_train_tensor):  # input of the shape Xtrain_tensor[5,:,:]
    Xargmax = torch.argmax(X_train_tensor, dim=1) #consider replacing argmax by replace_argmax
    Xmax = torch.max(Xargmax)
    return Xmax

def get_expected_Y_relu(X_train_tensor):
    x_t = torch.transpose(X_train_tensor, 0, 1)
    dirac_2d = torch.zeros(x_t.shape)
    stim = get_stimuli_input(
        X_train_tensor)  # input of the shape Xtrain_tensor[5,:,:] # here get_stimuli not differenciable
    
    for pop, t in enumerate(x_t):
        tpop = torch.tensor(pop)# replace where function by relu functio which is differentiable
        dirac_2d[pop, :] = torch.nn.functional.relu(1.0 - torch.abs(tpop - stim), inplace=False).requires_grad_(False)
    dirac_2d = torch.transpose(dirac_2d, 1, 0)
    return dirac_2d



""" 
Optimization function
Make a function which save parameters of trained model and upload the new model with the updated parameters
"""
def model_with_saved_trained_param(old_model, optimizer, Model, param, sim, dicJ):
    # or to save the parameters only
    torch.save(old_model.state_dict(),"Old_model_optimized_parameters.pth")
    torch.save(optimizer.state_dict(),"optimizer_optimized_parameters.pth")
    #load these parameters in a new model instance
    new_mymodel = Model(param, sim, dicJ)
    new_mymodel.load_state_dict(torch.load("Old_model_optimized_parameters.pth")) 
    optimizer.load_state_dict(torch.load('optimizer_optimized_parameters.pth'))

    #print(optimizer.param_groups[0]['params'])
    if old_model.Jee == new_mymodel.Jee:
        print("it works")
    print("old model Jee:",old_model.Jee,"new model Jee:", new_mymodel.Jee)
    #print(optimizer.param_groups)
    #print(optimizer.state)
    return new_mymodel, optimizer


def load_weights(newmodel, modelpath): #string
        if '.pt' not in modelpath:
            modelpath += '.pt'      
        newmodel.load_state_dict(torch.load(modelpath))
        #new_mymodel = Model(param, sim, model.state_dict())
        return newmodel #, newmodel.state_dict() to access the param
    
def save_weights(oldmodel, modelpath, epoch=None):  #string
    if '.pt' not in modelpath:
            modelpath += '.pt'
    torch.save(oldmodel.state_dict(), modelpath)  


"""
Model evaluation
function wich test the accuracy of a model with new parameters compared to expected results + loss values for every samples
"""
def test_model(model, test_dataloader, loss_f):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for x_test, y_test in test_dataloader:
            # Calculate output
            Y_prediction, _, dredt, dridt = model(x_test[0])
            #Y_prediction_prob = make_it_proba(Y_prediction)

            # Calculate loss
            #loss = loss_f(Y_prediction_prob, y_test[0], dredt, dridt)
            loss = loss_f(Y_prediction, y_test[0], dredt, dridt)

            # Accuracy
            predictions = Y_prediction.detach().round() # rounds the predictions to the nearest integer (0 or 1), assuming they represent probabilities.
            #predictions = Y_prediction_prob.detach().round() # rounds the predictions to the nearest integer (0 or 1), assuming they represent probabilities.
            correct_predictions += (predictions == y_test[0]).sum().item() # calculates the number of correct predictions by comparing the rounded predictions with the true labels (y_test). It sums up the correct predictions over the batch.
            total_samples += y_test[0].numel() # adds the total number of samples/item in the current batch to the overall count.

            test_loss += loss.item()

    accuracy = correct_predictions / total_samples
    average_loss = test_loss / len(test_dataloader)

    print(f'Test Accuracy: {accuracy:.4f}, Test Loss: {average_loss:.4f}')
    return accuracy, average_loss


""" 
LOSS
"""
def easyLoss(Y_pred_prob,target):
    #loss = torch.sum(torch.sum((Y_prediction_prob-Y_target), axis =1))
    return torch.mean((Y_pred_prob - target)**2)

#for l2 reg with w params
def sum_relu(w):
      return torch.sum(F.relu(-w))


""" 
Run simulation over batchXtime
"""
def run_model_across_batch(Input, len_sim, model_instance):
    count = 0 
    P0 = torch.zeros_like(Input)  
    I0 = torch.zeros_like(Input)
    dPdt = torch.zeros_like(Input)  
    model_instance.initiate_state()

    for i in range(Input.shape[0]):
            count +=1
            P0[i,:], I0[i,:], dPdt[i,:], dridt, ampa, gaba = model_instance.forward(Input[int(i),:])#.item()
            if count == len_sim: #train_IN.shape[1]: #if we end the simulation time and go to the next batch
                count = 0
                model_instance.initiate_state()

    return P0, I0, dPdt

""" 
Plot Normalised predictions
"""
def plot_normalized_plot(P0, Ptrained, t, legend = None): #legend = list of 2 strings 
    P0_np = np.array(P0.detach().numpy())
    PT_np = np.array(Ptrained.detach().numpy())

    P0_norm = preprocessing.normalize(P0_np, axis= 1) 
    PT_norm = preprocessing.normalize(PT_np, axis= 1) 
    if legend is None:
        plt.plot(np.arange(P0_norm.shape[1]), P0_norm[t,:], label = f"{t}ms, untrained")
        plt.plot(np.arange(PT_norm.shape[1]), PT_norm[t,:], label = f"{t}ms, trained")
    else:
        plt.plot(np.arange(P0_norm.shape[1]), P0_norm[t,:], label = f"{t}ms," + legend[0])
        plt.plot(np.arange(PT_norm.shape[1]), PT_norm[t,:], label = f"{t}ms," + legend[1])

""""
Compute level of sparcity
"""
def measure_sparcity(A):
    return 1.0 - ( np.count_nonzero(A) / float(A.size) )

""""
Make a logistic matrix for penalizing long term connections in inh pop
"""
def logistic_func_dd(N, k= 1):#2#.35 # goo1): #.35): #.35): or 1 and 8
  L = 1# 8*N #5*N
  x_diff = np.abs(np.arange(N)[np.newaxis, :] - np.arange(N)[:, np.newaxis])
  return L/(1+ np.exp(-k*(x_diff-4)))#4 good good !#2#3good#4 good #5 good #6 #8))) #(12))))#N/3


 
# ***************** CLASS ***************************************

@dataclass
class Parameter:
    # °°° Load the parameters °°°
    taue: float = pars["taue"]
    ae: float = pars['ae']
    be, hme, I_noise = pars['be'], pars['hme'], pars['I_noise']
    Jee: float = pars['Jee']
    taui, ai, bi, hmi = pars['taui'], pars['ai'], pars['bi'], pars['hmi']
    tauNMDA, tauAMPA, tauGABA = pars['tauNMDA'], pars['tauAMPA'], pars['tauGABA']
    gamma: float = pars['gamma']  # nmda coupling parameter from brunel
    c_dash = pars['c_dash']
    sigma = pars['sigma']  # param.sigma = .0007 for Noise
    I_noise = pars['sigma'] * np.random.randn(3, 1)
    I1 = pars['Jext'] * pars['mu0'] * (1 + pars['c_dash'] / 100)
    I2 = pars['Jext'] * pars['mu0'] * (1 - pars['c_dash'] / 100)
    # I1, I2 = pars['I1'], pars['I2']
    sigmaIn = pars['sigmaIn']

    # Input parameters
    In0 = pars['In0']  # % Spontaneous firing rate of input populations (Hz)
    InMax = pars['InMax']  # % Max firing rate of input populations (Hz)
    Iq0 = pars['Iq0']  # % Spontaneous firing rate of feedback populations (Hz)
    IqMax = pars['IqMax']  # % Max firing rate of feedback populations (Hz)

    # Gaussian filter
    # sIn = pars['sigmaInh'][0]
    # sOut = pars['sigmaInh'][1]

    #°°° Hard encode these parameters °°°
    Jee: float = pars['Jee']#0.2609
    Jii: float = pars['Jii']
    Jei: float = pars['Jei']
    Jie: float = pars['Jie']
    #Jes, Jsi = pars['Jes'], pars['Jsi']
    #Jiq: float = pars['Jiq']  # 0.85; #nA
    Jin: float = pars['Jin']
    #N=20, sIn=.1, sOut=3., sEI=.2

    def __init__(self, sEI=.2, sIn=.1, sOut=3., N=40):  # sEI=4, sIn=.2, sOut=1.2,
        # Weights (from gaussian filter)
        self.N = N  # pars['NumN']
        self.wei = torch.tensor(dog_filter(sOut, int(N)), dtype=torch.float32)   # .astype( torch.float32))  # , dtype='float64'# fun.dLogGaus(.61, N)  #fun.dog_filter(sIn, sOut, N)#gaussian_filter(sEI, N)
        self.wii = torch.tensor(np.eye(int(N)), dtype=torch.float32) #.astype(torch.float32))  # dog_filter(sIn, sOut, N)#np.eye(N) #
        self.wie = torch.tensor(gaussian_filter(sEI, int(N)), dtype=torch.float32) #.astype(torch.float32))  # dog_filter(sIn, sOut, N)
        self.wes = torch.tensor(np.eye(int(N)), dtype=torch.float32)  #.astype(torch.float32))  # Identity matrix
        self.f = np.arange(1, N + 1)
        self.sEI = sEI
        self.sIn = sIn
        self.sOut = sOut

    def reset(self):  # https://stackoverflow.com/questions/56878667/setting-default-values-in-a-class

        for name, field in self.__dataclass_fields__.items():
            if field.default != MISSING:
                setattr(self, name, field.default)
            else:
                setattr(self, name, field.default_factory())


# °°° Time of the simulation °°°
class Simulation:
    def __init__(self, dt, T):
        self.dt = dt
        self.T = T
        self.range_t = (np.arange(0, self.T, self.dt))
        ic(self.range_t.shape)
        self.Lt = self.range_t.size - 1

    def printSim(self):
        print("Time step of the simulation (dt):", self.dt, "  Duration of simulation (T):", self.T,"s",
              "  Length of the time frame (Lt):", self.Lt)


#  °°° Initialisation of the variables °°°
class Stim:
    def __init__(self, param, simu, f, ISI=0, dur=0.05, mul = 0):#ISI=0.5, dur=0.2): #ISI=1, dur=0.2   # 8 #[10]
        self.f = f  # array of frequency stimulus types
        self.ISI = ISI  # inter-stimulus interval
        self.dur = dur  # duration in s of a specific stimulus segment . The time the frequency fi ll be maintained in the f array
        self.tail = 0.05
        self.mul = mul
        self.predDt = 0
        self.pred = 0
        self.InMax = param.InMax
        self.In0 = param.In0
        #ic(self.InMax, self.In0)
        self.N = param.N

        # Instantaneous frequency
        #f_instant = np.zeros((int(self.ISI / simu.dt) + 1, 1))  # size ISI : 1 /dt : 1000

        for fx in self.f: #+2 to not lose dimension ! becareful! 
            fx_array = np.ones((int(self.dur / simu.dt), 1)) * fx 
            #np.concatenate((np.ones((int(self.dur / simu.dt), 1)) * fx, #
                                       # just 1 frequency of 8 . # inter-stim interval is aslong as stim interval
                                       #np.zeros((int(self.ISI / simu.dt),
                                        #         1))))  # so I get 1 list with 1000 lists containing 8 and 1000 lists containing 0
    
        self.f_stim = fx_array # np.vstack((f_instant, fx_array))  # stack vertically these arrays # [0] *1000 , [8]*1000, [0]*1000
        #print(len(f_stim))
        #self.f_stim = f_stim[1:]  # 1400*1
        #print(len(self.f_stim))
      
    # bottom up sensory Input # duration 1sec
    def sensoryInput(self, parameter, simu, sigmaIn=None, f_stim=None, InMax=None, In0=None):
        paramf = np.arange(0, self.N)#np.arange(1, self.N+1)
        
        """y= np.asarray(paramf) - (self.f_stim)
        ok = plt.matshow(y)
        plt.colorbar(ok)
        plt.show()"""
        #mask the ground truth f_stim by a gaussian function
        w = np.exp(-(((paramf) - (self.f_stim)) ** 2) / (
                2 * (sigmaIn or parameter.sigmaIn) ** 2))  # pars['f'] = 1:N
        #ic((w).shape)
        #if I want to normalize w:
        # totalAct = w.sum(axis = 1) #sum over each row
        # norm_w = (w.T / totalAct).T # elementwise division
        
        """In = np.where(f_stim or self.f_stim > 0, (InMax or self.InMax) * w + (In0 or self.In0),
                        0)  # if stim >0 give InMax * weight + In0 otherwise give 0
        """
        In = (InMax or self.InMax) * w + (In0 or self.In0)
        if self.tail != 0:
            tail_zeros = np.zeros((parameter.N , int(self.tail / simu.dt))).transpose() #permute(1,0)
            In = np.concatenate((In, tail_zeros), axis = 0) #np.hstack((In, tail_zeros))
            #ic(In.shape)
            if self.mul != 0:
                repeatIN = np.tile(In, (self.mul, 1)) #np.stack([In for _ in range(self.mul)], axis=1)
                In = np.vstack(repeatIN)
                #ic(In.shape)
        range_sim = np.arange(0, In.shape[0]) #
        len_sim = len(range_sim)
        self.In = In
        self.w = w
        self.sigmaIn = sigmaIn
        #ic(len(range_sim), In.shape )
        return self.In , range_sim, len_sim ,w, sigmaIn


    def printStim(self):
        print("frequence of stimulus f:", self.f, "  ISI:", self.ISI,"s","  Size In", self.In.shape, "Size w:",
              self.w.shape, "  f_stim = total length simulation:", self.f_stim.shape,
              "sigmaIn:", self.sigmaIn)


class Model(nn.Module):
    def __init__(self, param, sim, dicJ, In):
        super(Model, self).__init__()

        #--- Define other model parameters, layers, or components here if needed
        self.dt = sim.dt #torch.tensor(1e-4) #
        self.N = In.shape[1]#param.N - 6 #20
        self.taue = self.taui = torch.tensor(param.taue) #torch.tensor(0.005)
         # ¤ parameter of the phi function Not tweakable parameters
        self.ae = torch.tensor(param.ae)# 18.26)  # 2 #Wong have to check # Modelling and Meg Gain of the E populaiton
        self.be = torch.tensor(param.be) #-5.38)  # Threshold of the E populaiton
        self.hme = torch.tensor(param.hme)#78.67)
        self.ai = torch.tensor(param.ai)#21.97)
        self.bi = torch.tensor(param.bi)#-4.81)
        self.hmi = torch.tensor(param.hmi)#125.62)
        #create the smallest possible number
        self.epsilon = sys.float_info.epsilon

        self.sIn = torch.tensor(.1)
        self.sOut= 3.
        self.sEI = .2
        self.tauAMPA = torch.tensor(0.002)
        self.tauGABA = torch.tensor(0.005)


        """with open(file_name, "rb") as file:
            loaded_wee_wii = pickle.load(file)"""
        #self.Inoise = np.random.randn()
        #self.I_noise_E = torch.randn(size= (self.N,), dtype=torch.float32) #* 0.02 # if I want to change sigma #.astype(torch.float32)#(.2, .2, 1) #(0, .02, 1)
        #self.I_noise_E = torch.tensor(init_random_Tgaussian(self.N, mu =0., sigma = 1/np.sqrt(self.N)), dtype = torch.float32) #sigma = 1/np.sqrt(N)  
        self.Se_noise = torch.tensor(init_random_Tgaussian(self.N, mu =0., sigma = 1/np.sqrt(self.N)), requires_grad = False, dtype = torch.float32)#sigma = 1/np.sqrt(N)  
        self.Si_noise = torch.tensor(init_random_Tgaussian(self.N, mu =0., sigma = 1/np.sqrt(self.N)), requires_grad = False,dtype = torch.float32)#sigma = 1/np.sqrt(N)  

        #self.I_noise_I = torch.randn(0, .02, 1)

        self.wii =  torch.tensor(np.eye(int(self.N)), dtype=torch.float32) # self.wii =torch.tensor(loaded_wee_wii.detach().numpy() , dtype=torch.float32) #!!! N = 30  ## dog_filter(sIn, sOut, N)#np.eye(N) #
        self.wee = torch.tensor(np.eye(int(self.N)), dtype=torch.float32) #self.wee = torch.tensor(loaded_wee_wii.detach().numpy() , dtype=torch.float32) #

        #wee_o = init_random_matrix(N=self.N)
        #self.wee = nn.Parameter(torch.tensor(wee_o, requires_grad = True, dtype = torch.float32)) 
        #self.wii = nn.Parameter(torch.tensor(wii_o, requires_grad = True, dtype = torch.float32)) 

        # Example usage for w0 initialization 
        wie_o = init_random_matrix(N=self.N)
        wei_o = init_random_matrix(N=self.N)
        
        self.wes = torch.tensor(np.eye(int(self.N)), dtype=torch.float32)  # Identity matrix


        # initial parameters
        self.wei = nn.Parameter(torch.tensor(wei_o, requires_grad = True, dtype = torch.float32))  
        self.wie = nn.Parameter(torch.tensor(wie_o, requires_grad = True, dtype = torch.float32))
        #self.wie = torch.tensor(gaussian_filter(self.sEI, int(self.N)), dtype=torch.float32) #.astype(torch.float32))  # dog_filter(sIn, sOut, N)
        #self.wei = torch.tensor(dog_filter(self.sOut, int(self.N)), dtype=torch.float32)         
        
        self.dicJ = dicJ #kwargs
        
        self.Jee = torch.tensor(self.dicJ['Jee'], requires_grad = False, dtype=torch.float64)#0.2609) #nn.Parameter(torch.tensor(self.dicJ['Jee'], requires_grad = True, dtype=torch.float64)) #0.2609 nA, wong and wang
        self.Jei = nn.Parameter(torch.tensor(self.dicJ['Jei'], requires_grad = True, dtype=torch.float64))
        self.Jie = nn.Parameter(torch.tensor(self.dicJ['Jie'], requires_grad = True, dtype=torch.float64))
        self.Jii = nn.Parameter(torch.tensor(self.dicJ['Jii'], requires_grad = True, dtype=torch.float64))
        self.Jin = nn.Parameter(torch.tensor(self.dicJ['Jin'], requires_grad = True, dtype=torch.float64))

        #--- Initialize model variables here
    def initiate_state(self):
        self.prev_r_e = torch.zeros((self.N)) # torch.ones(self.N) shows more obvious results
        self.prev_r_i = torch.zeros((self.N))
        self.prev_s_ampa = torch.zeros((self.N))
        self.prev_s_gaba = torch.zeros((self.N))
        self.dr_e_dt = torch.zeros((self.N))
        self.dr_i_dt = torch.zeros((self.N))
        self.s_ampa = torch.tensor(0.)
        self.i_tot_e = torch.tensor(0.)
        self.i_tot_i = torch.tensor(0.)
		

    def phi(self, I_tot, a, b, hm): #)))  # this use a lot of memory - exponential part
      
        if (torch.isnan(I_tot).any())== True:
                    np.savetxt('NaNinput.txt', I_tot.detach().numpy())
                    #for param_tensor in model.state_dict():
                    #print(param_tensor, "\t", model.state_dict()[param_tensor].size())
                    #optimizer.state_dict()['params']
                    inhomo = model_new.state_dict() #.values()
                    homogeneous_array = [str(value) for value in inhomo.items()]
                    L2 = np.array(list(homogeneous_array))
                    np.savetxt('param.txt', L2, delimiter=" ", fmt='%s')# model.state_dict())

                    gradients_dict = {}
                    print("loss is nan")
                    # Access gradients
                    for name, param in model_new.named_parameters():
                         gradients_dict[name] = param.grad
                    #homogeneous_array = [str(value) for value in inhomo.items()]
                    #L3 = np.array(list(gradients_dict.items()))
                    #ic(L3) does not work
                    #np.savetxt('grad_param.txt', L3)#, delimiter=" ", fmt='%s')# model.state_dict())
                    #ic(I_tot)
                    quit()
                    sys.exit() #sys.
        mulaI = a * I_tot
        addB = mulaI + b
        expo = torch.exp(-addB)
        return hm / (1 + expo)
    #the operation Jee_re = self.Jee * prev_r_e => triggers inplace error  

    def forward(self, In):
        #--- Compute values of interest
   
        s_gaba_wie = self.prev_s_gaba @ self.wie
        s_ampa_wei = self.prev_s_ampa @ self.wei
        s_gaba_wii = self.prev_s_gaba @ self.wii
        s_ampa_wee = self.prev_s_ampa @ self.wee
      
        self.i_tot_e = (self.Jee * s_ampa_wee) - (self.Jie * s_gaba_wie) + (self.Jin* (In)) #+ self.I_noise_E
        self.i_tot_i = (self.Jei * s_ampa_wei) - (self.Jii * s_gaba_wii)

        phi_arr_e = self.phi(self.i_tot_e, self.ae, self.be, self.hme)
        phi_arr_i = self.phi(self.i_tot_i, self.ai, self.bi, self.hmi)

        self.dr_e_dt = ((-self.prev_r_e + phi_arr_e) / self.taue)
        self.dr_i_dt = ((-self.prev_r_i + phi_arr_i) / self.taui)

        r_e = self.prev_r_e+ self.dr_e_dt * self.dt
        r_i = self.prev_r_i + self.dr_i_dt * self.dt

        dS_amp_dt = (- self.prev_s_ampa / self.tauAMPA) + r_e + self.Se_noise
        s_ampa = self.prev_s_ampa+ dS_amp_dt * self.dt

        dS_gab_dt = (- self.prev_s_gaba / self.tauGABA) + r_i + self.Si_noise
        s_gaba = self.prev_s_gaba + dS_gab_dt * self.dt

        self.prev_r_e = r_e
        self.prev_r_i = r_i
        self.prev_s_ampa = s_ampa
        self.prev_s_gaba = s_gaba


        return self.prev_r_e, self.prev_r_i, self.dr_e_dt, self.dr_i_dt, self.prev_s_ampa, self.prev_s_gaba
    


    # \\\\ Parameters

class Batch:
    def __init__(self, param, simu,len_sim, mul):
     # 1  \\\\\\\\\\\ BIG Bottom up sensory input
        self.N_short = param.N # - 6
        print("batch len simu", len_sim)
        self.IN= torch.zeros(len_sim, self.N_short ,self.N_short)
        
        self.get_sensory_input(param,simu, mul)
        

    def get_sensory_input(self, param, simu, mul):
        for i in range(0, self.N_short):
            #index = i+1
            st = Stim(param,simu, dur=simu.T,f =[i], ISI=0, mul= mul) 
            In, _,_,_,_ =st.sensoryInput(param, simu, sigmaIn = 2.)
            self.IN[:,:, i] = torch.tensor(In)
            #sti = torch.tensor(In).float()

    # 2
    def train_test_dataset(self):
        #create a random list containing each of our stimuli types
        num_stimuli = self.IN.shape[2]
        rand_idx = np.arange(0, num_stimuli)
        rng = np.random.default_rng(1245)
        rng.shuffle(rand_idx)

        # split this random list into test and train index. and filter the IN with those indexes
        val_split_index = int(np.floor(0.7 * num_stimuli))
        test_idx, train_idx = rand_idx[val_split_index:], rand_idx[:val_split_index]
        
        train_IN = self.IN[:,:, train_idx].permute(2,0,1)
        test_IN = self.IN[:,:, test_idx].permute(2,0,1)

        return train_IN, test_IN
    
    # 2 bis, just permute the entire dataset
    def permuted_pitch(self):
        #create a random list containing each of our stimuli types
        num_stimuli = self.IN.shape[2]
        rand_idx = np.arange(0, num_stimuli)
        rng = np.random.default_rng(1245)
        rng.shuffle(rand_idx)

        # split this random list into test and train index. and filter the IN with those indexes
        #val_split_index = int(np.floor(0.7 * num_stimuli))
        #new_idx = rand_idx[val_split_index:], rand_idx[:val_split_index]
        
        train_IN = self.IN[:,:, rand_idx].permute(2,0,1)
        #test_IN = self.IN[:,:, test_idx].permute(2,0,1)

        return train_IN #, test_IN


    # 3
    def get_Targets(self, Inputs):
        Targets = torch.zeros_like(Inputs)
        num_stimuli, _,_ = Inputs.shape
        for stim_idx in range(num_stimuli):
            Targets[stim_idx,:,:] = get_expected_Y_relu(Inputs[stim_idx,:,:])
        return Targets

    # 4
    def create_dataloader(self, Inputs, Targets):   
        dataset = TensorDataset(Inputs, Targets)
        return DataLoader(dataset, batch_size=1, shuffle = True) #one sample per batch

    # 5
    def preprocess_data(self):
        train_IN, test_IN = self.train_test_dataset()
        #get expected target for every stimuli type /batch
        train_Targets = self.get_Targets(train_IN) 
        test_Targets = self.get_Targets(test_IN) 

        train_dataloader = self.create_dataloader(train_IN, train_Targets)
        test_dataloader = self.create_dataloader(test_IN, test_Targets)
        return train_dataloader, train_Targets, train_IN, test_dataloader, test_Targets, test_IN
    
    #6
    def safety_plot(self, train_IN, train_Targets):  
        X_proba = make_it_proba(train_IN[4,:,:])
        Y = train_Targets[4,:,:]
        N = X_proba.shape[1]      
        t=30
        plt.plot(torch.arange(1, N+1), X_proba[t,:], label= f"Proba Input at t={t}") #not in proba
        plt.plot(torch.arange(1, N+1), Y.detach().numpy()[t,:], label = f"target at t={t}") #in proba
        plt.legend()
        plt.show()

class Batch_for_NLLL:
    def __init__(self, IN):
        self.Input =  IN
        BatchxTime = self.Input.shape[0]*  self.Input.shape[1]
        self.Target_index = torch.zeros(BatchxTime)
        self.make_Target_index(self.Input)
        
        self.Input_reshaped = self.Input.reshape(self.Input.shape[0]*  self.Input.shape[1], self.Input.shape[2])
    
    def make_Target_index(self, Input):
        time_stim = 0
        for batch in range(0, Input.shape[0]):
            for time in range(0, Input.shape[1]):
                self.Target_index[time + time_stim] = get_stimuli_input(Input[batch,:,:]).item()
            time_stim += Input.shape[1]



param = Parameter(N = 30)#30 # N=20
# \\\\ Simulation time: T in s  (2s before)
sim = Simulation(dt=1e-3,T=.0510) 
#sim.printSim()
# \\\\ Bottom up sensory input
repeat_input = 5
stimuli = Stim(param, sim, dur=sim.T, f=[10], mul = repeat_input)


In, range_sim, len_sim, w, sigmaIn = stimuli.sensoryInput(param, sim, sigmaIn=2.) #2.


plt.plot(np.arange(In.shape[1]), In[40,:])


plt.show()

batch_instance_tot = Batch(param=param, simu=sim, len_sim = len_sim, mul = repeat_input)
IN_tot= batch_instance_tot.permuted_pitch() #IN_tot = batch_instance_tot.IN#.permute(2,0,1)
Target = batch_instance_tot.get_Targets(IN_tot)

batch_instance_NLLL_tot = Batch_for_NLLL(IN_tot)
Target_idx_NLLL_tot = batch_instance_NLLL_tot.Target_index
Input_NLLL_tot =batch_instance_NLLL_tot.Input_reshaped

J1 = {'Jee': 0.2609, 'Jei': 0.004, 'Jie': 0.05, 'Jii': 0.6, 'Jin': 0.00695} #old Jee: 0.072

HeatMap(Input_NLLL_tot.detach().numpy(), Input_NLLL_tot.detach().numpy(), J = J1)
# HeatMap(Input_NLLL_tot.detach().numpy(), Input_NLLL_tot.detach().numpy(), title = "Firing rate in Hz of the external input unit over time")
sys.exit()



def gaussian_(x, sigma = 3):
    w = np.exp(-((x) ** 2) / (
                2 * (sigma) ** 2))
    return w
#(paramf) - (self.f_stim)
x = np.arange(0,20)
FP = np.ones(1) * 10
X = x - 0 #- FP
plt.plot(X, gaussian_(X))
plt.show()

sys.exit()






J1 = {'Jee': 0.2609, 'Jei': 0.004, 'Jie': 0.05, 'Jii': 0.6, 'Jin': 0.00695} #old Jee: 0.072

model_n = Model(param,sim, J1, Input_NLLL_tot)
model_n.initiate_state()      
r_e = torch.zeros(Input_NLLL_tot.shape)   
r_i= torch.zeros(Input_NLLL_tot.shape)
_2 = torch.zeros(Input_NLLL_tot.shape)
print(r_e.shape)
#P[i,:], I[i,:], dPdt[i,:], dridt, ampa, gaba 
for i, b in enumerate(Target_idx_NLLL_tot):
    #print(Target_idx_NLLL_tot.shape)
    #X = model_n.forward(Input_NLLL_tot)
    #ic(len(X))
    r_e[i,:], r_i[i,:], _2[i,:], _, _, _ = model_n.forward(Input_NLLL_tot[int(i), :])

HeatMap(r_e[0:50,:].detach().numpy(), r_i[0:50,:].detach().numpy(), J = J1)


