import numpy as np

def default_parameters_network(**kwargs):
    pars = {}

    # ¤ parameter of the phi function Not tweakable parameters
    pars['ae'] = 18.26;  # 2 #Wong have to check # Modelling and Meg Gain of the E populaiton
    pars['be'] = -5.38;  # Threshold of the E populaiton
    pars['hme'] = 78.67

    pars['ai'] = 21.97
    pars['bi'] = -4.81
    pars['hmi'] = 125.62

    # ¤ time constants
    pars['taue'] = 0.005#0.005#0.020;  # seconds [Brunel 2001] (4.5 +- 2.4 ms)
    pars['taui'] = 0.005#0.005 #0.010;  # seconds [Brunel 2001] (4.5 +- 2.4 ms)
    pars['tauNMDA'] = 0.100;  # NMDA-gating time constants (s) [Brunel 2001]
    pars['tauGABA'] = 0.005; #0.005 # GABA-gating time constants (s) [Brunel 2001]
    pars['tauAMPA'] = 0.002; #0.002 # AMPA-gating time constants (s) [Brunel 2001]  #tells how quickly the neurons integrates its input.
    # the smaller it is, the faster the voltage decay to 0

    # ¤ population parameters
    pars['gamma'] = 0.641;  # NMDA coupling   [Brunel 2001] or 0.641/1000
    pars['sigma'] = 0.0007;  # noise amplitude [Wong 2006]   # it s 0.007 (nA) instead !!!
    pars['I0e'] = 0.2346;  # constant population input, exc (nA) [Wong2006] -- called I0 in the paper
    pars['I0i'] = 0.17;  # constant population input, inh (nA)

    # connectivity matrix:
    pars['sigmaIn'] = 3#3
    pars['sigmaEI'] = 3
    pars['sigmaInh'] = [0.2, pars['sigmaIn']]
    # External input
    pars['I_ext'] = 0.
    pars['c_dash'] = 90 #40 #90 # imput by default
    pars['mu0'] = 30  # Hz wong 2006  -
    pars['Jext'] = 0.01  # % nA/Hz; # B_wong easy  -  on the paper I see #0.2245e-3 #na.Hz-1 for AMPAr JA,ext
    pars['I1'] = pars['Jext'] * pars['mu0'] * (1 + pars['c_dash'] / 100)
    pars['I2'] = pars['Jext'] * pars['mu0'] * (1 - pars['c_dash'] / 100)
    # ¤ conductivities: the strength of the synaptic connection
    pars['Jee'] = 0.2609#.072#0.2; Wong and wang # nA #recurrence parameter - Parameter I can tweak
    
    
    pars['Jie'] = .05 #0.2;  # nA
    pars['Jei'] = .004#1.4;  # nA
    pars['Jii'] = .6#6.7;  # nA
    pars['Jin'] = .00695 #0.008;  # nA

    #pars['Jiq'] = 0.85;  # nA
    #pars['Jes'] = 3.5;  # nA
    #pars['Jsi'] = 0.12;  # nA
    #pars['Jem'] = 2.2;  # nA

    #J1 = {'Jee': 0.072, 'Jei': 0.004, 'Jie': 0.05, 'Jii': 0.6, 'Jin': 0.00695}

    # I noise
    pars['sigma'] = 0.0007;  # noise amplitude [Wong 2006]   # it s 0.007 (nA) instead !!!
    eta = np.random.normal(size=(4, 1))  # return N by 4 matrix of normal random number
    pars['I_noise'] = pars['sigma'] * eta  # nA #Inoise is nA sigma is nA
    # I need to add the noise from the external input. I put just I_noise but not the gaussian filter neither other type of noise

    # simulation parameters
    pars['T'] = 3  # 2#0.002 #2.       # Total duration of simulation [s]
    pars['dt'] = .00002  # .00002 #.01       # Simulation time step [s]
    pars['r_init'] = 0.2  # Initial value of E
    

    # range_t = np.arange(0,2500e-3,0.00002)

    # External parameters if any
    pars.update(kwargs)

    # time parameter - Vector of discretized time points [s]
    pars['range_t'] = np.arange(0, pars['T'], pars['dt'])
    pars['Lt'] = pars['range_t'].size


    # general parameter:
    pars['NumN'] = 20#number of neurons
    pars['f'] = np.arange(1, pars['NumN']+1) #% unitless

    # Input paramters
    pars['In0']    = 0;    #% Spontaneous firing rate of input populations (Hz)
    pars['InMax']  = 50;   #% Max firing rate of input populations (Hz)
    pars['Iq0']    = 0;    #% Spontaneous firing rate of feedback populations (Hz)
    pars['IqMax']  = 10;   #% Max firing rate of feedback populations (Hz)


    return pars


pars = default_parameters_network()
print(pars)