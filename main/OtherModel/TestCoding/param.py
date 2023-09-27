#import matlab.engine

import math
import numpy
from scipy.linalg import expm, sinm, cosm #exponential Sin and cos for matrx objects
import matplotlib.pyplot as plt
import numpy as np

# General parameters
N = 20
#T = 0.2:0.01:1; # periods of the generative oscillator (seconds)
#f = numpy.matrix(range(1, N))   #; # unitless
f = list(range(1, N))
dt = 0.001; # seconds
initTime = 1;   # initalisation time (seconds)




# //////////////  FUNCTIONS ////////////////////////////////////////
def gaussianFilter(s, N):
    k = numpy.matrix(range(1, N+1))
    n = 1 / (math.sqrt(2 * math.pi) * N * s)
    gaussW = n * expm(-np.power((k - numpy.transpose(k)),int(2/(2 * s**2))))
    gaussW = numpy.divide( gaussW, numpy.divide((0.01**2), max(gaussW.flatten('F'))))
    return gaussW

def dogFilter(sIn, sOut, N):
    k = numpy.matrix(range(1, N))
    gaussIn = (expm(-numpy.divide(numpy.power((k - numpy.transpose(k)),int(2)), (2 * sIn**2)))) # (2 * sin) **2  or 2 * (sin**2) ??
    gaussOut = (expm(-numpy.divide(numpy.power((k - numpy.transpose(k)),int(2)), (2 * sOut**2))))

    dog = numpy.subtract(gaussOut, gaussIn)
    dog1 = numpy.divide( dog, numpy.divide(0.88**2, max(dog.flatten())))
    return dog1

#//////////////////////////////////////////////////////////////////////////////////
    # Runtime
    verb  = false  ;
    storeFields.v  = {'he', 'hi', 'hs'};
    storeFields.dv = {};
    storeFields.x  = {};




# ¤ time constants
#ostoj   = flse;


taue    = 0.020; # seconds [Brunel 2001] (4.5 +- 2.4 ms)
taui    = 0.010; # seconds [Brunel 2001] (4.5 +- 2.4 ms)
tauNMDA = 0.100; # NMDA-gating time constants (s) [Brunel 2001]
tauGABA = 0.005; # GABA-gating time constants (s) [Brunel 2001]
tauAMPA = 0.002; # AMPA-gating time constants (s) [Brunel 2001]



# ¤ population parameters
gamma = 0.641;  # NMDA coupling   [Brunel 2001]

sigma = 0.0007; # noise amplitude [Wong 2006]   # it s 0.007 (nA) instead !!!
I0e   = 0.2346; # constant population input, exc (nA) [Wong2006]
I0i   = 0.17;   # constant population input, inh (nA)


#????
ae = 18.26;
be    = -5.38;
hme   = 78.67;
ai    = 21.97;
bi    = -4.81;
hmi   = 125.62;


# ¤ Input param3ters
In0    = 0;    # Spontaneous firing rate of input populations (Hz)
InMax  = 50;   # Max firing rate of input populations (Hz)
Iq0    = 0;    ## Spontaneous firing rate of feedback populations (Hz)
IqMax  = 10;   ## Max firing rate of feedback populations (Hz)


# ¤ conductivities
Jee = 0.2;
Jie = 0.2;
Jei = 1.4;
Jii = 6.7;
Jin = 0.008;
Jiq = 0.85;
Jes = 3.5;
Jsi = 0.12;
Jem = 2.2;





## ????+

# ¤ adaptation dynamics
#alpha    = 0; # adaptation strenght
alpha = 0.022; # adaptation strenght
tauAdapt = 1.50;  # Adaptation time constant (s)

# connectivity matrices


sigmaIn  = 3;
sigmaEI  = sigmaIn;
sigmaQie = sigmaIn;
sigmaInh = [0.2, sigmaIn];
wei = gaussianFilter(sigmaEI, N);
wes = numpy.identity(N) # eye = identiy matrix of size N*N
wie = dogFilter(sigmaInh[0], sigmaInh[1], N);
wii = dogFilter(sigmaInh[0], sigmaInh[1], N);




## Whats the optimal bandwith optimising robustness and resolution?
#plot(f, [wei(5, :); wii(5, :); wie(5, :)]); legend('wei', 'wii', 'wie');
# plot lines

#plt.matshow(wei[0:5,0:5])
flat_wei = wei.flatten()

#fbig = f * 19
#plt.plot(f * 19, flat_wei, label = "wei")#wei[5:], label = "wei")
plt.plot(f, wii[1], label = "wii")#wii[5:], label = "wii")
plt.plot(f, wie[1], label = "wie")#wie[5:], label = "wie")"""
plt.legend()
plt.show()