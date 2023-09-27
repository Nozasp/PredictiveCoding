import numpy as np
from scipy.linalg import expm, sinm, cosm
import math
import param

#import linalg package of the SciPy module for the LU decomp
import scipy.linalg as linalg
import matplotlib.pyplot as plt

import smop
import decimal, numpy
from decimal import Decimal as D
import scipy

N = 20
s = 2
f = list(range(1, N+1))


k = numpy.matrix(range(1, N+1))

#print(k)
n = 1 / (math.sqrt(2 * math.pi) * N * s)
#gaussW = numpy.multiply(expm(- numpy.power((k- numpy.transpose(k)),(2/(2 * s^2)))),n)

b = numpy.dot(expm(- numpy.power((k - numpy.transpose(k)), int(2/(2 * s^2)))), n)
#print(b)



def dogFilter(sIn, sOut, N):
    k = numpy.matrix(range(1, N+1))

    gaussIn = (expm(-numpy.divide(numpy.power((k - k.getH()),int(2)), (2 * sIn**2)))) # (2 * sin) **2  or 2 * (sin**2) ??
    gaussOut = (expm(-numpy.divide(numpy.power((k - k.getH()),int(2)), (2 * sOut**2))))
    #print("gaussIn", gaussIn)
    #print("decimal_In", decimal_sIn)
    #print("Decimal_In", Decimal_In)
    #print("Power part", numpy.power((k - numpy.transpose(k)),int(2)))
    #print("division part", -numpy.divide(numpy.power((k - numpy.transpose(k)),int(2)), (2 * sIn**2)))
    dog = numpy.subtract(gaussOut, gaussIn)
    #print("dog:", dog)
    dog1 = numpy.divide(dog, numpy.divide(0.88**2, max(dog.flatten())))

    return dog1


sigmaIn  = 3;
sigmaEI  = sigmaIn;
sigmaQie = sigmaIn;
sigmaInh = [0.2, sigmaIn];

wes  = numpy.identity(N) # eye = identiy matrix of size N*N
#wie  = dogFilter(sigmaInh[0], sigmaInh[1], N);
#wii  = dogFilter(sigmaInh[0], sigmaInh[1], N);

#print(wii)

#gaussIn = expm(-numpy.power((k[1] - k[2]),int(2/(2 * 3**2)))) # (2 * sin) **2  or 2 * (sin**2) ??
#gaussOut = math.exp(-(((k[1] - k[2])**int(2))/(2 * 0.2**2)))
#gaussIn = math.exp(-(((k[1] - k[2])**int(2))/(2 * 3**2)))

In = 0.2
Out = 3

Hmatrix = k - k.getH()
sub = (2 * (Out**2))
sub2 = np.full(20*20, sub, float)
subOut = sub2.reshape((20,20))
subIn = np.full(20*20, 2 * (In**2), float)
subIn = subIn.reshape((20,20))

#sub3 = np.array([2 * (3**2)]*20)

A = np.array(-np.power(Hmatrix,2))
#exponent = numpy.linalg.lstsq(numpy.transpose(A), (sub4),rcond=-1)
#exponent2 = numpy.dot(A, numpy.linalg.pinv(sub4,rcond=-1))
#exponent3 = scipy.sparse.linalg.lsqr(A, sub3, damp=0.0, atol=1e-08, btol=1e-08, conlim=100000000.0, iter_lim=None, show=False, calc_var=False, x0=None)

#call the lu_factor function
LU = scipy.linalg.lu_factor(A)

#solve given LU and B
xOut = scipy.linalg.lu_solve(LU, subOut)
xIn = scipy.linalg.lu_solve(LU, subIn)

gaussIn = expm(xIn)
gaussOut = expm(xOut)

dog = numpy.subtract(gaussOut, gaussIn)
#print("dog:", dog)
#dog1 = numpy.divide(dog, numpy.divide(0.88**2, max(dog.flatten())))
LU2 = scipy.linalg.lu_factor(dog)
DENO = numpy.divide(0.88**2, max(dog.flatten()))
DENO2 = np.full(20*20, DENO, float)
Deno3 = DENO2.reshape((20,20))

x2 = scipy.linalg.lu_solve(LU, Deno3)

#plt.plot(f, x2[:,3:5], label = ["wii","wii2"])#wii[5:], label = "wii")
#plt.plot(f, param.wei[:, 1:5], label = "wei")
plt.plot(f, param.wes[:, 1:5], label = "wes")
plt.legend()
plt.show()

#print(x2)


#print(expm(exponent3))
#power= numpy.power((k - k.getH()), 2) / (2/(2 * 3**2))
#print(power)
#print(gaussOut - gaussIn)
#scipy.sparse.linalg.lsqr(A, b, damp=0.0, atol=1e-08, btol=1e-08, conlim=100000000.0, iter_lim=None, show=False, calc_var=False, x0=None)
