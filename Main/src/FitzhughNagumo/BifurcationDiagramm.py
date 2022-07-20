
import os
import sys
import numpy as np
from matplotlib import colors as colors
from matplotlib import pyplot as plt
from matplotlib import animation as animation

from functools import partial
import numpy as np
import scipy.integrate
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches #used to write custom legends

# Implement the jacobian of the system and use numpy.linalg.eig to determine the local topology of the equilibria.

# You can use the following color code:
EQUILIBRIUM_COLOR = {'Stable point':'C2',
                     'Unstable point':'C3',
                     'Saddle':'C0',
                     #'Stable node':'C4',
                     #'Unstable node':'C6',
                     'Center':'C5'}

scenarios = [
    {"a":-.3, "b":1.4, "tau":20, "I":0},
    {"a":-.3, "b":1.4, "tau":20, "I":0.23},
    {"a":-.3, "b":1.4, "tau":20, "I":0.5}
]

#one fixed point
scenarios = [
    {"a":.25, "b":0.002, "tau":0.002, "I":0},
    {"a":.25, "b":0.002, "tau":0.002, "I":0.4},
    {"a":.25, "b":0.002, "tau":0.002, "I":0.9}
]

#three fixed point
scenarios = [
    {"a":.25, "b":0.002, "tau":0.02, "I":0},
    {"a":.20, "b":0.002, "tau":0.02, "I":0},
    {"a":.1, "b":0.002, "tau":0.02, "I":0}
]



def fitzhugh_nagumo(x, t, a, b, tau, I):
    """Time derivative of the Fitzhugh-Nagumo neural model.
    Args:
       x (array size 2): [Membrane potential, Recovery variable]
       a, b (float): Parameters.
       tau (float): Time scale.
       t (float): Time (Not used: autonomous system)
       I (float): Constant stimulus current.
    Return: dx/dt (array size 2)
    """
    return np.array([(x[0] * (a - x[0]) * (x[0]-1)) - x[1] + I,
                     (b * x[0]) - (tau * x[1])])#np.array([x[0] - x[0]**3 - x[1] + I,
    #         (x[0] - a - b * x[1])/tau])

def get_displacement(param, dmax=0.5,time_span=np.linspace(0,600, 1000), number=20): #0,100,1000
    # We start from the resting point...
    ic = scipy.integrate.odeint(partial(fitzhugh_nagumo, **param),
                                y0=[0,0],  #0 0
                                t= np.linspace(0,999, 1000))[-1] #999
    # and do some displacement of the potential.
    traj = []
    for displacement in np.linspace(0,dmax, number):
        traj.append(scipy.integrate.odeint(partial(fitzhugh_nagumo, **param),
                                           y0=ic+np.array([displacement, 0]),
                                           t=time_span))
    return traj

def plot_isocline(ax, a, b, tau, I, color='k', style='--', opacity=.5, vmin=-1,vmax=1):
    """Plot the null iscolines of the Fitzhugh nagumo system"""
    v = np.linspace(vmin,vmax,100)
    ax.plot(v, (v * (a - v) * (v - 1)) + I, style, color='darkgrey', alpha=1,linewidth = 2 ) #v - v**3 + I
    ax.plot(v, (b * v)/tau, style, color='black', alpha=1, linewidth = 2)   #(v - a)/b

def plot_vector_field(ax, param, xrange, yrange, steps=50):
    # Compute the vector field
    x = np.linspace(xrange[0], xrange[1], steps)
    y = np.linspace(yrange[0], yrange[1], steps)
    X, Y = np.meshgrid(x,y)

    dx, dy = fitzhugh_nagumo([X,Y],0,**param)

    # streamplot is an alternative to quiver
    # that looks nicer when your vector filed is
    # continuous.
    ax.streamplot(X,Y,dx, dy, color=(0,0,0,.1))

    ax.set(xlim=(xrange[0], xrange[1]), ylim=(yrange[0], yrange[1]))


def find_roots(a,b,I, tau):
    # The coeficients of the polynomial equation are:
    # 1           * v**3
    # 0           * v**2
    # - (1/b - 1) * v**1
    # - (a/b + I) * v**0
    coef = [-1,a+1, -a-(b/tau), I]

    # We are only interested in real roots. # np.isreal(x) returns True only if x is real.
    # The following line filter the list returned by np.roots     # and only keep the real values.
    roots = [np.real(r) for r in np.roots(coef) if np.isreal(r)]

    # We store the position of the equilibrium.
    return [ [r, r *(a - r) *( r -1 ) + I] for r in roots]#[r, r - r**3 + I] [r, r (a - r) ( r -1 ) + I] [r, (-b/tau)*r]

eqnproot = {}
for i, param in enumerate(scenarios):
    eqnproot[i] = find_roots(**param)

#print(eqnproot)

def jacobian_fitznagumo(v, w, a, b, tau, I):
    """ Jacobian matrix of the ODE system modeling Fitzhugh-Nagumo's excitable system
    Args
    ====
        v (float): Membrane potential
        w (float): Recovery variable
        a,b (float): Parameters
        tau (float): Recovery timescale.
    Return: np.array 2x2"""
    return np.array([[(- 3 * v**2) + (2*v) - a + (2*a*v) , -1],
                     [b, -tau]])

def stability(jacobian):
    """ Stability of the equilibrium given its associated 2x2 jacobian matrix.
    Use the eigenvalues.
    Args:
        jacobian (np.array 2x2): the jacobian matrix at the equilibrium point.
    Return:
        (string) status of equilibrium point.
    """

    eigv = np.linalg.eigvals(jacobian)

    if all(np.real(eigv)==0) and all(np.imag(eigv)!=0):
        nature = "Center"
    elif np.real(eigv)[0]*np.real(eigv)[1]<0:
        nature = "Saddle"
    else:
        stability = 'Unstable' if all(np.real(eigv)>0) else 'Stable'
        nature = stability + (' point') #node' if all(np.imag(eigv)!=0) else
    return nature


def plot_phase_diagram(param, ax=None, title=None):
    """Plot a complete Fitzhugh-Nagumo phase Diagram in ax.
    Including isoclines, flow vector field, equilibria and their stability"""
    if ax is None:
        ax = plt.gca()
    if title is None:
        title = "Parameters, {}".format(param)

    ax.set(xlabel='v', ylabel='w', title=title)

    # Isocline and flow...
    #xlimit = (-1.5, 1.5)
    #ylimit = (-.6, .9)
    xlimit = (-0.25, 1)
    ylimit = (0.30, 0.7)#0.80, 1.20 #-0.1, 0.3#[(1/sc['b'])*(x-sc['tau']) for x in xrange]

    plot_vector_field(ax, param, xlimit, ylimit)
    plot_isocline(ax, **param, vmin=xlimit[0],vmax=xlimit[1])

    # Plot the equilibria
    eqnproot = find_roots(**param)
    #e0 = r(v)   e1 = w = ( w= v(a-v)(v-1) +I)
    eqstability = [stability(jacobian_fitznagumo(e[0],e[1], **param)) for e in eqnproot]
    for e,n in zip(eqnproot,eqstability):
        ax.scatter(*e, color= EQUILIBRIUM_COLOR[n])
        print(EQUILIBRIUM_COLOR[n])
        print(n[:6]) #n[:6]
        print(*e)
        e[0] = 0.2
        e[1] = 0.6  #w = 0.9 for I = 0.9
        # Show a small perturbation of the stable equilibria...
        time_span = np.linspace(0, 600, num=2500)#0, 200, num=1500)
        if n[:6] == 'Unstab':#'Saddle': [:6]
            for perturb in (0.1, 0.4):#(0.1, 0.6): (0.01, -0.3
                ic = [e[0]+(perturb*e[0]),e[1]] #abs

                traj = scipy.integrate.odeint(partial(fitzhugh_nagumo, **param),
                                              y0=ic,
                                              t=time_span)
                ax.plot(traj[:,0], traj[:,1])
                print(traj[:,0], traj[:,1])

    # Legend
    labels = frozenset(eqstability)
    ax.legend([mpatches.Patch(color=EQUILIBRIUM_COLOR[n]) for n in labels], labels,
              loc='lower right')


# Plot the bifurcation diagram for v with respect to parameter I.

ispan = np.linspace(0.0,0.06,1000)
bspan = np.linspace(-0.1,0.8,1000)

I_list = []
eqs_list = []
nature_legends = []
trace = []
det = []

for I in ispan:
    param = {'I': I, 'a': 0.38, 'b':0.002, 'tau': 0.02
             }
    roots = find_roots(**param)
    for v,w in roots:
        J = jacobian_fitznagumo(v,w, **param)
        nature = stability(J)
        nature_legends.append(nature)
        I_list.append(I)
        eqs_list.append(v)
        det.append(np.linalg.det(J))
        trace.append(J[0,0]+J[1,1])


fig, ax = plt.subplots(1,1,figsize=(10,5))
labels = frozenset(nature_legends)
ax.scatter(I_list, eqs_list, c=[EQUILIBRIUM_COLOR[n] for n in nature_legends], s=5.9)
ax.legend([mpatches.Patch(color=EQUILIBRIUM_COLOR[n]) for n in labels], labels,
          loc='lower right')
ax.set(xlabel='External stimulus, $I_{ext}$',
       ylabel='Equilibrium Membrane potential, $v^*$');

plt.show()