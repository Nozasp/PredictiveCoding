from functools import partial
import numpy as np
import scipy.integrate
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches #used to write custom legends


# Implement the flow of the Fitzhugh-Nagumo model.
# And simulate some trajectories.
# Try to use small perturbation of the resting potential as inital conditions.

scenarios = [
{"a":-.3, "b":1.4, "tau":20, "I":0},
{"a":-.3, "b":1.4, "tau":20, "I":0.23},
{"a":-.3, "b":1.4, "tau":20, "I":0.5}
]

scenarios = [
    {"a":.25, "b":0.002, "tau":0.02, "I":0},
    {"a":.25, "b":0.002, "tau":0.02, "I":0.02},
    {"a":.25, "b":0.002, "tau":0.02, "I":0.04}
]


scenarios = [
    {"a":.25, "b":0.002, "tau":0.002, "I":0},
    {"a":.25, "b":0.002, "tau":0.02, "I":0},
    {"a":.25, "b":0.002, "tau":0.2, "I":0}
]

time_span = np.linspace(0, 200, num=1500)

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
                     (b * x[0]) - (tau * x[1])])    #(x[0] - a - b * x[1])/tau])   # x[0] - x[0]**3 - x[1] + I

def get_displacement(param, dmax=0.5,time_span=np.linspace(0,200, 1000), number=20):
    # We start from the resting point...
    ic = scipy.integrate.odeint(partial(fitzhugh_nagumo, **param),
                                y0=[0,0],
                                t= np.linspace(0,999, 1000))[-1]
    # and do some displacement of the potential.
    traj = []
    for displacement in np.linspace(0,dmax, number):
        traj.append(scipy.integrate.odeint(partial(fitzhugh_nagumo, **param),
                                           y0=ic+np.array([displacement,0]),
                                           t=time_span))
    return traj

# Do the numerical integration.
trajectories = {} # We store the trajectories in a dictionnary, it is easier to recover them.
for i,param in enumerate(scenarios):
    trajectories[i] = get_displacement(param, number=3, time_span=time_span, dmax=0.5)



def plot_isocline(ax, a, b, tau, I, color='k', style='--', opacity=.5, vmin=-1,vmax=1):
    """Plot the null iscolines of the Fitzhugh nagumo system"""
    v = np.linspace(vmin,vmax,100)
    ax.plot(v, (v * (a - v) * (v-1)) + I, style, color='darkgrey', alpha=1,linewidth = 2 ) #v - v**3 + I
    ax.plot(v, (b * v)/tau, style, color='black', alpha=1,linewidth = 2)#(v - a)/b



def plot_vector_field(ax, param, xrange, yrange, steps=50):
    # Compute the vector field
    x = np.linspace(xrange[0], xrange[1], steps)
    y = np.linspace(yrange[0], yrange[1], steps)
    X,Y = np.meshgrid(x,y)

    dx,dy = fitzhugh_nagumo([X,Y],0,**param)

    # streamplot is an alternative to quiver
    # that looks nicer when your vector filed is
    # continuous.
    ax.streamplot(X,Y,dx, dy, color=(0,0,0,.1))

    ax.set(xlim=(xrange[0], xrange[1]), ylim=(yrange[0], yrange[1]))



fig, ax = plt.subplots(1, 3, figsize=(20, 6))
for i, sc in enumerate(scenarios):
    plot_isocline(ax[i], **sc)
    xrange = (-0.25, 1)
    yrange = (-0.1, 0.3)#[(1/sc['b'])*(x-sc['tau']) for x in xrange]
    plot_vector_field(ax[i], sc, xrange, yrange)
    ax[i].axvline(x=0, color = "r", linewidth = 0.4)
    ax[i].axhline(y=0, color = "r", linewidth = 0.4)
    ax[i].set_ylim([-0.02, 0.03])# null
    ax[i].set_xlim([-0.1, 0.1])# null

    ax[i].set(xlabel='v', ylabel='w',
              title='Parameter of the model:'+'{}'.format(sc))

plt.show()