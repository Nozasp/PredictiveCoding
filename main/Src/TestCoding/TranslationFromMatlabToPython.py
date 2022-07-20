from functools import partial
import numpy as np
import scipy.integrate
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches #used to write custom legends
x0 = []
r0 = []
y0 = []
#variables
v = {'he':  x0, 'hi':  x0, 'hs':  x0, 'hm': y0, 'dhm': y0,
     'rei': r0, 'res': r0, 'rii': r0, 'Sa':  x0, 'Sg':  x0, 'Sn': x0,
     'Sm': x0}

"test"
def computeDerivatives(v, In, Iq, pars):
    v['he']

    dv=0
    return v


    """
function [dv, x] = computeDerivatives(v, In, Iq, pars)

% 1. Gating variables
eta = pars.sigma * randn([pars.N, 4]);
dv.Sa  = -v.Sa  / pars.tauAMPA + v.he + eta(:,1);
dv.Sg  = -v.Sg  / pars.tauGABA + v.hi + eta(:,2);
dv.Sn  = -v.Sn  / pars.tauNMDA + 0.641 * (1 - v.Sn) .* v.hs + eta(:,3);
dv.Sm  = -v.Sm  / pars.tauAMPA + sum(v.hm, 2) + eta(:,4);

% 2. Synaptic inputs
x.he = pars.Jee * v.Sa - pars.Jie * pars.wie * v.Sg + pars.Jin * In;
x.hi = pars.Jei * (pars.wei .* v.rei) * v.Sa - ...
pars.Jii * (pars.wii .* v.rii) * v.Sg + pars.Jsi * v.Sn;
x.hs = pars.Jes * (pars.wes .* v.res) * v.Sa + pars.Jiq * (Iq + v.Sm);
x.hm = pars.Jem * v.Sa;

% 3. Plasticiy
dv.rei = (1 - v.rei) / pars.tauAdapt + pars.alpha * v.hi * (v.he');
dv.res = (1 - v.res) / pars.tauAdapt + pars.alpha * v.hs * (v.he');
dv.rii = 0;% (1 - v.rii) / pars.tauAdapt + pars.alpha * v.hi * (v.hi');

% 3. Rate variables
dv.he = (- v.he + psix(x.he, 'e', pars)) ./ pars.taue;
dv.hi = (- v.hi + psix(x.hi, 'i', pars)) ./ pars.taui;
dv.hs = (- v.hs + psix(x.hs, 'e', pars)) ./ pars.taue;

% 4. Generative oscillator
T = repmat(pars.T, [pars.N, 1]);
dv.dhm = - (2*pi ./ T).^2 .* v.hm;
dv.hm  = -v.hm ./ T + psix(x.hm, 'e', pars) + v.dhm;

end



function px = psix(x, typ, pars)

a   = pars.(['a',  typ]);
b   = pars.(['b',  typ]);
hm  = pars.(['hm',  typ]);

px = hm  * (1 ./ (1 + exp(- (a * x + b))));

end
"""