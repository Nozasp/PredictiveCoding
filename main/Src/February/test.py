import matplotlib.pyplot as plt
import numpy as np
import torch
I_tot = torch.zeros([200,200])
#I_tot[:] = np.nan
I_tot.fill_(np.nan)
print(I_tot.shape)

#for i in range(I_tot.shape[0]):
    #print(i)


if (torch.isnan(I_tot).any())== True:
                    print("haako", I_tot.shape) 


def normal(x, mean, sigma):
        f = 1/np.sqrt(2 * np.pi * sigma) * np.exp(-(np.power((x-mean), 2) / (2* sigma)))
        return f

def g(v):
        return np.power(v, 2)
dv_step = 0.01
x = np.arange(0.01, 5, dv_step)

p_u_given_v = normal(x, 3,1)
p_v = normal(2, g(x), 1)
numerator = p_u_given_v* p_v
p_u = np.sum(numerator * dv_step)


p_v_given_u = (numerator) / p_u

plt.plot(x, p_v_given_u)
plt.xlabel('v')
plt.ylabel('p(v|u)')
plt.show()