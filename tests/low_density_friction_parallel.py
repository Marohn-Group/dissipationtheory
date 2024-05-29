from dissipationtheory.constants import ureg
from dissipationtheory.dissipation import CantileverModel, SampleModel1, SampleModel2
from dissipationtheory.dissipation import theta1norm, theta2norm, gamma_parallel, gamma_perpendicular
from dissipationtheory.dissipation import gamma_parallel_approx, gamma_perpendicular_approx
from dissipationtheory.dissipation import CantileverModelJit, SampleModel1Jit, SampleModel2Jit
from dissipationtheory.dissipation import theta1norm_jit, theta2norm_jit, gamma_parallel_jit, gamma_perpendicular_jit
import pandas as pd
import numpy as np
import os
from copy import deepcopy 
import matplotlib.pyplot as plt
import scipy

cantilever =  CantileverModel(
    f_c = ureg.Quantity(81, 'kHz'), 
    V_ts = ureg.Quantity(3, 'V'), 
    R = ureg.Quantity(80, 'nm'),  # from 80
    d = ureg.Quantity(300, 'nm')
)

self = {
    'sample1': SampleModel1(
        cantilever = cantilever,
        h_s = ureg.Quantity(3000., 'nm'),
        epsilon_s = ureg.Quantity(complex(11.9, -0.05), ''),
        mu = ureg.Quantity(2.7E-10, 'm^2/(V s)'),
        rho = ureg.Quantity(1e21, '1/m^3'),
        epsilon_d = ureg.Quantity(complex(11.9, -0.05), ''),
        z_r = ureg.Quantity(300, 'nm')
    ),   
    'sample2': SampleModel2(
        cantilever = cantilever,
        h_d = ureg.Quantity(0., 'nm'),
        epsilon_d = ureg.Quantity(complex(11.9, -0.05), ''),
        epsilon_s = ureg.Quantity(complex(11.9, -0.05), ''),
        mu = ureg.Quantity(2.7e-10, 'm^2/(V s)'),
        rho = ureg.Quantity(1e21, '1/m^3'),
        z_r = ureg.Quantity(300, 'nm')
    )
}

print("parallel case")
rho_trial = ureg.Quantity(np.logspace(start=np.log10(1e15), stop=np.log10(1e23), num=9), '1/m^3')
(rho, gamma0) = gamma_parallel_approx(rho_trial, self['sample2'])

gamma1 = ureg.Quantity(np.zeros_like(rho), 'pN s/m')
gamma2 = ureg.Quantity(np.zeros_like(rho), 'pN s/m')

err1 = np.zeros_like(rho)
err2 = np.zeros_like(rho)
ratio = np.zeros_like(rho)

for index, rho_ in enumerate(rho):

    self['sample1'].rho = rho_
    gamma1[index] = gamma_parallel(theta1norm, self['sample1'])

    self['sample2'].rho = rho_
    gamma2[index] = gamma_parallel(theta2norm, self['sample2'])

    x = rho_.to('1/m^3').magnitude

    y0 = gamma0[index].to('pN s/m').magnitude
    y1 = gamma1[index].to('pN s/m').magnitude
    y2 = gamma1[index].to('pN s/m').magnitude
    

    err1[index] = (y1 - y0)/y0
    err2[index] = (y2 - y0)/y0
    ratio[index] = y0/y1

    print("{:0.2e} 1/m^2 0:{:0.2e} pN s/m 1:{:0.2e} pN s/m 2:{:0.2e} pN s/m 1:{:0.2e} 2:{:0.2e} r:{:0.3f}".format(
        x, y0, y1, y2, err1[index], err1[index], ratio[index]))

if 1:

    plt.loglog(rho.to('1/m^3').magnitude, gamma0.to('pN s/m').magnitude, 'o', label='approx')
    plt.loglog(rho.to('1/m^3').magnitude, gamma1.to('pN s/m').magnitude, 's-', label='exact, sample 1')
    plt.loglog(rho.to('1/m^3').magnitude, gamma2.to('pN s/m').magnitude, 'x--', label='exact, sample 2')
    plt.title('parallel friction vs charge density')
    plt.legend()
    plt.show()