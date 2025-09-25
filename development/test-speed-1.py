import numpy as np
# import pandas as pd
# import matplotlib.pylab as plt
import time

from dissipationtheory.constants import ureg
from dissipationtheory.dissipation9a import CantileverModel, SampleModel1, SampleModel2, SampleModel3
from dissipationtheory.dissipation9b import SampleModel1Jit, SampleModel2Jit, SampleModel3Jit

from dissipationtheory.dissipation13e import twodimCobject

def test1():

    cantilever = CantileverModel(
        f_c = ureg.Quantity(60.360, 'kHz'),
        k_c = ureg.Quantity(2.8, 'N/m'), 
        V_ts = ureg.Quantity(1, 'V'), 
        R = ureg.Quantity(57, 'nm'),
        angle = ureg.Quantity(24.2, 'degree'),
        L = ureg.Quantity(2250, 'nm'))

    sample1 = SampleModel1(
        cantilever = cantilever,
        h_s = ureg.Quantity(15, 'nm'),
        epsilon_s = ureg.Quantity(complex(18.2, -0.1), ''),
        epsilon_d = ureg.Quantity(complex(1e6, 0), ''),
        sigma = ureg.Quantity(9.7e-7, 'S/cm'),
        rho = ureg.Quantity(1.9e15, '1/cm^3'),
        z_r = ureg.Quantity(1, 'nm'))

    sample1_jit = SampleModel1Jit(**sample1.args())

    start = time.perf_counter()

    obj = twodimCobject(sample1_jit)
    obj.addtip(h=ureg.Quantity(200, 'nm'))
    obj.set_alpha(1.0e-6)
    obj.set_breakpoints(15)
    # obj.properties_dc()
    # obj.properties_ac(omega_m=1.0e5)
    obj.properties_am(omega_m=1.0e5, omega_am = 250.)

    finish = time.perf_counter()
    duration = finish - start

    print(f'test 1 duration = {duration:4.2e} seconds')

    return duration

if __name__ == "__main__":   

    t1 = test1()

    