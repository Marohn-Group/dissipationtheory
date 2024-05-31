from dissipationtheory.constants import ureg
from dissipationtheory.dissipation import blds_perpendicular_jit
from dissipationtheory.dissipation import CantileverModelJit, SampleModel1Jit
from dissipationtheory.dissipation import theta1norm_jit
import numpy as np
import matplotlib.pyplot as plt
import time

cantilever_jit = CantileverModelJit(
            f_c = 81e3, 
            V_ts = 3.0,
            R = 80e-9,
            d = 300e-9
)

self = {
    'sample1_jit': SampleModel1Jit(
            cantilever = cantilever_jit,
            h_s = 3000e-9,
            epsilon_s = complex(3.4, -0.34),
            mu = 2.7e-10,
            rho = 1e21,
            epsilon_d = complex(3.4, -0.34),
            z_r = 300e-9
    )
}

omega_m = ureg.Quantity(np.logspace(start=np.log10(1e0), stop=np.log10(1e6), num=21), 'Hz')
freq = ureg.Quantity(np.zeros_like(omega_m), 'Hz')

tic = time.perf_counter()
for index, omega_ in enumerate(omega_m):

    freq[index] = blds_perpendicular_jit(theta1norm_jit, self['sample1_jit'], omega_).to('Hz')

toc = time.perf_counter()
print(f"BLDS computation time {toc - tic:0.4f} s")

plt.semilogx(omega_m.to('Hz').magnitude, np.abs(freq.to('Hz').magnitude), 'o-')
plt.title('magnitude of the blds frequency shift vs modulation frequency')
plt.show()