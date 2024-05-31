from dissipationtheory.constants import ureg
from dissipationtheory.dissipation import CantileverModel, SampleModel1, SampleModel2
from dissipationtheory.dissipation import theta1norm, theta2norm, gamma_parallel, gamma_perpendicular
from dissipationtheory.dissipation import blds_perpendicular, blds_perpendicular_approx, blds_perpendicular_jit
from dissipationtheory.dissipation import gamma_parallel_approx, gamma_perpendicular_approx
from dissipationtheory.dissipation import CantileverModelJit, SampleModel1Jit, SampleModel2Jit
from dissipationtheory.dissipation import theta1norm_jit, theta2norm_jit, gamma_parallel_jit, gamma_perpendicular_jit
import numpy as np
import matplotlib.pyplot as plt

cantilever =  CantileverModel(
    f_c = ureg.Quantity(81, 'kHz'), 
    V_ts = ureg.Quantity(3, 'V'), 
    R = ureg.Quantity(80, 'nm'), 
    d = ureg.Quantity(300, 'nm')
)

cantilever_jit = CantileverModelJit(
            f_c = 81e3, 
            V_ts = 3.0,
            R = 80e-9,
            d = 300e-9
)

self = {
    'sample1': SampleModel1(
        cantilever = cantilever,
        h_s = ureg.Quantity(3000., 'nm'),
        epsilon_s = ureg.Quantity(complex(3.4, -0.34), ''),  # 11.9, -0.05
        mu = ureg.Quantity(2.7e-10, 'm^2/(V s)'), # 2.7e-10
        rho = ureg.Quantity(1e21, '1/m^3'), # 1e21
        epsilon_d = ureg.Quantity(complex(3.4, -0.34), ''),
        z_r = ureg.Quantity(300, 'nm')
    ),
    'sample2': SampleModel2(
            cantilever = cantilever,
            h_d = ureg.Quantity(0., 'nm'),
            epsilon_d = ureg.Quantity(complex(3.4, -0.34), ''),
            epsilon_s = ureg.Quantity(complex(3.4, -0.34), ''),
            mu = ureg.Quantity(2.7e-10, 'm^2/(V s)'), # 2.7e-10
            rho = ureg.Quantity(1e21, '1/m^3'), # 1e21
            z_r = ureg.Quantity(300, 'nm')
    ),
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

print("--------------")
print("Model I sample")
print("--------------")

print(self['sample1'])

print("")
print("---------------")
print("Model II sample")
print("---------------")

print(self['sample2'])

print("")
print("----")
print("blds")
print("----")
print("")

omega_m = ureg.Quantity(np.logspace(start=np.log10(1e0), stop=np.log10(1e6), num=21), 'Hz')
freq0 = np.ones_like(omega_m) * blds_perpendicular_approx(self['sample1'])
freq1 = ureg.Quantity(np.zeros_like(omega_m), 'Hz')
freq2 = ureg.Quantity(np.zeros_like(omega_m), 'Hz')
freq3 = ureg.Quantity(np.zeros_like(omega_m), 'Hz')

for index, omega_ in enumerate(omega_m):

    freq1[index] = blds_perpendicular(theta1norm, self['sample1'], omega_).to('Hz')
    freq2[index] = blds_perpendicular(theta2norm, self['sample2'], omega_).to('Hz')
    freq3[index] = blds_perpendicular_jit(theta1norm_jit, self['sample1_jit'], omega_).to('Hz')

    print("| omega {:0.2e} rad/s | blds approx {:0.2e} rad/s | sample 1 {:0.2e} rad/s | sample 2 {:0.2e} rad/s | sample 1 jit {:0.2e} rad/s |".format(
        omega_.to('Hz').magnitude,
        freq0[index].to('Hz').magnitude,
        freq1[index].to('Hz').magnitude,
        freq2[index].to('Hz').magnitude,
        freq3[index].to('Hz').magnitude))

plt.semilogx(omega_m.to('Hz').magnitude, np.abs(freq0.to('Hz').magnitude), 'o-', label='dielectric-only approx')
plt.semilogx(omega_m.to('Hz').magnitude, np.abs(freq1.to('Hz').magnitude), '>--', label='exact, sample1')
plt.semilogx(omega_m.to('Hz').magnitude, np.abs(freq2.to('Hz').magnitude), '<--', label='exact, sample2')
plt.semilogx(omega_m.to('Hz').magnitude, np.abs(freq3.to('Hz').magnitude), 's-', label='exact jit, sample1')
plt.title('magnitude of the blds frequency shift vs modulation frequency')
plt.legend()
plt.show()


