import numpy as np
from dissipationtheory.constants import ureg, epsilon0, qe, kb

class CantileverModel(object):

    def __init__(self, f_c, V_ts, R, d):

        self.f_c = f_c
        self.V_ts = V_ts
        self.R = R
        self.d = d

    @property
    def omega_c(self):
        return 2 * np.pi * self.f_c

    def __repr__(self):

        str = 'cantilever\n\n'
        str = str + '      resonance freq = {:.3f} kHz\n'.format(self.f_c.to('kHz').magnitude)
        str = str + '                     = {:.3e} rad/s\n'.format(self.omega_c.to('Hz').magnitude)
        str = str + '  tip-sample voltage = {:.3f} V\n'.format(self.V_ts.to('V').magnitude)
        str = str + '              radius = {:.3f} nm\n'.format(self.R.to('nm').magnitude)
        str = str + '              height = {:.3f} nm\n'.format(self.d.to('nm').magnitude)
        
        return str
    
class SampleModel1(object):
    
    def __init__(self, cantilever, h_s, epsilon_s, mu, rho, epsilon_d, z_r):

        self.cantilever = cantilever
        self.epsilon_s = epsilon_s
        self.h_s = h_s
        self.mu = mu
        self.rho = rho
        self.epsilon_d = epsilon_d
        self.z_r = z_r

    @property
    def D(self):
        return ((kb * ureg.Quantity(300., 'K') * self.mu) / qe).to('m^2/s')

    @property
    def Ld(self):
        return (np.sqrt(self.D / self.cantilever.omega_c)).to('nm')
    
    @property
    def LD(self):
        return (np.sqrt((epsilon0 * kb * ureg.Quantity(300., 'K'))/(self.rho * qe * qe))).to('nm')

    @property
    def epsilon_eff(self):
        return (self.epsilon_s - complex(0,1) * self.Ld**2 / self.LD**2).to_base_units()

    def __repr__(self):

        str = self.cantilever.__repr__()
        str = str + '\nsemiconductor\n\n'
        str = str + '             epsilon (real) = {:0.3f}\n'.format(self.epsilon_s.to('').real.magnitude)
        str = str + '             epsilon (imag) = {:0.3f}\n'.format(self.epsilon_s.to('').imag.magnitude)
        str = str + '                  thickness = {:.3} nm\n'.format(self.h_s.to('nm').magnitude)
        str = str + '                   mobility = {:0.3e} m^2/(V s)\n'.format(self.mu.to('m^2/(V s)').magnitude)
        str = str + '         diffusion constant = {:0.3e} m^2/s\n'.format(self.D.to('m^2/s').magnitude)
        str = str + '             charge density = {:0.3e} m^{{-3}}\n'.format(self.rho.to('1/m^3').magnitude)
        str = str + '           reference height = {:0.3e} nm\n'.format(self.z_r.to('nm').magnitude)
        str = str + '               Debye length = {:0.3e} nm\n'.format(self.LD.to('nm').magnitude)
        str = str + '           diffusion length = {:0.3e} nm\n'.format(self.Ld.to('nm').magnitude)
        str = str + '   effective epsilon (real) = {:0.3f}\n'.format(self.epsilon_eff.to('').real.magnitude)
        str = str + '   effective epsilon (imag) = {:0.3f}\n\n'.format(self.epsilon_eff.to('').imag.magnitude)
        str = str + 'dielectric\n\n'
        str = str + '  epsilon (real) = {:0.3f}\n'.format(self.epsilon_d.to('').real.magnitude)
        str = str + '  epsilon (imag) = {:0.3f}\n'.format(self.epsilon_d.to('').imag.magnitude)
        str = str + '       thickness = infinite'
        
        return str

class SampleModel2(object):
    
    def __init__(self, cantilever, epsilon_d, h_d, epsilon_s, mu, rho, z_r):

        self.cantilever = cantilever
        self.epsilon_d = epsilon_d
        self.h_d = h_d
        self.epsilon_s = epsilon_s
        self.mu = mu
        self.rho = rho
        self.z_r = z_r

    @property
    def D(self):
        return ((kb * ureg.Quantity(300., 'K') * self.mu) / qe).to('m^2/s')

    @property
    def Ld(self):
        return (np.sqrt(self.D / self.cantilever.omega_c)).to('nm')
    
    @property
    def LD(self):
        return (np.sqrt((epsilon0 * kb * ureg.Quantity(300., 'K'))/(self.rho * qe * qe))).to('nm')

    @property
    def epsilon_eff(self):
        return (self.epsilon_s - complex(0,1) * self.Ld**2 / self.LD**2).to_base_units()

    def __repr__(self):

        str = self.cantilever.__repr__()
        str = str + '\ndielectric\n\n'
        str = str + '  epsilon (real) = {:0.3f}\n'.format(self.epsilon_d.to('').real.magnitude)
        str = str + '  epsilon (imag) = {:0.3f}\n'.format(self.epsilon_d.to('').imag.magnitude)
        str = str + '       thickness = {:.1f} nm\n\n'.format(self.h_d.to('nm').magnitude)
        str = str + 'semiconductor\n\n'
        str = str + '             epsilon (real) = {:0.3f}\n'.format(self.epsilon_s.to('').real.magnitude)
        str = str + '             epsilon (imag) = {:0.3f}\n'.format(self.epsilon_s.to('').imag.magnitude)
        str = str + '                  thickness = infinite\n'
        str = str + '                   mobility = {:0.2e} m^2/(V s)\n'.format(self.mu.to('m^2/(V s)').magnitude)
        str = str + '         diffusion constant = {:0.2e} m^2/s\n'.format(self.D.to('m^2/s').magnitude)
        str = str + '             charge density = {:0.2e} m^{{-3}}\n'.format(self.rho.to('1/m^3').magnitude)
        str = str + '           reference height = {:0.1f} nm\n'.format(self.z_r.to('nm').magnitude)
        str = str + '               Debye length = {:0.1f} nm\n'.format(self.LD.to('nm').magnitude)
        str = str + '           diffusion length = {:0.1f} nm\n'.format(self.Ld.to('nm').magnitude)
        str = str + '   effective epsilon (real) = {:0.3f}\n'.format(self.epsilon_eff.to('').real.magnitude)
        str = str + '   effective epsilon (imag) = {:0.3f}\n\n'.format(self.epsilon_eff.to('').imag.magnitude)

        return str

def mysech(x):
    
    x = np.array(x)
    mask = abs(x.real) < 710.4
    values = np.zeros_like(x, dtype=complex)
    values[mask] = 1/np.cosh(x[mask])
    
    return values

def mycsch(x):
    
    x = np.array(x)
    mask = abs(x.real) < 710.4
    values = np.zeros_like(x, dtype=complex)
    values[mask] = 1/np.sinh(x[mask])
    
    return values

def main():

    sample1 = SampleModel1(

        cantilever = CantileverModel(
            f_c = ureg.Quantity(81.01, 'kHz'), 
            V_ts = ureg.Quantity(3.01, 'V'), 
            R = ureg.Quantity(80.01, 'nm'), 
            d = ureg.Quantity(300.01, 'nm')),
        h_s = ureg.Quantity(1e6, 'nm'),
        epsilon_s = ureg.Quantity(complex(11.9, -0.05), ''),
        mu = ureg.Quantity(1e-5, 'm^2/(V s)'),
        rho = ureg.Quantity(1e21, '1/m^3'),
        epsilon_d = ureg.Quantity(complex(11.9, -0.05), ''),
        z_r = ureg.Quantity(100, 'nm')
    )

    print('\n=== sample1 ===')
    print(sample1)

    sample2 = SampleModel2(

        cantilever = CantileverModel(
            f_c = ureg.Quantity(81, 'kHz'), 
            V_ts = ureg.Quantity(3, 'V'), 
            R = ureg.Quantity(80, 'nm'), 
            d = ureg.Quantity(300, 'nm')),
        h_d = ureg.Quantity(0., 'nm'),
        epsilon_d = ureg.Quantity(complex(11.9, -0.05), ''),
        epsilon_s = ureg.Quantity(complex(11.9, -0.05), ''),
        mu = ureg.Quantity(2.7e-10, 'm^2/(V s)'),
        rho = ureg.Quantity(1e21, '1/m^3'),
        z_r = ureg.Quantity(300, 'nm')
    )

    print('\n=== sample2 ===')
    print(sample2)

if __name__ == "__main__":
    main()