import numpy as np
import cmath
from dissipationtheory.constants import ureg, epsilon0, qe, kb
from dissipationtheory.capacitance import Csphere
from dissipationtheory.capacitance import CsphereOverSemi
from dissipationtheory.dissipation import CantileverModel, CantileverModelJit_type, CantileverModelJit
from dissipationtheory.dissipation import kb_, T_, qe_, epsilon0_ 
from dissipationtheory.dissipation import mysech_jit, mycsch_jit 

from scipy import integrate
from numba import jit
from numba import float64, complex128, boolean
from numba.experimental import jitclass
from numba import deferred_type

class SampleModel1(object):
    """Model I sample object defined in Lekkala2013nov, Fig. 1(b)::
    
        cantilever | vacuum gap | semiconductor | dielectric
    
    The dielectric substrate is semi-infinite.
    
    :param CantileverModel cantilever: an object storing the cantilever properties, including the tip-sample separation, i.e. the vacuum-gap thickness
    :param ureg.Quantity epsilon_s: semiconductor layer's complex relative dielectric constant [unitless]
    :param ureg.Quantity h_s: semiconductor layer's thickness [m]
    :param ureg.Quantity sigma: semiconductor layer's conductivity [S/m]
    :param ureg.Quantity rho: semiconductor layer's charge density [1/m^3]
    :param ureg.Quantity epsilon_d: dielectric layer's complex relative dielectric constant [unitless]
    :param ureg.Quantity z_r: reference height [m] (used in computations)
    """

    def __init__(self, cantilever, h_s, epsilon_s, sigma, rho, epsilon_d, z_r):

        self.cantilever = cantilever
        self.epsilon_s = epsilon_s
        self.h_s = h_s
        self.sigma = sigma
        self.rho = rho
        self.epsilon_d = epsilon_d
        self.z_r = z_r

    @property
    def omega0(self):
        """Omega0, the roll-off frequency."""
        return (self.sigma / epsilon0).to('Hz')

    @property
    def mu(self):
        """Mobility."""
        return (self.sigma / (qe * self.rho)).to('m^2/(V s)')

    @property
    def D(self):
        """Diffusion constant."""
        return ((kb * ureg.Quantity(300., 'K') * self.mu) / qe).to('m^2/s')

    @property
    def Ld(self):
        """Diffusion length."""
        return (np.sqrt(self.D / self.cantilever.omega_c)).to('nm')
    
    @property
    def LD(self):
        """Debye length."""
        return (np.sqrt((epsilon0 * kb * ureg.Quantity(300., 'K'))/(self.rho * qe * qe))).to('nm')

    @property
    def epsilon_eff(self):
        """Effective relative complex dielectric constant."""
        return (self.epsilon_s - complex(0,1) * self.Ld**2 / self.LD**2).to_base_units()

    def __repr__(self):

        str = self.cantilever.__repr__()
        str = str + '\nsemiconductor\n\n'
        str = str + '             epsilon (real) = {:0.3f}\n'.format(self.epsilon_s.to('').real.magnitude)
        str = str + '             epsilon (imag) = {:0.3f}\n'.format(self.epsilon_s.to('').imag.magnitude)
        str = str + '                  thickness = {:.3} nm\n'.format(self.h_s.to('nm').magnitude)
        str = str + '               conductivity = {:0.3e} S/m\n'.format(self.sigma.to('S/m').magnitude)
        str = str + '             charge density = {:0.3e} m^{{-3}}\n'.format(self.rho.to('1/m^3').magnitude)
        str = str + '           reference height = {:0.3e} nm\n'.format(self.z_r.to('nm').magnitude)
        str = str + '\n'
        str = str + '        roll-off frequency  = {:0.3e} Hz\n'.format(self.omega0.to('Hz').magnitude)
        str = str + '                   mobility = {:0.3e} m^2/(V s)\n'.format(self.mu.to('m^2/(V s)').magnitude)
        str = str + '         diffusion constant = {:0.3e} m^2/s\n'.format(self.D.to('m^2/s').magnitude)                
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
    """Model II sample object defined in Lekkala2013nov, Fig. 1(b)::
    
        cantilever | vacuum gap | dielectric | semiconductor 
    
    The semiconductor substrate is semi-infinite.
    
    :param CantileverModel cantilever: an object storing the cantilever properties, including the tip-sample separation, i.e. the vacuum-gap thickness
    :param ureg.Quantity epsilon_d: dielectric layer's complex relative dielectric constant [unitless]
    :param ureg.Quantity h_d: dielectric layer's thickness [m]
    :param ureg.Quantity epsilon_s: semiconductor layer's complex relative dielectric constant [unitless]
    :param ureg.Quantity mu: semiconductor layer's charge conductivity [S/m]
    :param ureg.Quantity rho: semiconductor layer's charge density [1/m^3]
    :param ureg.Quantity z_r: reference height [m] (used in computations)
    """

    def __init__(self, cantilever, epsilon_d, h_d, epsilon_s, sigma, rho, z_r):

        self.cantilever = cantilever
        self.epsilon_d = epsilon_d
        self.h_d = h_d
        self.epsilon_s = epsilon_s
        self.sigma = sigma
        self.rho = rho
        self.z_r = z_r

    @property
    def omega0(self):
        """Omega0, the roll-off frequency."""
        return (self.sigma / epsilon0).to('Hz')

    @property
    def mu(self):
        """Mobility."""
        return (self.sigma / (qe * self.rho)).to('m^2/(V s)')

    @property
    def D(self):
        """Diffusion constant."""
        return ((kb * ureg.Quantity(300., 'K') * self.mu) / qe).to('m^2/s')

    @property
    def Ld(self):
        """Diffusion length."""
        return (np.sqrt(self.D / self.cantilever.omega_c)).to('nm')
    
    @property
    def LD(self):
        """Debye length."""
        return (np.sqrt((epsilon0 * kb * ureg.Quantity(300., 'K'))/(self.rho * qe * qe))).to('nm')

    @property
    def epsilon_eff(self):
        """Effective relative complex dielectric constant."""
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
        str = str + '               conductivity = {:0.3e} S/m\n'.format(self.sigma.to('S/m').magnitude)
        str = str + '             charge density = {:0.2e} m^{{-3}}\n'.format(self.rho.to('1/m^3').magnitude)
        str = str + '           reference height = {:0.1f} nm\n'.format(self.z_r.to('nm').magnitude)
        str = str + '\n'
        str = str + '         roll-off frequency = {:0.3e} Hz\n'.format(self.omega0.to('Hz').magnitude)        
        str = str + '                   mobility = {:0.2e} m^2/(V s)\n'.format(self.mu.to('m^2/(V s)').magnitude)
        str = str + '         diffusion constant = {:0.2e} m^2/s\n'.format(self.D.to('m^2/s').magnitude)
        str = str + '               Debye length = {:0.1f} nm\n'.format(self.LD.to('nm').magnitude)
        str = str + '           diffusion length = {:0.1f} nm\n'.format(self.Ld.to('nm').magnitude)
        str = str + '   effective epsilon (real) = {:0.3f}\n'.format(self.epsilon_eff.to('').real.magnitude)
        str = str + '   effective epsilon (imag) = {:0.3f}\n\n'.format(self.epsilon_eff.to('').imag.magnitude)

        return str

SampleModel1Spec = [
    ('cantilever', CantileverModelJit_type),
    ('epsilon_s',complex128),
    ('h_s', float64),
    ('sigma',float64),
    ('rho',float64),
    ('epsilon_d',complex128),
    ('z_r',float64)]

@jitclass(SampleModel1Spec)
class SampleModel1Jit(object):
    
    def __init__(self, cantilever, epsilon_s, h_s, sigma, rho, epsilon_d, z_r):

        self.cantilever = cantilever
        self.epsilon_s = epsilon_s
        self.h_s = h_s
        self.sigma = sigma
        self.rho = rho
        self.epsilon_d = epsilon_d
        self.z_r = z_r

    @property
    def omega0(self):
        return self.sigma / epsilon0_

    @property
    def mu(self):
        return self.sigma / (qe_ * self.rho)
    
    @property
    def D(self):
        return (kb_ * T_ * self.mu) / qe_

    @property
    def Ld(self):
        return np.sqrt(self.D / self.cantilever.omega_c)
    
    @property
    def LD(self):
        return np.sqrt((epsilon0_ * kb_ * T_)/(self.rho * qe_ * qe_))

    @property
    def epsilon_eff(self):
        return self.epsilon_s - complex(0,1) * self.Ld**2 / self.LD**2
    
    def print(self):

        print("cantilever")
        print("==========")
        self.cantilever.print()
        print("")
        print("semiconductor")
        print("=============")
        print("          epsilon (real) = ", self.epsilon_s.real)
        print("          epsilon (imag) = ", self.epsilon_s.imag)
        print("               thickness = ", self.h_s, "m")
        print("            conductivity = ", self.sigma, "S/m")
        print("          charge density = ", self.rho, "m^{{-3}}")
        print("        reference height = ", self.z_r, "m")
        print(" ")
        print("      roll-off frequency = ", self.omega0, "Hz")        
        print("                mobility = ", self.mu, "m^2/(V s)")
        print("      diffusion constant = ", self.D, "m^2/s")
        print("            Debye length = ", self.LD, "m")
        print("        diffusion length = ", self.Ld, "m")
        print("effective epsilon (real) = ", self.epsilon_eff.real)
        print("effective epsilon (imag) = ", self.epsilon_eff.imag)
        print("")
        print("dielectric")
        print("==========")
        print(" epsilon (real) = ", self.epsilon_d.real)
        print(" epsilon (imag) = ", self.epsilon_d.imag)
        print("      thickness = infinite")

SampleModel2Spec = [
    ('cantilever', CantileverModelJit_type),
    ('epsilon_d',complex128),
    ('h_d', float64),
    ('epsilon_s',complex128),
    ('sigma',float64),
    ('rho',float64),
    ('z_r',float64)]

@jitclass(SampleModel2Spec)
class SampleModel2Jit(object):
    
    def __init__(self, cantilever, epsilon_d, h_d, epsilon_s, sigma, rho, z_r):

        self.cantilever = cantilever
        self.epsilon_d = epsilon_d
        self.h_d = h_d
        self.epsilon_s = epsilon_s
        self.sigma = sigma
        self.rho = rho
        self.z_r = z_r

    @property
    def omega0(self):
        return self.sigma / epsilon0_

    @property
    def mu(self):
        return self.sigma / (qe_ * self.rho)

    @property
    def D(self):
        return (kb_ * T_ * self.mu) / qe_

    @property
    def Ld(self):
        return np.sqrt(self.D / self.cantilever.omega_c)
    
    @property
    def LD(self):
        return np.sqrt((epsilon0_ * kb_ * T_)/(self.rho * qe_ * qe_))

    @property
    def epsilon_eff(self):
        return self.epsilon_s - complex(0,1) * self.Ld**2 / self.LD**2

    def print(self):

        print("cantilever")
        print("==========")
        self.cantilever.print()
        print("")
        print("dielectric")
        print("==========")
        print(" epsilon (real) = ", self.epsilon_d.real)
        print(" epsilon (imag) = ", self.epsilon_d.imag)
        print("      thickness = ", self.h_d, "m")
        print("")
        print("semiconductor")
        print("=============")
        print("          epsilon (real) = ", self.epsilon_s.real)
        print("          epsilon (imag) = ", self.epsilon_s.imag)
        print("               thickness = infinite")
        print("            conductivity = ", self.sigma, "S/m")        
        print("                mobility = ", self.mu, "m^2/(V s)")
        print("        reference height = ", self.z_r, "m")
        print(" ")
        print("      roll-off frequency = ", self.omega0, "Hz")
        print("      diffusion constant = ", self.D, "m^2/s")
        print("          charge density = ", self.rho, "m^{{-3}}")
        print("            Debye length = ", self.LD, "m")
        print("        diffusion length = ", self.Ld, "m")
        print("effective epsilon (real) = ", self.epsilon_eff.real)
        print("effective epsilon (imag) = ", self.epsilon_eff.imag)

# To compile the theta1norm_jit() and theta2norm_jit() functions, jit needs an actual 
#    instance of the function input "sample". 

sample1 = SampleModel1Jit(
    cantilever=CantileverModelJit(81.0e3, 3.,  80E-9, 300E-9),
    epsilon_s=complex(11.9,-0.05),
    h_s=3000E-9,
    sigma = 1e-8,
    rho=1e21,
    epsilon_d=complex(11.9,-0.05),
    z_r=300E-9)

sample2 = SampleModel2Jit(
    cantilever=CantileverModelJit(81.0e3, 3.,  80E-9, 300E-9),
    epsilon_d=complex(11.9,-0.05),
    h_d=0.,
    epsilon_s=complex(11.9,-0.05),
    sigma = 1e-8,
    rho=1e21,
    z_r=300E-9)

# Recompile, since the underlying data structure is new.

@jit(float64(float64,SampleModel1Jit.class_type.instance_type,float64,boolean), nopython=True)
def theta1norm_jit(omega, sample1, power, isImag):

    r1 = sample1.Ld**2 / (sample1.epsilon_s * sample1.LD**2)
    r2 = sample1.z_r**2 / (sample1.epsilon_s * sample1.LD**2)
    r3 = sample1.z_r**2 / sample1.Ld**2

    lambduh = complex(0,1) * r1 * omega / np.sqrt(omega**2 + r2 + complex(0,1) * r3)
 
    khs = complex(1,0) * omega * sample1.h_s / sample1.z_r

    r1 = sample1.h_s**2 / sample1.z_r**2
    r2 = sample1.h_s**2 / (sample1.epsilon_s * sample1.LD**2)
    r3 = sample1.h_s**2 / sample1.Ld**2

    etahs = complex(1,0) * np.sqrt(r1 * omega**2 + r2 + complex(0,1) * r3)

    alpha = sample1.epsilon_eff/sample1.epsilon_d
 
    r1 = 1/sample1.epsilon_eff
    t1 = - lambduh / cmath.tanh(etahs)
    t2 = cmath.tanh(khs) * cmath.tanh(etahs) \
         + alpha * cmath.tanh(etahs) \
         - lambduh \
         + 2 * lambduh * mysech_jit(khs) * mysech_jit(etahs) \
         - lambduh**2 * cmath.tanh(khs) * mysech_jit(etahs) * mycsch_jit(etahs)
    t3 = cmath.tanh(etahs) \
         + cmath.tanh(khs) * (-1 * lambduh + alpha * cmath.tanh(etahs))

    theta = r1 * (t1 + t2 / t3)

    ratio = (1 - theta) / (1 + theta)
    exponent = 2 * sample1.cantilever.d / sample1.z_r

    if isImag:
        integrand = omega**power * np.exp(-1 * omega * exponent) * np.imag(ratio)
    else:
        integrand = omega**power * np.exp(-1 * omega * exponent) * np.real(ratio)

    return integrand    

@jit(float64(float64,SampleModel2Jit.class_type.instance_type,float64,boolean), nopython=True)
def theta2norm_jit(omega, sample2, power, isImag):

    r1 = sample2.Ld**2 / (sample2.epsilon_s * sample2.LD**2)
    r2 = sample2.z_r**2 / (sample2.epsilon_s * sample2.LD**2)
    r3 = sample2.z_r**2 / sample2.Ld**2

    lambduh = complex(0,1) * r1 * omega / np.sqrt(omega**2 + r2 + complex(0,1) * r3)
     
    khd = complex(1,0) * omega * sample2.h_d / sample2.z_r

    r1 = 1/sample2.epsilon_d
    t1 = sample2.epsilon_eff * cmath.tanh(khd) + (1 - lambduh) * sample2.epsilon_d
    t2 = sample2.epsilon_eff + (1 - lambduh) * sample2.epsilon_d * cmath.tanh(khd)

    theta = (r1 * t1) / t2

    ratio = (1 - theta) / (1 + theta)
    exponent = 2 * sample2.cantilever.d / sample2.z_r

    if isImag:
        integrand = omega**power * np.exp(-1 * omega * exponent) * np.imag(ratio) 
    else:
        integrand = omega**power * np.exp(-1 * omega * exponent) * np.real(ratio)

    return integrand

def main():

    sample1 = SampleModel1(

        cantilever = CantileverModel(
            f_c = ureg.Quantity(81.01, 'kHz'), 
            V_ts = ureg.Quantity(3.01, 'V'), 
            R = ureg.Quantity(80.01, 'nm'), 
            d = ureg.Quantity(300.01, 'nm')),
        h_s = ureg.Quantity(1e6, 'nm'),
        epsilon_s = ureg.Quantity(complex(11.9, -0.05), ''),
        sigma = ureg.Quantity(1E-8, 'S/m'),
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
        sigma = ureg.Quantity(1E-8, 'S/m'),
        rho = ureg.Quantity(1e21, '1/m^3'),
        z_r = ureg.Quantity(300, 'nm')
    )

    print('\n=== sample2 ===')
    print(sample2)

if __name__ == "__main__":
    main()