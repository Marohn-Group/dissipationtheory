import numpy as np
import cmath
from dissipationtheory.constants import ureg, epsilon0, qe, kb
from dissipationtheory.capacitance import CsphereOverSemi
from scipy import integrate
from numba import jit
from numba import float64, complex128
from numba.experimental import jitclass
from numba import deferred_type

class CantileverModel(object):
    """Cantilever object.  
    SI units for each parameter are indicated below.
    A parameter can be given in any equivalent unit using the units package ``pint``.
    For example, :: 
    
        from dissipationtheory.constants import ureg
        R = ureg.Quantity(50., 'nm')
    
    :param f_c: cantilever frequency [Hz] 
    :type f_c: ureg.Quantity
    :param V_ts: tip-sample voltage [V]
    :type V_ts: ureg.Quantity
    :param R: tip radius [m]
    :type R: ureg.Quantity
    :param d: tip-sample separation [m]
    :type d: ureg.Quantity

    """
    def __init__(self, f_c, V_ts, R, d):

        self.f_c = f_c
        self.V_ts = V_ts
        self.R = R
        self.d = d

    @property
    def omega_c(self):
        """Cantilever resonance frequency in units of radians/second."""
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
    """Model I sample object defined in Lekkala2013nov, Fig. 1(b)::
    
        cantilever | vacuum gap | semiconductor | dielectric
    
    The dielectric substrate is semi-infinite.
    
    :param cantilever: an object storing the cantilever properties, including the tip-sample separation, i.e. the vacuum-gap thickness
    :type CantileverModel: cantilever
    :param epsilon_s: semiconductor layer's complex relative dielectric constant [unitless]
    :type epsilon_s: ureg.Quantity
    :param h_s: semiconductor layer's thickness [m]
    :type h_s: ureg.Quantity
    :param mu: semiconductor layer's charge mobility [m^2/Vs]
    :type mu: ureg.Quantity
    :param rho: semiconductor layer's charge density [1/m^3]
    :type rho: ureg.Quantity
    :param epsilon_d: dielectric layer's complex relative dielectric constant [unitless]
    :type epsilon_d: ureg.Quantity
    :param z_r: reference height [m] (used in computations)
    :type z_r: ureg.Quantity
    """

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
    """Model II sample object defined in Lekkala2013nov, Fig. 1(b)::
    
        cantilever | vacuum gap | dielectric | semiconductor 
    
    The semiconductor substrate is semi-infinite.
    
    :param cantilever: an object storing the cantilever properties, including the tip-sample separation, i.e. the vacuum-gap thickness
    :type CantileverModel: cantilever
    :param epsilon_d: dielectric layer's complex relative dielectric constant [unitless]
    :type epsilon_d: ureg.Quantity
    :param h_d: dielectric layer's thickness [m]
    :type h_d: ureg.Quantity
    :param epsilon_s: semiconductor layer's complex relative dielectric constant [unitless]
    :type epsilon_s: ureg.Quantity
    :param mu: semiconductor layer's charge mobility [m^2/Vs]
    :type mu: ureg.Quantity
    :param rho: semiconductor layer's charge density [1/m^3]
    :type rho: ureg.Quantity
    :param z_r: reference height [m] (used in computations)
    :type z_r: ureg.Quantity
    """

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
    """Define my own ``sech()`` function to avoid overflow problems."""

    x = np.array(x)
    mask = abs(x.real) < 710.4
    values = np.zeros_like(x, dtype=complex)
    values[mask] = 1/np.cosh(x[mask])
    
    return values

def mycsch(x):
    """Define my own ``csch()`` function to avoid overflow problems."""

    x = np.array(x)
    mask = abs(x.real) < 710.4
    values = np.zeros_like(x, dtype=complex)
    values[mask] = 1/np.sinh(x[mask])
    
    return values

def theta1norm(omega, sample, power):

    r1 = sample.Ld**2 / (sample.epsilon_s * sample.LD**2)
    r2 = sample.z_r**2 / (sample.epsilon_s * sample.LD**2)
    r3 = sample.z_r**2 / sample.Ld**2
    lambduh = (complex(0,1) * r1 * omega / np.sqrt(omega**2 + r2 + complex(0,1) * r3)).to('dimensionless').magnitude
     
    khs = (omega * sample.h_s / sample.z_r).to('dimensionless').magnitude

    r1 = sample.h_s**2 / sample.z_r**2
    r2 = sample.h_s**2 / (sample.epsilon_s * sample.LD**2)
    r3 = sample.h_s**2 / sample.Ld**2
    etahs = (np.sqrt(r1 * omega**2 + r2 + complex(0,1) * r3)).to('dimensionless').magnitude

    alpha = (sample.epsilon_eff/sample.epsilon_d).to('dimensionless').magnitude
    
    r1 = 1/sample.epsilon_eff
    t1 = - lambduh / np.tanh(etahs)
    t2 = np.tanh(khs) * np.tanh(etahs) \
         + alpha * np.tanh(etahs) \
         - lambduh \
         + 2 * lambduh * mysech(khs) * mysech(etahs) \
         - lambduh**2 * np.tanh(khs) * mysech(etahs) * mycsch(etahs)
    t3 = np.tanh(etahs) \
         + np.tanh(khs) * (-1 * lambduh + alpha * np.tanh(etahs))

    theta = (r1 * (t1 + t2 / t3)).to('dimensionless').magnitude

    ratio = (1 - theta) / (1 + theta)
    exponent = (2 * sample.cantilever.d / sample.z_r).to('dimensionless').magnitude

    integrand = omega**power * np.exp(-1 * omega * exponent) * np.imag(ratio)

    return integrand

def theta2norm(omega, sample, power):

    r1 = sample.Ld**2 / (sample.epsilon_s * sample.LD**2)
    r2 = sample.z_r**2 / (sample.epsilon_s * sample.LD**2)
    r3 = sample.z_r**2 / sample.Ld**2
    lambduh = (complex(0,1) * r1 * omega / np.sqrt(omega**2 + r2 + complex(0,1) * r3)).to('dimensionless').magnitude
     
    khd = (omega * sample.h_d / sample.z_r).to('dimensionless').magnitude

    r1 = 1/sample.epsilon_d
    t1 = sample.epsilon_eff * np.tanh(khd) + (1 - lambduh) * sample.epsilon_d
    t2 = sample.epsilon_eff + (1 - lambduh) * sample.epsilon_d * np.tanh(khd)

    theta = ((r1 * t1) / t2).magnitude

    ratio = (1 - theta) / (1 + theta)
    exponent = (2 * sample.cantilever.d / sample.z_r).to('dimensionless').magnitude

    integrand = omega**power * np.exp(-1 * omega * exponent) * np.imag(ratio) 

    return integrand

def C(power, theta, sample):
    
    prefactor = (-1**(power + 1) * kb * ureg.Quantity(300., 'K')) / \
          (4 * np.pi * epsilon0 * sample.cantilever.omega_c * sample.z_r**(power+1))
    
    integral = integrate.quad(theta, 0., np.inf, args=(sample, power))[0]
    
    return (prefactor * integral).to_base_units()

def gamma_parallel(theta, sample):
    """Compute :math:`\\gamma_{\\parallel}`."""

    prefactor = (sample.cantilever.V_ts**2 / (kb * ureg.Quantity(300., 'K'))).to('V/C')
    
    c0 = CsphereOverSemi(index=0, 
            height=sample.cantilever.d * np.ones(1), 
            radius=sample.cantilever.R, 
            epsilon=sample.epsilon_d.real.magnitude)

    return prefactor * 0.50 * c0 * c0 * C(2, theta, sample)

def gamma_perpendicular(theta, sample):
    """Compute :math:`\\gamma_{\\perp}`."""

    prefactor = (sample.cantilever.V_ts**2 / (kb * ureg.Quantity(300., 'K'))).to('V/C')

    c0 = CsphereOverSemi(0, sample.cantilever.d * np.ones(1), sample.cantilever.R, sample.epsilon_d.real.magnitude)        
    c1 = CsphereOverSemi(1, sample.cantilever.d * np.ones(1), sample.cantilever.R, sample.epsilon_d.real.magnitude)
    
    return prefactor * (c1 * c1 * C(0 , theta, sample) + 2 * c0 * c1 * C(1, theta, sample) + c0 * c0 * C(2, theta, sample))

def gamma_parallel_approx(rho, sample):
    """Low-density expansion for :math:`\\gamma_{\\parallel}`."""

    c0 = CsphereOverSemi(index=0, 
        height=sample.cantilever.d * np.ones(1), 
        radius=sample.cantilever.R, 
        epsilon=sample.epsilon_s.real.magnitude)
    
    qc = c0 * sample.cantilever.V_ts

    rho_x = ((sample.cantilever.omega_c * epsilon0)/(qe * sample.mu)).to('1/m^3').magnitude
    r2  = ((sample.epsilon_s.real**2 + sample.epsilon_s.imag**2)/sample.epsilon_s.real).to('dimensionless').magnitude
    rho2crit = r2 * rho_x
    
    gamma = ureg.Quantity(np.zeros(len(rho)), 'pN s/m')
    mask = np.zeros(len(rho), dtype=bool)
    
    for index, rho_ in enumerate(rho):

        sample.rho = rho_

        if (rho_.to('1/m^3').magnitude <= rho2crit):

            t1 = - sample.epsilon_s.imag/(np.abs(sample.epsilon_s + 1)**2)
            t2 = (qc * qc)/(16 * np.pi * epsilon0 * sample.cantilever.omega_c * sample.cantilever.d**3)

            t3 =  (qc * qc)/(16 * np.pi * epsilon0 * sample.cantilever.omega_c**2)
            t4 = ((1 + sample.epsilon_s.real)**2 - (sample.epsilon_s.imag)**2)/(np.abs(sample.epsilon_s + 1)**4)
            t5 = sample.D/(sample.LD**2 * sample.cantilever.d**3)

            gamma[index] = t1 * t2 + t3 * t4 * t5
            mask[index] = True
    
    return rho.to('1/m^3')[mask], gamma.to('pN s/m')[mask]

def gamma_perpendicular_approx(rho, sample):
    """Low-density expansion for :math:`\\gamma_{\\perp}`."""

    c0 = CsphereOverSemi(index=0, 
        height=sample.cantilever.d * np.ones(1), 
        radius=sample.cantilever.R, 
        epsilon=sample.epsilon_s.real.magnitude)

    c1 = CsphereOverSemi(index=1, 
        height=sample.cantilever.d * np.ones(1), 
        radius=sample.cantilever.R, 
        epsilon=sample.epsilon_s.real.magnitude)
    
    rho_x = ((sample.cantilever.omega_c * epsilon0)/(qe * sample.mu)).to('1/m^3').magnitude
    r2  = ((sample.epsilon_s.real**2 
            + sample.epsilon_s.imag**2)/sample.epsilon_s.real).to('dimensionless').magnitude
    rho2crit = r2 * rho_x

    gamma = ureg.Quantity(np.zeros(len(rho)), 'pN s/m')
    mask = np.zeros(len(rho), dtype=bool)

    for index, rho_ in enumerate(rho):

        sample.rho = rho_

        if (rho_.to('1/m^3').magnitude <= rho2crit):

            t1 = - sample.epsilon_s.imag/(np.abs(sample.epsilon_s + 1)**2)
            t2 = ((1 + sample.epsilon_s.real)**2 - (sample.epsilon_s.imag)**2)/(np.abs(sample.epsilon_s + 1)**4)
            t3 = sample.Ld**2/sample.LD**2
            t4= sample.cantilever.V_ts**2/(4 * np.pi * epsilon0 * sample.cantilever.omega_c)
            d = sample.cantilever.d
            t5 = c1**2/d - (c0 * c1)/d**2 + (0.5 * c0**2)/d**3

            gamma[index] = (t1 + t2 * t3) * t4 * t5
            mask[index] = True
    
    return rho.to('1/m^3')[mask], gamma.to('pN s/m')[mask]

CantileverModelSpec = [
    ('f_c',float64),
    ('V_ts',float64),
    ('R',float64),
    ('d',float64)]  # <== can't use pint, so no units :-(

@jitclass(CantileverModelSpec)
class CantileverModelJit(object):

    def __init__(self, f_c, V_ts, R, d):

        self.f_c = f_c
        self.V_ts = V_ts
        self.R = R
        self.d = d

    @property
    def omega_c(self):
        return 2 * np.pi * self.f_c

    def print(self):  # <=== can't use the __repr__ function :-(

        print("   cantilever freq = ", self.f_c, "Hz") # <== can't use formatting strings here
        print("                   = ", self.omega_c, "rad/s")
        print("tip-sample voltage = ", self.V_ts, "V")
        print("            radius = ", self.R, "m")
        print("            height = ", self.d, "m")

CantileverModelJit_type = deferred_type()
CantileverModelJit_type.define(CantileverModelJit.class_type.instance_type)

kb_= kb.to('J/K').magnitude
T_ = 300.
qe_ = qe.to('C').magnitude
epsilon0_ = epsilon0.to('C/(V m)').magnitude

SampleModel1Spec = [
    ('cantilever', CantileverModelJit_type),
    ('epsilon_s',complex128),
    ('h_s', float64),
    ('mu',float64),
    ('rho',float64),
    ('epsilon_d',complex128),
    ('z_r',float64)]

@jitclass(SampleModel1Spec)
class SampleModel1Jit(object):
    
    def __init__(self, cantilever, epsilon_s, h_s, mu, rho, epsilon_d, z_r):

        self.cantilever = cantilever
        self.epsilon_s = epsilon_s
        self.h_s = h_s
        self.mu = mu
        self.rho = rho
        self.epsilon_d = epsilon_d
        self.z_r = z_r

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
        print("                mobility = ", self.mu, "m^2/(V s)")
        print("      diffusion constant = ", self.D, "m^2/s")
        print("          charge density = ", self.rho, "m^{{-3}}")
        print("        reference height = ", self.z_r, "m")
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
    ('mu',float64),
    ('rho',float64),
    ('z_r',float64)]

@jitclass(SampleModel2Spec)
class SampleModel2Jit(object):
    
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
        print("                mobility = ", self.mu, "m^2/(V s)")
        print("      diffusion constant = ", self.D, "m^2/s")
        print("          charge density = ", self.rho, "m^{{-3}}")
        print("        reference height = ", self.z_r, "m")
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
    mu=2.7E-10,
    rho=1e21,
    epsilon_d=complex(11.9,-0.05),
    z_r=300E-9)

sample2 = SampleModel2Jit(
    cantilever=CantileverModelJit(81.0e3, 3.,  80E-9, 300E-9),
    epsilon_d=complex(11.9,-0.05),
    h_d=0.,
    epsilon_s=complex(11.9,-0.05),
    mu=2.7E-10,
    rho=1e21,
    z_r=300E-9)

@jit(complex128(complex128), nopython=True)
def mysech_jit(x):
    """Define my own just-in-time-compiled ``sech()`` function to avoid overflow problems."""
    if x.real < 710.4:
        return 1/np.cosh(x)
    else:
        return complex(0., 0.)

@jit(complex128(complex128), nopython=True)
def mycsch_jit(x):
    """Define my own just-in-time-compiled ``csch()`` function to avoid overflow problems."""
    if x.real < 710.4:
        return 1/np.sinh(x)
    else:
        return complex(0., 0.,)

# https://github.com/numba/numba/issues/2922
# Inaccurate complex tanh implementation #2922
# "Numba complex tanh implementation returns nans for large real values, 
# probably due to some intermediate under/overflow."
#
# Replace np.tanh(a) with cmath.tanh(a), but take care to make the argument
# a is complex by replacing a -> a * complex(1,0) when a is real.
#

@jit(float64(float64,SampleModel1Jit.class_type.instance_type,float64), nopython=True)
def theta1norm_jit(omega, sample1, power):

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

    integrand = omega**power * np.exp(-1 * omega * exponent) * np.imag(ratio)

    return integrand    

@jit(float64(float64,SampleModel2Jit.class_type.instance_type,float64), nopython=True)
def theta2norm_jit(omega, sample2, power):

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

    integrand = omega**power * np.exp(-1 * omega * exponent) * np.imag(ratio) 
    
    return integrand

def C_jit(power, theta, sample):
    
    prefactor = (-1**(power + 1) * kb * ureg.Quantity(300., 'K')) / \
          (4 * np.pi * epsilon0 * ureg.Quantity(sample.cantilever.omega_c,'1/s') * ureg.Quantity(sample.z_r,'m')**(power+1))
    
    integral = integrate.quad(theta, 0., np.inf, args=(sample, power))[0]
    
    return (prefactor * integral).to_base_units()

def gamma_parallel_jit(theta, sample):

    prefactor = (ureg.Quantity(sample.cantilever.V_ts,'V')**2 / (kb * ureg.Quantity(300., 'K'))).to('V/C')
    
    c0 = CsphereOverSemi(index=0, 
            height=ureg.Quantity(sample.cantilever.d,'m') * np.ones(1), 
            radius=ureg.Quantity(sample.cantilever.R,'m'), 
            epsilon=sample.epsilon_d.real)

    return prefactor * 0.50 * c0 * c0 * C_jit(2, theta, sample)

def gamma_perpendicular_jit(theta, sample):

    prefactor = (ureg.Quantity(sample.cantilever.V_ts,'V')**2 / (kb * ureg.Quantity(300., 'K'))).to('V/C')

    c0 = CsphereOverSemi(index=0, 
        height=ureg.Quantity(sample.cantilever.d,'m') * np.ones(1), 
        radius=ureg.Quantity(sample.cantilever.R,'m'),
        epsilon=sample.epsilon_d.real)
        
    c1 = CsphereOverSemi(index=1, 
        height=ureg.Quantity(sample.cantilever.d,'m') * np.ones(1), 
        radius=ureg.Quantity(sample.cantilever.R,'m'),
        epsilon=sample.epsilon_d.real)
    
    t1 = c1 * c1 * C_jit(0 , theta, sample)
    t2 = 2 * c0 * c1 * C_jit(1, theta, sample)
    t3 = c0 * c0 * C_jit(2, theta, sample)

    return prefactor * (t1 + t2 + t3)

def gamma_perpendicular_fit(x, a, R, fc):
    """Compute the friction experienced by a cantilever oscillating in the perpendicular orientation 
    over a conductive substrate, in the low-conductivity limit.  The function computes the tip-sample
    capacitance assuming the sample is metallic.  Because the inputs and outputs of this function are 
    unitless, the function is suitable for curve fitting.
    
    inputs:
        x (float): tip-sample separation [nm]
        a (float): sample-conductivity parameter [unitless]
        R (float): tip radius [nm]
        fc (float): cantilever frequency [Hz]

    output:
        voltage-normalized cantilever dissipation constant [pN s/(V^2 m)]
    """

    h = ureg.Quantity(x, 'nm')
    r = ureg.Quantity(R, 'nm')
    
    c0 = Csphere(0, height=h, radius=r, nterm=21)
    c1 = Csphere(1, height=h, radius=r, nterm=21)
    
    d = h + r
    
    t1 = (c0 * c0)/(2 * d**3)
    t2 = - (c0*c1)/d**2
    t3 = (c1 * c1)/d
    
    omega = 2 * np.pi * ureg.Quantity(fc, '1/s')
    prefactor = 1/(4 * np.pi * epsilon0 * omega)
    expression = (prefactor * (t1 + t2 + t3)).to('pN s/(V^2 m)').magnitude
    
    return a * expression

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