# Author: John A. Marohn
# Date: 2025-06-05
# Summary: numba/jit functions from dissipation7.py

import numpy as np
import cmath
from dissipationtheory.constants import ureg, epsilon0, qe, kb
from scipy import integrate
import scipy
import pint
from numba import jit
import numba.types as nb_types
from numba import float64, complex128, boolean
from numba.experimental import jitclass
from numba import deferred_type

CantileverModelSpec = [
    ('f_c', float64),
    ('k_c', float64),
    ('V_ts', float64),
    ('R', float64),
    ('angle', float64),
    ('L', float64),
    ('d', float64)] 

@jitclass(CantileverModelSpec)
class CantileverModelJit(object):

    def __init__(self, f_c, k_c, V_ts, R, angle, L, d):

        self.f_c = f_c
        self.k_c = k_c
        self.V_ts = V_ts
        self.R = R
        self.angle = angle
        self.L = L
        self.d = d

    @property
    def omega_c(self):
        return 2 * np.pi * self.f_c

    def print(self):  # <=== can't use the __repr__ function :-(

        print("        cantilever freq = ", self.f_c, "Hz") # <== can't use formatting strings here
        print("                        = ", self.omega_c, "rad/s")
        print("        spring constant = ", self.k_c, "N/m")
        print("     tip-sample voltage = ", self.V_ts, "V")
        print("                 radius = ", self.R, "m")
        print("        cone half angle = ", self.angle, "degree")
        print("            cone length = ", self.L, "m")
        print("  tip-sample separation = ", self.d, "m")

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
    def kD(self):
        return np.sqrt((self.rho * qe_ * qe_)/(epsilon0_ * kb_ * T_))
    
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
        print("    inverse Debye length = ", self.kD, "m^{{-1}}")
        print("            Debye length = ", 1/self.kD, "m")
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
    def kD(self):
        return np.sqrt((self.rho * qe_ * qe_)/(epsilon0_ * kb_ * T_))

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
        print("          charge density = ", self.rho, "m^{{-3}}")
        print("        reference height = ", self.z_r, "m")
        print(" ")
        print("      roll-off frequency = ", self.omega0, "Hz")
        print("    inverse Debye length = ", self.kD, "m^{{-1}}")
        print("            Debye length = ", 1/self.kD, "m")

# To compile the integrand1_jit() and integrand2_jit() functions, jit needs an actual 
#    instance of the function input "sample". 

sample1 = SampleModel1Jit(
    cantilever=CantileverModelJit(75.0e3, 2.8, 1.0, 35.0E-9, 20.0, 1000.0E-9, 38.0E-9),
    epsilon_s=complex(11.9,-0.05),
    h_s=3000E-9,
    sigma = 1e-8,
    rho=1e21,
    epsilon_d=complex(11.9,-0.05),
    z_r=300E-9)

sample2 = SampleModel2Jit(
    cantilever=CantileverModelJit(75.0e3, 2.8, 1.0, 35.0E-9, 20.0, 1000.0E-9, 38.0E-9),
    epsilon_d=complex(11.9,-0.05),
    h_d=0.,
    epsilon_s=complex(11.9,-0.05),
    sigma = 1e-8,
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

# Can bessel functions (from scipy.special) be used with Numba?
# https://stackoverflow.com/questions/46499816/can-bessel-functions-from-scipy-special-be-used-with-numba
# https://github.com/numba/numba-scipy
#
# To implement the bessel function in a jit-compiled function, 
# the numba-scipy package is required. See README.md for the (painful) upgrade procedure.
# I ended up having to reinstall poetry!

@jit(float64(float64,
             float64,
             SampleModel1Jit.class_type.instance_type,
             float64,
             nb_types.float64[::1],
             nb_types.float64[::1],
             boolean), nopython=True)

def integrand1jit(y, power, sample, omega, location1, location2, isImag):
    """Theta function for the Sample I geometry of Lekkala.
    
    In the code below, `y` is the unitless integration variable."""

    es = sample.epsilon_s
    ed = sample.epsilon_d
    zr = sample.z_r
    hs = sample.h_s

    Omega = omega/sample.omega0
    theta1 = complex(1,0) * np.sqrt(y**2 * (hs / zr)**2 + (hs * sample.kD)**2 * (1/es + complex(0,1) * Omega))  # depends on y
    theta2 = complex(1,0) * y * hs / zr                                                                         # depends on y

    k_over_eta = y / np.sqrt(y**2 + (zr * sample.kD)**2 * (1/es + complex(0,1) * Omega)) # depends on y

    p0 = 1 + complex(0,1) * es * Omega
    p1 = k_over_eta / (es * p0)                           # depends on y
    p2 = - Omega**2 / (p0 * p0)
    p3 = complex(0,1) * Omega / (ed * p0)
    p4 = complex(0,1) * Omega * k_over_eta / (es * p0**2) # depends on y
    p5 = - k_over_eta**2 / (es**2 * p0**2)                # depends on y
    p6 = complex(0,1) * Omega / p0
    p7 = 1 / ed

    t1 = p1 / cmath.tanh(theta1)
    n1 = p2 * cmath.tanh(theta1) * cmath.tanh(theta2) + p3 * cmath.tanh(theta1) + p4
    n2 = - 2 * p4 * mysech_jit(theta2) * mysech_jit(theta1) 
    n3 = p5 * cmath.tanh(theta2) * mysech_jit(theta1) * mycsch_jit(theta1)
    d1 = p6 * cmath.tanh(theta1) + cmath.tanh(theta2) * (p1 + p7 * cmath.tanh(theta1))
    
    theta_norm = t1 + (n1 + n2 + n3)/d1
    rp = (1 - theta_norm) / (1 + theta_norm)

    rhoX = (location1[0] - location2[0])/ zr
    rhoY = (location1[1] - location2[1])/ zr
    argument = y * np.sqrt(rhoX**2 + rhoY**2)
    exponent = y * (location1[2] + location2[2])/ zr

    if isImag:
        integrand = y**power * scipy.special.j0(argument) * np.exp(-1 * exponent) * np.imag(rp) 
    else:
        integrand = y**power * scipy.special.j0(argument) * np.exp(-1 * exponent) * np.real(rp) 

    return integrand

@jit(float64(float64,
             float64,
             SampleModel2Jit.class_type.instance_type,
             float64,
             nb_types.float64[::1],
             nb_types.float64[::1],
             boolean), nopython=True)

def integrand2jit(y, power, sample, omega, location1, location2, isImag):
    """Theta function for the Sample II geometry of Lekkala.
    
    In the code below, `y` is the unitless integration variable."""

    es = sample.epsilon_s
    zr = sample.z_r

    Omega = omega/sample.omega0
    theta2 = complex(1,0) * y * sample.h_d / zr # depends on y

    k_over_eta = y / np.sqrt(y**2 + (zr * sample.kD)**2 * (1/es + complex(0,1) * Omega)) # depends on y

    p0 = 1 + complex(0,1) * es * Omega
    p1 = k_over_eta / (es * p0)         # depends on y
    p6 = complex(0,1) * Omega / p0
    p7 = 1 / sample.epsilon_d

    theta_norm = p7 * (p7 * cmath.tanh(theta2) + p6 + p1)/(p7 + (p6 + p1) * cmath.tanh(theta2))
    rp = (1 - theta_norm) / (1 + theta_norm)

    rhoX = (location1[0] - location2[0])/ zr
    rhoY = (location1[1] - location2[1])/ zr
    argument = y * np.sqrt(rhoX**2 + rhoY**2)
    exponent = y * (location1[2] + location2[2])/ zr

    if isImag:
        integrand = y**power * scipy.special.j0(argument) * np.exp(-1 * exponent) * np.imag(rp) 
    else:
        integrand = y**power * scipy.special.j0(argument) * np.exp(-1 * exponent) * np.real(rp) 

    return integrand

# Can't jit this because it outputs an answer with units

def K_jit(integrand, power, sample, omega, location1, location2, isImag):
    """Compute the integral :math:`K`.  The answer is returned with units. """
    
    prefactor = 1/np.power(ureg.Quantity(sample.z_r, 'm'), power+1)

    if isImag:
        integral = integrate.quad(integrand, 0., np.inf, args=(power, sample, omega, location1, location2, True))[0]
    else:
        integral = integrate.quad(integrand, 0., np.inf, args=(power, sample, omega, location1, location2, False))[0]
  
    return (prefactor * integral).to_base_units()

def gamma_perpendicular_jit(integrand, sample, location):
    """Compute the friction for a cantilever oscillating in the perpendicular 
    geometry using equation 33 in Roger Loring's 2025-06-05 document."""

    # Shorthand

    wc = sample.cantilever.omega_c

    # The parameters sample_jit object lacks units, so add them back in manually

    R = ureg.Quantity(sample.cantilever.R, 'm')
    V = ureg.Quantity(sample.cantilever.V_ts, 'V')
    omega_c = ureg.Quantity(wc, 'Hz')

    # Main formula

    q0 = 4 * np.pi * epsilon0 * R * V

    prefactor = -q0**2 / (8 * np.pi * epsilon0 * omega_c)
    gamma = prefactor * K_jit(integrand, 2, sample, wc, location, location, True)

    # Return a frequency shift with units

    return gamma.to('pN s/m')

def freq_perpendicular_jit(integrand, sample, location):
    """Compute the frequency shift for a cantilever oscillating in the perpendicular 
    geometry using equations 30 and 31 in Roger Loring's 2025-06-05 document."""

    # Shorthand

    fc = sample.cantilever.f_c
    wc = sample.cantilever.omega_c
    kc = sample.cantilever.k_c

    # The parameters sample_jit object lacks units, so add them back in manually

    R = ureg.Quantity(sample.cantilever.R, 'm')
    V = ureg.Quantity(sample.cantilever.V_ts, 'V')
    f_c = ureg.Quantity(fc, 'Hz')
    k_c = ureg.Quantity(kc, 'N/m')

    # Main formulas

    q0 = 4 * np.pi * epsilon0 * R * V

    prefactor1 = -(f_c * q0**2) / (4 * np.pi * epsilon0 * k_c)
    prefactor2 = -(f_c * q0**2) / (16 * np.pi * epsilon0 * k_c)

    df1 = prefactor1 * K_jit(integrand, 2, sample, 0, location, location, False)
    df2 = prefactor2 * (K_jit(integrand, 2, sample, wc, location, location, False)
                      - K_jit(integrand, 2, sample, 0, location, location, False))
    
    return pint.Quantity.from_list([df1, df2]).to('Hz')