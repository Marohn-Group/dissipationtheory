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
from numba import float64, complex128, boolean, uint16
from numba.experimental import jitclass
from numba import deferred_type


CantileverModelSpec = [
    ('f_c', float64),
    ('k_c', float64),
    ('V_ts', float64),
    ('R', float64),
    ('angle', float64),
    ('L', float64)] 

@jitclass(CantileverModelSpec)
class CantileverModelJit(object):

    def __init__(self, f_c, k_c, V_ts, R, angle, L):

        self.f_c = f_c
        self.k_c = k_c
        self.V_ts = V_ts
        self.R = R
        self.angle = angle
        self.L = L

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
    ('z_r',float64),
    ('type',uint16)]

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
        self.type = 1

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
        print("sample type = ", self.type)
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
    ('z_r',float64),
    ('type',uint16)]

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
        self.type = 2

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
        print("sample type = ", self.type)
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


SampleModel3Spec = [
    ('cantilever', CantileverModelJit_type),
    ('epsilon_s',complex128),
    ('sigma',float64),
    ('rho',float64),
    ('z_r',float64),
    ('type',uint16)]

@jitclass(SampleModel3Spec)
class SampleModel3Jit(object):
    
    def __init__(self, cantilever, epsilon_s, sigma, rho, z_r):

        self.cantilever = cantilever
        self.epsilon_s = epsilon_s
        self.sigma = sigma
        self.rho = rho
        self.z_r = z_r
        self.type = 3

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
        print("sample type = ", self.type)
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


SampleModel4Spec = [
    ('cantilever', CantileverModelJit_type),
    ('z_r',float64),
    ('type',uint16)]

@jitclass(SampleModel4Spec)
class SampleModel4Jit(object):
    """Model IV sample object::
    
        cantilever | vacuum gap | metal
    
    The metal substrate is semi-infinite.
    
    :param CantileverModel cantilever: an object storing the cantilever properties
    :param ureg.Quantity z_r: reference height [m] (used in computations)

    """

    def __init__(self, cantilever, z_r):

        self.cantilever = cantilever
        self.z_r = z_r
        self.type = 4

    def print(self):

        print("cantilever")
        print("==========")
        self.cantilever.print()
        print("")
        print("sample type = ", self.type)
        print("")
        print("metal")
        print("=====")
        print("thickness = infinite")


# To compile the integrand1_jit(), integrand2_jit(), integrand3_jit(), and 
# integrand3_jit() functions, jit needs an actual instance of the function
# input "sample". 

sample1 = SampleModel1Jit(
    cantilever=CantileverModelJit(75.0e3, 2.8, 1.0, 35.0E-9, 20.0, 1000.0E-9),
    epsilon_s=complex(11.9,-0.05),
    h_s=3000E-9,
    sigma = 1e-8,
    rho=1e21,
    epsilon_d=complex(11.9,-0.05),
    z_r=300E-9)

sample2 = SampleModel2Jit(
    cantilever=CantileverModelJit(75.0e3, 2.8, 1.0, 35.0E-9, 20.0, 1000.0E-9),
    epsilon_d=complex(11.9,-0.05),
    h_d=0.,
    epsilon_s=complex(11.9,-0.05),
    sigma = 1e-8,
    rho=1e21,
    z_r=300E-9)

sample3 = SampleModel3Jit(
    cantilever=CantileverModelJit(75.0e3, 2.8, 1.0, 35.0E-9, 20.0, 1000.0E-9),
    epsilon_s=complex(11.9,-0.05),
    sigma = 1e-8,
    rho=1e21,
    z_r=300E-9)

sample4 = SampleModel4Jit(
    cantilever=CantileverModelJit(75.0e3, 2.8, 1.0, 35.0E-9, 20.0, 1000.0E-9),
    z_r=100E-9)


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

@jit(float64[:](float64,
             SampleModel1Jit.class_type.instance_type,
             float64,
             float64[:],
             float64[:]), nopython=True)

def integrand1jit(y, sample, omega, location1, location2):
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

    integrand = np.array([       np.real(rp),        np.imag(rp), 
                          y    * np.real(rp), y    * np.imag(rp),
                          y**2 * np.real(rp), y**2 * np.imag(rp)]) * scipy.special.j0(argument) * np.exp(-1 * exponent)

    return integrand

@jit(float64[:](float64,
             SampleModel2Jit.class_type.instance_type,
             float64,
             float64[:],
             float64[:]), nopython=True)

def integrand2jit(y, sample, omega, location1, location2):

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

    integrand = np.array([       np.real(rp),        np.imag(rp), 
                          y    * np.real(rp), y    * np.imag(rp),
                          y**2 * np.real(rp), y**2 * np.imag(rp)]) * scipy.special.j0(argument) * np.exp(-1 * exponent)

    return integrand

@jit(float64[:](float64,
             SampleModel3Jit.class_type.instance_type,
             float64,
             float64[:],
             float64[:]), nopython=True)

def integrand3jit(y, sample, omega, location1, location2):
    """Theta function for a Sample III object, a semi-infinite dielectric.
    
    In the code below, `y` is the unitless integration variable."""

    es = sample.epsilon_s
    zr = sample.z_r

    Omega = omega/sample.omega0
    k_over_eta = y / np.sqrt(y**2 + (zr * sample.kD)**2 * (1/es + complex(0,1) * Omega)) # depends on y

    p0 = 1 + complex(0,1) * es * Omega
    p1 = k_over_eta / (es * p0)          # depends on y
    p6 = complex(0,1) * Omega / p0

    theta_norm = p6 + p1
    rp = (1 - theta_norm) / (1 + theta_norm)

    rhoX = (location1[0] - location2[0])/ zr
    rhoY = (location1[1] - location2[1])/ zr
    argument = y * np.sqrt(rhoX**2 + rhoY**2)
    exponent = y * (location1[2] + location2[2])/ zr

    integrand = np.array([       np.real(rp),        np.imag(rp), 
                          y    * np.real(rp), y    * np.imag(rp),
                          y**2 * np.real(rp), y**2 * np.imag(rp)]) * scipy.special.j0(argument) * np.exp(-1 * exponent)

    return integrand

# For now, just a copy of the dissipation9a.py function
# Maps the 6-component vector of real numbers with components
# 
#   Re[K0], Im[K0], Re[K1], Im[K1], Re[K2], Im[K2]
#
# to the the 3-component complex vector with components
#
#   Re[K0] + i Im[K0], Re[K1] + i Im[K1], Re[K2] + i Im[K2]

Kp = np.array([[complex(1,0), 0, 0],
               [complex(0,1), 0, 0],
               [0, complex(1,0), 0],
               [0, complex(0,1), 0],
               [0, 0, complex(1,0)],
               [0, 0, complex(0,1)]])

# For now, just a copy of the dissipation9a.py function
def K_jit(integrand, sample, omega, location1, location2):
    """Compute the integrals :math:`K_0, K_1, K_2`, unscaled by $z_{\mathrm{r}}$, without units."""
    
    integrals = integrate.quad_vec(integrand, 0., np.inf, args=(sample, omega, location1, location2))[0]
    return integrals @ Kp

# For now, just a copy of the dissipation9a.py function
def Kunits_jit(integrand, sample, omega, location1, location2):
    """Compute the integrals :math:`K_0, K_1, K_2`, scaled by $z_{\mathrm{r}}$, with units."""

    integrals = integrate.quad_vec(integrand, 0., np.inf, args=(sample, omega, location1, location2))[0]
    K0, K1, K2 = integrals @ Kp

    zr_u = ureg.Quantity(sample.z_r, 'm')

    K0u = K0 / np.power(zr_u, 1)
    K1u = K1 / np.power(zr_u, 2)
    K2u = K2 / np.power(zr_u, 3)

    return K0u.to('1/nm**1'), K1u.to('1/nm**2'), K2u.to('1/nm**3')

@jit(nb_types.UniTuple(complex128,3)(SampleModel4Jit.class_type.instance_type,
             float64[:],
             float64[:]), nopython=True)

def Kmetal_jit(sample, location1, location2):
    """The Green's function for a metal, the unitless image potential and derivatives."""

    # shorthand
    s = location1/sample.z_r
    r = location2/sample.z_r
    
    # location of image charge
    ri = r.copy()
    ri[2] = -1 * ri[2]

    # shorthand
    Rinv = np.power((s-ri).T @ (s-ri), -1/2)
    
    G0 = complex(-1,0) * Rinv
    G1 = complex( 2,0) * (s[2] + r[2]) *  np.power(Rinv, 3)
    G2 = complex( 4,0) * (np.power(Rinv, 3) - 3 * np.power(s[2] + r[2], 2) * np.power(Rinv, 5))
    
    K0, K1, K2 = -G0, G1/2, -G2/4

    return K0, K1, K2

# I cannot figure out how to get a `@jit` compiled function to return 
# a `ureg.Quantity` object, so I do not know how to write an analogous 
# `Kmetalunits_jit` function. The function below calls the compiled
# function Kmetal_jit and then adds on units afterwards.

def Kmetalunits_jit(sample, location1, location2):

    K0, K1, K2 = Kmetal_jit(sample, location1, location2)
    zr_u = ureg.Quantity(sample.z_r, 'm')
    K0u, K1u, K2u = K0/zr_u**1, K1/zr_u**2, K2/zr_u**3

    return K0u.to('1/nm**1'), K1u.to('1/nm**2'), K2u.to('1/nm**3')