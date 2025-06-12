# Author: John A. Marohn
# Date: 2025-06-05
# Summary: Pure python functions from dissipation7.py

import numpy as np
import cmath
from dissipationtheory.constants import ureg, epsilon0, qe, kb
from scipy import integrate
from scipy.special import j0
from dissipationtheory.dissipation8b import CantileverModelJit

class CantileverModel(object):
    """Cantilever object.  
    SI units for each parameter are indicated below.
    A parameter can be given in any equivalent unit using the units package ``pint``.
    For example, :: 
    
        from dissipationtheory.constants import ureg
        R = ureg.Quantity(50., 'nm')
    
    :param ureg.Quantity f_c: cantilever frequency [Hz] 
    :param ureg.Quantity k_c: cantilever spring constant [N/m]
    :param ureg.Quantity V_ts: tip-sample voltage [V]
    :param ureg.Quantity R: tip radius [m]
    :param ureg.Quantity angle: tip cone half angle [degrees]
    :param ureg.Quantity L: tip cone length [m]
    :param ureg.Quantity d: tip-sample separation [m]

    """
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
        """Cantilever resonance frequency in units of radians/second."""
        return 2 * np.pi * self.f_c

    def __repr__(self):

        str = 'cantilever\n\n'
        str = str + '         resonance freq = {:.3f} kHz\n'.format(self.f_c.to('kHz').magnitude)
        str = str + '                        = {:.3e} rad/s\n'.format(self.omega_c.to('Hz').magnitude)
        str = str + '        spring constant = {:.3f} N/m\n'.format(self.k_c.to('N/m').magnitude)
        str = str + '     tip-sample voltage = {:.3f} V\n'.format(self.V_ts.to('V').magnitude)
        str = str + '                 radius = {:.3f} nm\n'.format(self.R.to('nm').magnitude)
        str = str + '        cone half angle = {:.3f} degree\n'.format(self.angle.to('degree').magnitude)
        str = str + '            cone length = {:.3f} nm\n'.format(self.L.to('nm').magnitude)        
        str = str + '  tip-sample separation = {:.3f} nm\n'.format(self.d.to('nm').magnitude)
        
        return str
    
    def args(self):
        """A dictionary with the cantilever parameters, useful for initializing 
        a jit version of the cantilever object:
        
            cantilever = CantileverModel(
                f_c = ureg.Quantity(75, 'kHz'),
                k_c = ureg.Quantity(2.8, 'N/m'), 
                V_ts = ureg.Quantity(1, 'V'), 
                R = ureg.Quantity(35, 'nm'),
                angle = ureg.Quantity(20, 'degree'),
                L = ureg.Quantity(1000, 'nm'),
                d = ureg.Quantity(38, 'nm'),
            )
            cantilever_jit = CantileverModelJit(**cantilever.args())

        """

        args = {
            'f_c': self.f_c.to('Hz').magnitude, 
            'k_c': self.k_c.to('N/m').magnitude, 
            'V_ts': self.V_ts.to('V').magnitude, 
            'R': self.R.to('m').magnitude,
            'angle': self.angle.to('degree').magnitude,
            'L': self.L.to('m').magnitude,
            'd': self.d.to('m').magnitude
        }

        return args

class SampleModel1(object):
    """Model I sample object defined in Lekkala2013nov, Fig. 1(b)::
    
        cantilever | vacuum gap | semiconductor | dielectric
    
    The dielectric substrate is semi-infinite.
    
    :param CantileverModel cantilever: an object storing the cantilever properties
    :param ureg.Quantity epsilon_s: semiconductor layer's complex relative dielectric constant [unitless]
    :param ureg.Quantity h_s: semiconductor layer's thickness [m]
    :param ureg.Quantity sigma: semiconductor layer's conductivity [S/m]
    :param ureg.Quantity rho: semiconductor layer's charge density [1/m^3]
    :param ureg.Quantity epsilon_d: dielectric layer's complex relative dielectric constant [unitless]
    :param ureg.Quantity z_r: reference height [m] (used in computations)

    We compute two derived quantifies:

    :param ureg.Quantity omega0: the roll-off frequency [Hz]
    :param ureg.Quantity kD: the inverse Debye length [1/m]
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
    def kD(self):
        """Inverse Debye length."""
        return (np.sqrt((self.rho * qe * qe)/(epsilon0 * kb * ureg.Quantity(300., 'K')))).to('1/nm')

    def __repr__(self):

        str = self.cantilever.__repr__()
        str = str + '\nsemiconductor\n\n'
        str = str + '             epsilon (real) = {:0.3f}\n'.format(self.epsilon_s.to('').real.magnitude)
        str = str + '             epsilon (imag) = {:0.3f}\n'.format(self.epsilon_s.to('').imag.magnitude)
        str = str + '                  thickness = {:0.1f} nm\n'.format(self.h_s.to('nm').magnitude)
        str = str + '               conductivity = {:0.3e} S/m\n'.format(self.sigma.to('S/m').magnitude)
        str = str + '             charge density = {:0.3e} m^{{-3}}\n'.format(self.rho.to('1/m^3').magnitude)
        str = str + '           reference height = {:0.3e} nm\n'.format(self.z_r.to('nm').magnitude)
        str = str + '\n'
        str = str + '         roll-off frequency = {:0.3e} Hz\n'.format(self.omega0.to('Hz').magnitude)            
        str = str + '       inverse Debye length = {:0.3e} nm^{{-1}}\n'.format(self.kD.to('1/nm').magnitude)
        str = str + '               Debye length = {:0.3e} nm\n'.format(1/self.kD.to('1/nm').magnitude)
        str = str + 'dielectric\n\n'
        str = str + '  epsilon (real) = {:0.3f}\n'.format(self.epsilon_d.to('').real.magnitude)
        str = str + '  epsilon (imag) = {:0.3f}\n'.format(self.epsilon_d.to('').imag.magnitude)
        str = str + '       thickness = infinite'
        
        return str

    def args(self):
        """A dictionary with the sample parameters, useful for initializing 
        a jit version of the sample object:
        
            cantilever = CantileverModel(
                f_c = ureg.Quantity(75, 'kHz'),
                k_c = ureg.Quantity(2.8, 'N/m'), 
                V_ts = ureg.Quantity(1, 'V'), 
                R = ureg.Quantity(35, 'nm'),
                angle = ureg.Quantity(20, 'degree'),
                L = ureg.Quantity(1000, 'nm'), 
                d = ureg.Quantity(38, 'nm')
            )

            sample1 = SampleModel1(
                cantilever = cantilever,
                h_s = ureg.Quantity(500, 'nm'),
                epsilon_s = ureg.Quantity(complex(20, -0.2), ''),
                sigma = ureg.Quantity(1E-5, 'S/m'),
                rho = ureg.Quantity(1e21, '1/m^3'),
                epsilon_d = ureg.Quantity(complex(1e6, 0), ''),
                z_r = ureg.Quantity(300, 'nm')
            )

            sample_jit = SampleModel1Jit(**sample1.args())

        """

        cantilever_jit = CantileverModelJit(**self.cantilever.args())

        args = {
            'cantilever': cantilever_jit,
            'h_s': self.h_s.to('m').magnitude,
            'epsilon_s': self.epsilon_s.to('').magnitude,
            'sigma': self.sigma.to('S/m').magnitude,
            'rho': self.rho.to('1/m^3').magnitude,
            'epsilon_d': self.epsilon_d.to('').magnitude,
            'z_r': self.z_r.to('m').magnitude
        }

        return args

class SampleModel2(object):
    """Model II sample object defined in Lekkala2013nov, Fig. 1(b)::
    
        cantilever | vacuum gap | dielectric | semiconductor
    
    The semiconductor substrate is semi-infinite.
    
    :param CantileverModel cantilever: an object storing the cantilever properties
    :param ureg.Quantity epsilon_d: dielectric layer's complex relative dielectric constant [unitless]
    :param ureg.Quantity h_d: dielectric layer's thickness [m]
    :param ureg.Quantity epsilon_s: semiconductor layer's complex relative dielectric constant [unitless]
    :param ureg.Quantity mu: semiconductor layer's charge conductivity [S/m]
    :param ureg.Quantity rho: semiconductor layer's charge density [1/m^3]
    :param ureg.Quantity z_r: reference height [m] (used in computations)

    We compute two derived quantifies:

    :param ureg.Quantity omega0: the roll-off frequency [Hz]
    :param ureg.Quantity kD: the inverse Debye length [1/m]
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
    def kD(self):
        """Inverse Debye length."""
        return (np.sqrt((self.rho * qe * qe)/(epsilon0 * kb * ureg.Quantity(300., 'K')))).to('1/nm')

    def __repr__(self):

        str = self.cantilever.__repr__()
        str = str + '\ndielectric\n\n'
        str = str + '  epsilon (real) = {:0.3f}\n'.format(self.epsilon_d.to('').real.magnitude)
        str = str + '  epsilon (imag) = {:0.3f}\n'.format(self.epsilon_d.to('').imag.magnitude)
        str = str + '       thickness = {:0.1f} nm\n\n'.format(self.h_d.to('nm').magnitude)
        str = str + 'semiconductor\n\n'
        str = str + '             epsilon (real) = {:0.3f}\n'.format(self.epsilon_s.to('').real.magnitude)
        str = str + '             epsilon (imag) = {:0.3f}\n'.format(self.epsilon_s.to('').imag.magnitude)
        str = str + '                  thickness = infinite\n'
        str = str + '               conductivity = {:0.3e} S/m\n'.format(self.sigma.to('S/m').magnitude)
        str = str + '             charge density = {:0.2e} m^{{-3}}\n'.format(self.rho.to('1/m^3').magnitude)
        str = str + '           reference height = {:0.1f} nm\n'.format(self.z_r.to('nm').magnitude)
        str = str + '\n'
        str = str + '         roll-off frequency = {:0.3e} Hz\n'.format(self.omega0.to('Hz').magnitude)        
        str = str + '       inverse Debye length = {:0.3e} nm^{{-1}}\n'.format(self.kD.to('1/nm').magnitude)
        str = str + '               Debye length = {:0.3e} nm\n'.format(1/self.kD.to('1/nm').magnitude)     

        return str

    def args(self):
        """A dictionary with the sample parameters, useful for initializing 
        a jit version of the sample object:
        
            cantilever = CantileverModel(
                f_c = ureg.Quantity(75, 'kHz'),
                k_c = ureg.Quantity(2.8, 'N/m'), 
                V_ts = ureg.Quantity(1, 'V'), 
                R = ureg.Quantity(35, 'nm'),
                angle = ureg.Quantity(20, 'degree'),
                L = ureg.Quantity(1000, 'nm'), 
                d = ureg.Quantity(38, 'nm'),
                z_c = ureg.Quantity(73, 'nm')
            )

            sample2 = SampleModel2(
                cantilever = cantilever,
                epsilon_d = ureg.Quantity(complex(20, -0.2), ''),
                h_d = ureg.Quantity(1, 'nm'),
                epsilon_s = ureg.Quantity(complex(20, -0.2), ''),
                sigma = ureg.Quantity(1E-5, 'S/m'),
                rho = ureg.Quantity(1e21, '1/m^3'),
                z_r = ureg.Quantity(300, 'nm')
            )            

            sample_jit = SampleModel2Jit(**sample2.args())

        """

        cantilever_jit = CantileverModelJit(**self.cantilever.args())

        args = {
            'cantilever': cantilever_jit,
            'epsilon_d': self.epsilon_d.to('').magnitude,
            'h_d': self.h_d.to('m').magnitude,
            'epsilon_s': self.epsilon_s.to('').magnitude,
            'sigma': self.sigma.to('S/m').magnitude,
            'rho': self.rho.to('1/m^3').magnitude,
            'z_r': self.z_r.to('m').magnitude
        }

        return args

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

def integrand1(y, power, sample, omega, location1, location2, part=np.imag):
    """Theta function for the Sample I geometry of Lekkala.
    
    In the code below, `y` is the unitless integration variable."""

    es = sample.epsilon_s
    ed = sample.epsilon_d
    zr = sample.z_r
    hs = sample.h_s

    Omega = (omega/sample.omega0).to('dimensionless').magnitude
    theta1 = np.sqrt(y**2 * (hs / zr)**2 + (hs * sample.kD)**2 * (1/es + complex(0,1) * Omega)).to('dimensionless').magnitude
    theta2 = (y * hs/ zr).to('dimensionless').magnitude

    k_over_eta = (y / np.sqrt(y**2 + (zr * sample.kD)**2 * (1/es + complex(0,1) * Omega))).to('dimensionless').magnitude

    p0 = 1 + complex(0,1) * es * Omega
    p1 = k_over_eta / (es * p0)
    p2 = - Omega**2 / (p0 * p0)
    p3 = complex(0,1) * Omega / (ed * p0)
    p4 = complex(0,1) * Omega * k_over_eta / (es * p0**2)
    p5 = - k_over_eta**2 / (es**2 * p0**2)
    p6 = complex(0,1) * Omega / p0
    p7 = 1 / ed

    t1 = p1 / np.tanh(theta1)
    n1 = p2 * np.tanh(theta1) * np.tanh(theta2) + p3 * np.tanh(theta1) + p4
    n2 = - 2 * p4 * mysech(theta2) * mysech(theta1) + p5 * np.tanh(theta2) * mysech(theta1) * mycsch(theta1)
    d1 = p6 * np.tanh(theta1) + np.tanh(theta2) * (p1 + p7 * np.tanh(theta1))
    
    theta_norm = t1 + (n1 + n2)/d1
    rp = ((1 - theta_norm) / (1 + theta_norm)).to('dimensionless').magnitude

    rhoX = ((location1[0] - location2[0]) / zr).to('dimensionless').magnitude
    rhoY = ((location1[1] - location2[1])/ zr).to('dimensionless').magnitude
    argument = y * np.sqrt(rhoX**2 + rhoY**2)
    exponent = y * ((location1[2] + location2[2])/ zr).to('dimensionless').magnitude

    integrand = y**power * j0(argument) * np.exp(-1 * exponent) * part(rp) 

    return integrand

def integrand2(y, power, sample, omega, location1, location2, part=np.imag):
    """Theta function for the Sample II geometry of Lekkala.
    
    In the code below, `y` is the unitless integration variable."""

    es = sample.epsilon_s
    zr = sample.z_r

    Omega = (omega/sample.omega0).to('dimensionless').magnitude
    theta2 = (y * sample.h_d / zr).to('dimensionless').magnitude

    k_over_eta = (y / np.sqrt(y**2 + (zr * sample.kD)**2 * (1/es + complex(0,1) * Omega))).to('dimensionless').magnitude

    p0 = 1 + complex(0,1) * es * Omega
    p1 = k_over_eta / (es * p0)
    p6 = complex(0,1) * Omega / p0
    p7 = 1 / sample.epsilon_d

    theta_norm = p7 * (p7 * np.tanh(theta2) + p6 + p1)/(p7 + (p6 + p1) * np.tanh(theta2))
    rp = ((1 - theta_norm) / (1 + theta_norm)).to('dimensionless').magnitude

    rhoX = ((location1[0] - location2[0]) / zr).to('dimensionless').magnitude
    rhoY = ((location1[1] - location2[1])/ zr).to('dimensionless').magnitude
    argument = y * np.sqrt(rhoX**2 + rhoY**2)
    exponent = y * ((location1[2] + location2[2])/ zr).to('dimensionless').magnitude

    integrand = y**power * j0(argument) * np.exp(-1 * exponent) * part(rp) 

    return integrand

def K(integrand, power, sample, omega, location1, location2, part=np.imag):
    """Compute the integral :math:`K`."""
    
    prefactor = 1/np.power(sample.z_r, power+1)    
    integral = integrate.quad(integrand, 0., np.inf, args=(power, sample, omega, location1, location2, part))[0]
    
    return (prefactor * integral).to_base_units()