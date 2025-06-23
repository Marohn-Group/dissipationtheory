# python -m unittest -v test_dissipation9a.py
# python -m unittest -v tests/test_dissipation9a.py

import unittest
import numpy as np
from scipy import integrate
from dissipationtheory.constants import ureg
from dissipationtheory.dissipation9a import CantileverModel, SampleModel1, SampleModel2, SampleModel3, SampleModel4
from dissipationtheory.dissipation9a import integrand1, integrand2, integrand3, K, Kunits, Kmetal, Kmetalunits

def metalwrapper(x):
    return np.array([x[0].real, x[0].imag, x[1].real, x[1].imag, x[2].real, x[0].imag])

class TestDissipation9aMethods(unittest.TestCase):

    def setUp(self):

        cantilever = CantileverModel(
            f_c = ureg.Quantity(62, 'kHz'),
            k_c = ureg.Quantity(2.8, 'N/m'), 
            V_ts = ureg.Quantity(1, 'V'), 
            R = ureg.Quantity(55, 'nm'),
            angle = ureg.Quantity(20, 'degree'),
            L = ureg.Quantity(1000, 'nm')
        )

        self.omega = ureg.Quantity(1e5, 'Hz')
        self.loc1 = ureg.Quantity(np.array([ 10, 20, 50]), 'nm')
        self.loc2 = ureg.Quantity(np.array([ 0,   0, 50]), 'nm')

        self.sample1inf = SampleModel1(
            cantilever = cantilever,
            h_s = ureg.Quantity(100, 'um'),
            epsilon_s = ureg.Quantity(complex(20, 0), ''),
            sigma = ureg.Quantity(1e-7, 'S/m'),
            rho = ureg.Quantity(1e21, '1/m^3'),
            epsilon_d = ureg.Quantity(complex(1e6, 0), ''),
            z_r = ureg.Quantity(100, 'nm')
        )

        self.sample1metal = SampleModel1(
            cantilever = cantilever,
            h_s = ureg.Quantity(100, 'um'),
            epsilon_s = ureg.Quantity(complex(1e6, 0), ''),
            sigma = ureg.Quantity(1e7, 'S/m'),
            rho = ureg.Quantity(1e26, '1/m^3'),
            epsilon_d = ureg.Quantity(complex(1e6, 0), ''),
            z_r = ureg.Quantity(100, 'nm')
        )

        self.sample2zero = SampleModel2(
            cantilever = cantilever,
            epsilon_d = ureg.Quantity(complex(3, 0), ''),
            h_d = ureg.Quantity(1e-6, 'nm'),
            epsilon_s = ureg.Quantity(complex(20, 0), ''),
            sigma = ureg.Quantity(1e-7, 'S/m'),
            rho = ureg.Quantity(1e21, '1/m^3'),
            z_r = ureg.Quantity(100, 'nm')
        )

        self.sample2metal = SampleModel2(
            cantilever = cantilever,
            epsilon_d = ureg.Quantity(complex(3, 0), ''),
            h_d = ureg.Quantity(1e-6, 'nm'),
            epsilon_s = ureg.Quantity(complex(1e6, 0), ''),
            sigma = ureg.Quantity(1e7, 'S/m'),
            rho = ureg.Quantity(1e26, '1/m^3'),
            z_r = ureg.Quantity(100, 'nm')
        )

        self.sample3 = SampleModel3(
            cantilever = cantilever,
            epsilon_s = ureg.Quantity(complex(20, 0), ''),
            sigma = ureg.Quantity(1e-7, 'S/m'),
            rho = ureg.Quantity(1e21, '1/m^3'),
            z_r = ureg.Quantity(100, 'nm')
        )

        self.sample3metal = SampleModel3(
            cantilever = cantilever,
            epsilon_s = ureg.Quantity(complex(1e6, 0), ''),
            sigma = ureg.Quantity(1e7, 'S/m'),
            rho = ureg.Quantity(1e26, '1/m^3'),
            z_r = ureg.Quantity(100, 'nm')
        )

        self.sample4 = SampleModel4(
            cantilever = cantilever,
            z_r = ureg.Quantity(100, 'nm')
        )

        self.k3 = K(integrand3, self.sample3, self.omega, self.loc1, self.loc2)
        self.k4 = Kmetal(self.sample4, self.loc1, self.loc2)

        self.k3_flat = integrate.quad_vec(integrand3, 0., np.inf, args=(self.sample3, self.omega, self.loc1, self.loc2))[0]
        self.k4_flat = metalwrapper(Kmetal(self.sample4, self.loc1, self.loc2))

        self.units = ['1/nm','1/nm**2','1/nm**3']

        self.k3u = np.array(
            [val.to(unit).magnitude 
             for val, unit 
             in zip(Kunits(integrand3, self.sample3, self.omega, self.loc1, self.loc2), self.units)])

        self.k4u = np.array(
            [val.to(unit).magnitude 
             for val, unit 
             in zip(Kmetalunits(self.sample4, self.loc1, self.loc2), self.units)])

    def test_SampleModel1vs3_flat(self):
        """The Type I sample with large hs looks like a Type III sample (test reals)."""
        
        k1inf_flat = integrate.quad_vec(integrand1, 0., np.inf, args=(self.sample1inf, self.omega, self.loc1, self.loc2))[0]

        self.assertTrue(np.allclose(k1inf_flat, self.k3_flat, rtol=8e-4))

    def test_SampleModel1vs3(self):
        """The Type I sample with large hs looks like a Type III sample (test complex)."""

        k1inf = K(integrand1, self.sample1inf, self.omega, self.loc1, self.loc2)

        self.assertTrue(np.allclose(k1inf, self.k3, rtol=8e-4))

    def test_SampleModel1vs3_units(self):
        """"The Type I sample with large hs looks like a Type III sample (with units)."""
        
        k1inf  = np.array(
            [val.to(unit).magnitude 
             for val, unit 
             in zip(Kunits(integrand1, self.sample1inf, self.omega, self.loc1, self.loc2), self.units)])
        
        self.assertTrue(np.allclose(k1inf, self.k3u, rtol=8e-4))

    def test_SampleModel2vs3_flat(self):
        """"The Type II sample with small hd looks like a Type III sample (test reals)."""

        k2zero_flat = integrate.quad_vec(integrand2, 0., np.inf, args=(self.sample2zero, self.omega, self.loc1, self.loc2))[0]

        self.assertTrue(np.allclose(k2zero_flat, self.k3_flat, rtol=3e-8))

    def test_SampleModel1vs4_flat(self):
        """"The Type I sample with large hs, epsilon, charge density, and conductivity looks like a Type IV sample (test reals)."""

        k1metal_flat = integrate.quad_vec(integrand1, 0., np.inf, args=(self.sample1metal, ureg.Quantity(0, 'Hz'), self.loc1, self.loc2))[0]

        self.assertTrue(np.allclose(k1metal_flat[0::2], self.k4_flat[0::2], rtol=2e-6))

    def test_SampleModel1vs4(self):
        """"The Type I sample with large hs, epsilon, charge density, and conductivity looks like a Type IV sample (test complex)."""

        k1metal = K(integrand1, self.sample1metal, ureg.Quantity(0, 'Hz'), self.loc1, self.loc2)
    
        self.assertTrue(np.allclose(k1metal, self.k4, rtol=2e-6))

    def test_SampleModel1vs4_units(self):
        """"The Type I sample with large hs, epsilon, charge density, and conductivity looks like a Type IV sample (with units)."""

        k1metal  = np.array(
            [val.to(unit).magnitude 
             for val, unit 
             in zip(Kunits(integrand1, self.sample1metal, ureg.Quantity(0, 'Hz'), self.loc1, self.loc2), self.units)])

        self.assertTrue(np.allclose(k1metal, self.k4u, rtol=2e-6))

    def test_SampleModel2vs4_flat(self):
        """"The Type II sample with small hd and large epsilon, charge density, and conductivity looks like a Type IV sample (test reals)."""

        k2metal_flat = integrate.quad_vec(integrand2, 0., np.inf, args=(self.sample2metal, ureg.Quantity(0, 'Hz'), self.loc1, self.loc2))[0]

        self.assertTrue(np.allclose(k2metal_flat[0::2], self.k4_flat[0::2], rtol=2e-6))

    def test_SampleModel3vs4_flat(self):
        """"The Type III sample with large epsilon, charge density, and conductivity looks like a Type IV sample (test reals)."""

        k3metal_flat = integrate.quad_vec(integrand3, 0., np.inf, args=(self.sample3metal, ureg.Quantity(0, 'Hz'), self.loc1, self.loc2))[0]

        self.assertTrue(np.allclose(k3metal_flat[0::2], self.k4_flat[0::2], rtol=2e-6))