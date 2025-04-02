import unittest
from dissipationtheory.constants import ureg
from dissipationtheory.dissipation5 import CantileverModel, SampleModel1, SampleModel2
from dissipationtheory.dissipation5 import CantileverModelJit, SampleModel1Jit, SampleModel2Jit
import pandas as pd
import numpy as np

# Command line: python -m unittest -v test_dissipation5.py

class TestDissipation2Methods(unittest.TestCase):

    def setUp(self):

        self.cantilever = CantileverModel(
            f_c = ureg.Quantity(81, 'kHz'), 
            k_c = ureg.Quantity(2.8, 'N/m'),
            V_ts = ureg.Quantity(3, 'V'), 
            R = ureg.Quantity(80, 'nm'), 
            angle = ureg.Quantity(20, 'degree'),
            d = ureg.Quantity(300, 'nm'),
            z_c = ureg.Quantity(380, 'nm')
        )
        
        self.sample1 = SampleModel1(
            cantilever = self.cantilever,
            h_s = ureg.Quantity(3000., 'nm'),
            epsilon_s = ureg.Quantity(complex(11.9, -0.05), ''),
            sigma = ureg.Quantity(1E-8, 'S/m'),
            rho = ureg.Quantity(1e21, '1/m^3'),
            epsilon_d = ureg.Quantity(complex(11.9, -0.05), ''),
            z_r = ureg.Quantity(300, 'nm')
        )

        self.sample2 = SampleModel2(
            cantilever = self.cantilever,
            h_d = ureg.Quantity(0., 'nm'),
            epsilon_d = ureg.Quantity(complex(11.9, -0.05), ''),
            epsilon_s = ureg.Quantity(complex(11.9, -0.05), ''),
            sigma = ureg.Quantity(1E-8, 'S/m'),
            rho = ureg.Quantity(1e21, '1/m^3'),
            z_r = ureg.Quantity(300, 'nm')
        )

        self.cantilever_jit = CantileverModelJit(
            f_c = 81e3,
            k_c = 2.8, 
            V_ts = 3.0,
            angle = 20,
            R = 80e-9,
            d = 300e-9,
            z_c = 380e-9
        )
        
        self.sample1_jit = SampleModel1Jit(
            cantilever = self.cantilever_jit,
            h_s = 3000e-9,
            epsilon_s = complex(11.9, -0.05),
            sigma = 1e-8,
            rho = 1e21,
            epsilon_d = complex(11.9, -0.05),
            z_r = 300e-9
        )

        self.sample2_jit = SampleModel2Jit(
            cantilever = self.cantilever_jit,
            h_d = 0,
            epsilon_d = complex(11.9, -0.05),
            epsilon_s = complex(11.9, -0.05),
            sigma = 1e-8,
            rho = 1e21,
            z_r = 300e-9
        )

    def test_SampleModel1_a(self):
        """Check that the mobility in ``SampleModel1`` is computed correctly given the conductivity and the charge density.
        """

        mu0 = 6.241509090043337e-11
        mu1 = self.sample1.mu.to('m^2/(V s)').magnitude
        self.assertTrue(np.allclose(mu0, mu1))

    def test_SampleModel1_b(self):
        """Check that the roll-off frequency in ``SampleModel1`` is computed correctly given the conductivity.
        """

        omega0a = 1129.409067373019
        omega0b = self.sample1.omega0.to('Hz').magnitude
        self.assertTrue(np.allclose(omega0a, omega0b))

    def test_SampleModel1_c(self):
        """Check that the diffusion constant in ``SampleModel1`` is computed correctly given the conductivity and the charge density.
        """

        k = ureg.Quantity(1.380649e-23, 'J/K') * ureg.Quantity(300., 'K') / ureg.Quantity(1.60217663e-19, 'C')
        D0 = (k * ureg.Quantity(6.241509090043337e-11, 'm^2/(V s)')).to('m^2/s').magnitude
        D1 = self.sample1.D.to('m^2/s').magnitude
        self.assertTrue(np.allclose(D0, D1))

    def test_SampleModel2_a(self):
        """Check that the mobility  in ``SampleModel2``is computed correctly given the conductivity and the charge density.
        """

        mu0 = 6.241509090043337e-11
        mu1 = self.sample2.mu.to('m^2/(V s)').magnitude
        self.assertTrue(np.allclose(mu0, mu1))

    def test_SampleModel2_b(self):
        """Check that the roll-off frequency in ``SampleModel2`` is computed correctly given the conductivity.
        """

        omega0a = 1129.409067373019
        omega0b = self.sample2.omega0.to('Hz').magnitude
        self.assertTrue(np.allclose(omega0a, omega0b))

    def test_SampleModel2_c(self):
        """Check that the diffusion constant in ``SampleModel1`` is computed correctly given the conductivity and the charge density.
        """

        k = ureg.Quantity(1.380649e-23, 'J/K') * ureg.Quantity(300., 'K') / ureg.Quantity(1.60217663e-19, 'C')
        D0 = (k * ureg.Quantity(6.241509090043337e-11, 'm^2/(V s)')).to('m^2/s').magnitude
        D1 = self.sample2.D.to('m^2/s').magnitude
        self.assertTrue(np.allclose(D0, D1))

    def test_SampleModel1Jit_a(self):
        """Check that the mobility in ``SampleModel1Jit`` is computed correctly given the conductivity and the charge density.
        """

        mu0 = 6.241509090043337e-11
        mu1 = self.sample1_jit.mu
        self.assertTrue(np.allclose(mu0, mu1))

    def test_SampleModel1Jit_b(self):
        """Check that the roll-off frequency in ``SampleModel1Jit`` is computed correctly given the conductivity.
        """

        omega0a = 1129.409067373019
        omega0b = self.sample1_jit.omega0
        self.assertTrue(np.allclose(omega0a, omega0b))

    def test_SampleModel1Jit_c(self):
        """Check that the diffusion constant in ``SampleModel1Jit`` is computed correctly given the conductivity and the charge density.
        """

        k = 1.380649e-23 * 300 / 1.60217663e-19
        D0 = k * 6.241509090043337e-11
        D1 = self.sample1_jit.D
        self.assertTrue(np.allclose(D0, D1))

    def test_SampleModel2Jit_a(self):
        """Check that the mobility in ``SampleModel2Jit`` is computed correctly given the conductivity and the charge density.
        """

        mu0 = 6.241509090043337e-11
        mu1 = self.sample2_jit.mu
        self.assertTrue(np.allclose(mu0, mu1))

    def test_SampleModel2Jit_b(self):
        """Check that the roll-off frequency in ``SampleModel2Jit`` is computed correctly given the conductivity.
        """

        omega0a = 1129.409067373019
        omega0b = self.sample2_jit.omega0
        self.assertTrue(np.allclose(omega0a, omega0b))

    def test_SampleModel2Jit_c(self):
        """Check that the diffusion constant in ``SampleModel2Jit`` is computed correctly given the conductivity and the charge density.
        """

        k = 1.380649e-23 * 300 / 1.60217663e-19
        D0 = k * 6.241509090043337e-11
        D1 = self.sample2_jit.D
        self.assertTrue(np.allclose(D0, D1))        