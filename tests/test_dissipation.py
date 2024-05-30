import unittest
from dissipationtheory.constants import ureg
from dissipationtheory.dissipation import CantileverModel, SampleModel1, SampleModel2
from dissipationtheory.dissipation import theta1norm, theta2norm, gamma_parallel, gamma_perpendicular
from dissipationtheory.dissipation import gamma_parallel_approx, gamma_perpendicular_approx
from dissipationtheory.dissipation import CantileverModelJit, SampleModel1Jit, SampleModel2Jit
from dissipationtheory.dissipation import theta1norm_jit, theta2norm_jit, gamma_parallel_jit, gamma_perpendicular_jit
import pandas as pd
import numpy as np
import os

class TestDissipationMethods(unittest.TestCase):

    def setUp(self):

        self.cantilever = CantileverModel(
            f_c = ureg.Quantity(81, 'kHz'), 
            V_ts = ureg.Quantity(3, 'V'), 
            R = ureg.Quantity(80, 'nm'), 
            d = ureg.Quantity(300, 'nm')
        )
        
        self.sample1 = SampleModel1(
            cantilever = self.cantilever,
            h_s = ureg.Quantity(3000., 'nm'),
            epsilon_s = ureg.Quantity(complex(11.9, -0.05), ''),
            mu = ureg.Quantity(2.7E-10, 'm^2/(V s)'),
            rho = ureg.Quantity(1e21, '1/m^3'),
            epsilon_d = ureg.Quantity(complex(11.9, -0.05), ''),
            z_r = ureg.Quantity(300, 'nm')
        )

        self.sample2 = SampleModel2(
            cantilever = self.cantilever,
            h_d = ureg.Quantity(0., 'nm'),
            epsilon_d = ureg.Quantity(complex(11.9, -0.05), ''),
            epsilon_s = ureg.Quantity(complex(11.9, -0.05), ''),
            mu = ureg.Quantity(2.7e-10, 'm^2/(V s)'),
            rho = ureg.Quantity(1e21, '1/m^3'),
            z_r = ureg.Quantity(300, 'nm')
        )

        self.cantilever_jit = CantileverModelJit(
            f_c = 81e3, 
            V_ts = 3.0,
            R = 80e-9,
            d = 300e-9
        )
        
        self.sample1_jit = SampleModel1Jit(
            cantilever = self.cantilever_jit,
            h_s = 3000e-9,
            epsilon_s = complex(11.9, -0.05),
            mu = 2.7e-10,
            rho = 1e21,
            epsilon_d = complex(11.9, -0.05),
            z_r = 300e-9
        )

        self.sample2_jit = SampleModel2Jit(
            cantilever = self.cantilever_jit,
            h_d = 0,
            epsilon_d = complex(11.9, -0.05),
            epsilon_s = complex(11.9, -0.05),
            mu = 2.7e-10,
            rho = 1e21,
            z_r = 300e-9
        )

    def test_cantilever_model(self):
        """Enter the tip-sample separation in nanometers and check that it comes out 
        right in micrometers.
        """

        self.assertEqual(0.300, self.cantilever.d.to('um').magnitude)

    def test_Lekkala2013nov_Fig7b(self):
        """Reproduce the parallel friction :math:`\\gamma_{\\parallel}` 
        calculation in Lekkala2013nov, Fig. 7(b), with 
        :math:`\\mu = 2.7 \\times 10^{-10} \\: \\mathrm{m}^2 \\: \\mathrm{V}^{-1} \\: \\mathrm{s}^{-1}`.
        Lekkala performed the calculation in *Mathematica* and the code no longer exists.
        The data is from ``Lekkala2013--Fig7b--2.7e-10.csv``.
        This data was created by digitizing the paper figure.
        See ``development/dissipation-theory--Study-4.html``. 
        The difference between Lekkala's calculation and the ``dissipationtheory`` calculation is less than 15 percent. 
        """

        filename = os.path.join(os.path.dirname(__file__), 'Lekkala2013--Fig7b--2.7e-10.csv')
        df = pd.read_csv(filename, names=['rho', 'gamma'], encoding='utf-8')
        df = df[df['rho'] > 5e24]

        rho1_ = ureg.Quantity(df['rho'].to_numpy(), '1/m^3')
        gamma0 = ureg.Quantity(df['gamma'].to_numpy(), 'pN s/m')
        gamma1 = ureg.Quantity(np.zeros_like(rho1_), 'pN s/m')
        err = np.zeros_like(rho1_)

        rho_original = self.sample1.rho
        for index, rho__ in enumerate(rho1_):

            self.sample1.rho = rho__
            gamma1[index] = gamma_parallel(theta1norm, self.sample1)

            x = rho__.to('1/m^3').magnitude
            y0 = gamma0[index].to('pN s/m').magnitude
            y1 = gamma1[index].to('pN s/m').magnitude

            err[index] = (y1 - y0)/y0

        self.sample1.rho = rho_original
        self.assertLess(np.abs(err).mean(), 0.15) 

    def test_Lekkala2013nov_Fig7b_JIT(self):    
        """Reproduce ``test_Lekkala2013nov_Fig7b`` using JIT versions of all the objects and functions."""

        filename = os.path.join(os.path.dirname(__file__), 'Lekkala2013--Fig7b--2.7e-10.csv')
        df = pd.read_csv(filename, names=['rho', 'gamma'], encoding='utf-8')
        df = df[df['rho'] > 5e24]

        rho1_ = ureg.Quantity(df['rho'].to_numpy(), '1/m^3')
        gamma0 = ureg.Quantity(df['gamma'].to_numpy(), 'pN s/m')
        gamma1 = ureg.Quantity(np.zeros_like(rho1_), 'pN s/m')
        err = np.zeros_like(rho1_)

        rho_original = self.sample1_jit.rho
        for index, rho__ in enumerate(rho1_):

            self.sample1_jit.rho = rho__.to('1/m^3').magnitude
            gamma1[index] = gamma_parallel_jit(theta1norm_jit, self.sample1_jit).to('pN s/m')

            y0 = gamma0[index].to('pN s/m').magnitude
            y1 = gamma1[index].to('pN s/m').magnitude

            err[index] = (y1 - y0)/y0

        self.sample1_jit.rho = rho_original
        self.assertLess(np.abs(err).mean(), 0.15)

    def test_low_density_approx_parallel(self):
        """Test the exact numerical answer for :math:`\\gamma_{\\parallel}` for Sample I against a low-density approximation.
        See ``development/dissipation-theory--Study-4.html``.
        For this test, we create a list of densities below the turning point where the low-density expansion is valid.
        The maximum difference between the exact answer and the low-density expansion should be less than 2.5 percent.
        """

        rho_trial = ureg.Quantity(np.logspace(start=np.log10(1e15), stop=np.log10(1e23), num=9), '1/m^3')
        (rho1, gamma1) = gamma_parallel_approx(rho_trial, self.sample1)
        
        gamma0 = ureg.Quantity(np.zeros_like(rho1), 'pN s/m')
        err = np.zeros_like(rho1)     
        rho_original = self.sample1.rho
        for index, rho_ in enumerate(rho1):

            self.sample1.rho = rho_
            gamma0[index] = gamma_parallel(theta1norm, self.sample1)

            y0 = gamma0[index].to('pN s/m').magnitude
            y1 = gamma1[index].to('pN s/m').magnitude

            err[index] = (y1 - y0)/y0

        self.sample1.rho = rho_original
        self.assertLess(np.abs(err).max(), 0.025)

    def test_low_density_approx_parallel_JIT(self):
        """Reproduce ``test_low_density_approx_parallel()`` using JIT versions of all the objects and functions."""

        rho_trial = ureg.Quantity(np.logspace(start=np.log10(1e15), stop=np.log10(1e23), num=9), '1/m^3')
        (rho1, gamma1) = gamma_parallel_approx(rho_trial, self.sample1)

        gamma0 = ureg.Quantity(np.zeros_like(rho1), 'pN s/m')
        err = np.zeros_like(rho1)
        rho_original = self.sample1_jit.rho
        for index, rho_ in enumerate(rho1):

            self.sample1_jit.rho = rho_.to('1/m^3').magnitude
            gamma0[index] = gamma_parallel_jit(theta1norm_jit, self.sample1_jit).to('pN s/m')

            y0 = gamma0[index].to('pN s/m').magnitude
            y1 = gamma1[index].to('pN s/m').magnitude

            err[index] = (y1 - y0)/y0

        self.sample1_jit.rho = rho_original
        self.assertLess(np.abs(err).max(), 0.025)

    def test_low_density_approx_perpendicular(self):
        """Test the exact numerical answer for :math:`\\gamma_{\\perp}` for Sample II against a low-density approximation."""

        rho_trial = ureg.Quantity(np.logspace(start=np.log10(1e15), stop=np.log10(1e23), num=9), '1/m^3')
        (rho1, gamma1) = gamma_perpendicular_approx(rho_trial, self.sample2)

        gamma0 = ureg.Quantity(np.zeros_like(rho1), 'pN s/m')
        err = np.zeros_like(rho1)
        rho_original = self.sample2.rho       
        for index, rho_ in enumerate(rho1):

            self.sample2.rho = rho_
            gamma0[index] = gamma_perpendicular(theta2norm, self.sample2)

            y0 = gamma0[index].to('pN s/m').magnitude
            y1 = gamma1[index].to('pN s/m').magnitude

            err[index] = (y1 - y0)/y0

        self.sample2.rho = rho_original
        self.assertLess(np.abs(err).max(), 0.025)


    def test_low_density_approx_perpendicular_JIT(self):
        """Reproduce ``test_low_density_approx_perpendicular()`` using JIT versions of all the objects and functions."""

        rho_trial = ureg.Quantity(np.logspace(start=np.log10(1e15), stop=np.log10(1e23), num=9), '1/m^3')
        (rho1, gamma1) = gamma_perpendicular_approx(rho_trial, self.sample2)

        gamma0 = ureg.Quantity(np.zeros_like(rho1), 'pN s/m')
        err = np.zeros_like(rho1)
        rho_original = self.sample2_jit.rho
        for index, rho_ in enumerate(rho1):

            self.sample2_jit.rho = rho_.to('1/m^3').magnitude
            gamma0[index] = gamma_perpendicular_jit(theta2norm_jit, self.sample2_jit).to('pN s/m')

            y0 = gamma0[index].to('pN s/m').magnitude
            y1 = gamma1[index].to('pN s/m').magnitude

            err[index] = (y1 - y0)/y0

        self.sample2_jit.rho = rho_original
        self.assertLess(np.abs(err).max(), 0.025)