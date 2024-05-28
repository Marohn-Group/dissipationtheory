import unittest
from dissipationtheory.constants import ureg
from dissipationtheory.dissipation import CantileverModel, SampleModel1, theta1norm, gamma_parallel, gamma_parallel_approx
from dissipationtheory.dissipation import CantileverModelJit, SampleModel1Jit, theta1norm_jit, gamma_parallel_jit
import pandas as pd
import numpy as np
import os
from copy import deepcopy 

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

    def test_cantilever_model(self):
        """Enter the tip-sample separation in nanometers and check that it comes out 
        right in micrometers.
        """

        self.assertEqual(0.300, self.cantilever.d.to('um').magnitude)

    def test_Lekkala2013nov_Fig7b(self):
        """Reproduce the friction calculation in Lekkala2013nov, Fig. 7(b), the 
        :math:`\\mu = 2.7 \\times 10^{-10} \\: \\mathrm{m}^2 \\: \\mathrm{V}^{-1} \\: \\mathrm{s}^{-1}` case.
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

        # make a copy since we are going to overwrite elements
        sample1 = deepcopy(self.sample1)

        for index, rho__ in enumerate(rho1_):

            sample1.rho = rho__
            gamma1[index] = gamma_parallel(theta1norm, sample1)

            x = rho__.to('1/m^3').magnitude
            y0 = gamma0[index].to('pN s/m').magnitude
            y1 = gamma1[index].to('pN s/m').magnitude

            err[index] = (y1 - y0)/y0

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

        # Make a local copy since we are going to overwrite elements.
        # Can't use deepcopy() on a jit-class object, to copy elements manually.

        sample1_jit = SampleModel1Jit(
            cantilever = self.cantilever_jit,
            h_s = self.sample1_jit.h_s,
            epsilon_s = self.sample1_jit.epsilon_s,
            mu = self.sample1_jit.mu,
            rho = self.sample1_jit.rho,
            epsilon_d = self.sample1_jit.epsilon_d,
            z_r = self.sample1_jit.z_r
        )

        for index, rho__ in enumerate(rho1_):

            sample1_jit.rho = rho__.to('1/m^3').magnitude
            gamma1[index] = gamma_parallel_jit(theta1norm_jit, sample1_jit).to('pN s/m')

            x = rho__.to('1/m^3').magnitude
            y0 = gamma0[index].to('pN s/m').magnitude
            y1 = gamma1[index].to('pN s/m').magnitude

            err[index] = (y1 - y0)/y0

        self.assertLess(np.abs(err).mean(), 0.15)

    def test_low_density_approx(self):
        """Test the exact numerical answer against the low-density approximation.
        See ``development/dissipation-theory--Study-4.html``.
        For this test, we create a list of densities below the turning point where the low-density expansion is valid.
        The maximum difference between the exact answer and the low-density expansion should be less than 2.5 percent.
        """

        rho_trial = ureg.Quantity(np.logspace(start=np.log10(1e15), stop=np.log10(1e23), num=9), '1/m^3')
        (rho1, gamma1) = gamma_parallel_approx(rho_trial, self.sample1)

        gamma0 = ureg.Quantity(np.zeros_like(rho1), 'pN s/m')
        err = np.zeros_like(rho1)

        # make a copy since we are going to overwrite elements
    
        sample1 = deepcopy(self.sample1)

        for index, rho_ in enumerate(rho1):

            sample1.rho = rho_
            gamma0[index] = gamma_parallel(theta1norm, sample1)

            x = rho_.to('1/m^3').magnitude
            y0 = gamma0[index].to('pN s/m').magnitude
            y1 = gamma1[index].to('pN s/m').magnitude

            err[index] = (y1 - y0)/y0

        self.assertLess(np.abs(err).max(), 0.025)

    def test_low_density_approx_JIT(self):
        """Reproduce ``test_low_density_approx`` using JIT versions of all the objects and functions."""

        rho_trial = ureg.Quantity(np.logspace(start=np.log10(1e15), stop=np.log10(1e23), num=9), '1/m^3')
        (rho1, gamma1) = gamma_parallel_approx(rho_trial, self.sample1)

        gamma0 = ureg.Quantity(np.zeros_like(rho1), 'pN s/m')
        err = np.zeros_like(rho1)

        # Make a local copy since we are going to overwrite elements.
        # Can't use deepcopy() on a jit-class object, to copy elements manually.

        sample1_jit = SampleModel1Jit(
            cantilever = self.cantilever_jit,
            h_s = self.sample1_jit.h_s,
            epsilon_s = self.sample1_jit.epsilon_s,
            mu = self.sample1_jit.mu,
            rho = self.sample1_jit.rho,
            epsilon_d = self.sample1_jit.epsilon_d,
            z_r = self.sample1_jit.z_r
        )

        for index, rho_ in enumerate(rho1):

            sample1_jit.rho = rho_.to('1/m^3').magnitude
            gamma0[index] = gamma_parallel_jit(theta1norm_jit, sample1_jit).to('pN s/m')

            x = rho_.to('1/m^3').magnitude
            y0 = gamma0[index].to('pN s/m').magnitude
            y1 = gamma1[index].to('pN s/m').magnitude

            err[index] = (y1 - y0)/y0

        self.assertLess(np.abs(err).max(), 0.025)