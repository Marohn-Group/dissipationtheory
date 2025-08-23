# dissipationtheory9d.py
# Author: John A. Marohn (jam99@cornell.edu)
# Date: 2025-07-03
# Summary: Code from dissipation-theory--Study-58.ipynb.  Augment dissipation9e.py to compute 
# the BLDS spectrum using Loring's 07/23/25 pdf report and Marohn's 08/01/25 handwritten notes.
#
# 
# Functions:
#
# - Cmatrix_jit
# - rpIII_jit
# - KmatrixIII_jit
# - KmatrixIV_jit
#
# Classes:
# 
# - twodimCobject

import numpy as np
from numba import jit
import scipy
import matplotlib.pylab as plt
import pandas as pd
from dissipationtheory.constants import ureg, epsilon0, qe

@jit(nopython=True)
def Cmatrix_jit(sj, rk):
    """The unitless Coulomb potential Green's function matrix."""    

    result = np.zeros((len(sj),len(rk)))
    for j, sje in enumerate(sj):
        for k, rke in enumerate(rk):
            result[j,k] = 1 / np.linalg.norm(sje - rke)
    return result

@jit(nopython=True)
def rpIII_jit(y, omega, omega0, zr, kD, es):
    """Fresnel coefficient for Sample III object, a semi-infinite semiconductor.
    
    In the code below, `y` is the unitless integration variable.
    """

    Omega = omega/omega0
    k_over_eta = y / np.sqrt(y**2 + (zr * 1e-9 * kD)**2 * (1/es + complex(0,1) * Omega))

    p0 = 1 + complex(0,1) * es * Omega
    p1 = k_over_eta / (es * p0)
    p6 = complex(0,1) * Omega / p0

    theta_norm = p6 + p1
    rp = (1 - theta_norm) / (1 + theta_norm)
 
    return rp

@jit(nopython=True)
def KmatrixIII_jit(omega, omega0, kD, es, sj, rk, j0s, an):
    """The unitless response-function matrices for a Type III semiconductor sample."""

    y_min = 1.0e-7
    y_max = 20.0
    N = len(an) - 1
    
    K0 = np.zeros((len(sj),len(rk)), dtype=np.complex128)
    K1 = np.zeros((len(sj),len(rk)), dtype=np.complex128)
    K2 = np.zeros((len(sj),len(rk)), dtype=np.complex128)

    # Loop over image-charge points
    
    for k, rke in enumerate(rk):    

        # Loop over voltage-test points
        
        for j, sje in enumerate(sj):

            zjkref = sje[2] + rke[2]
            x = np.sqrt((sje[0] - rke[0])**2 + (sje[1] - rke[1])**2) / zjkref

            # Determine breakpoints
            
            mask = j0s/(x + 1.0e-6) < y_max
            yb = np.hstack(
                (np.array([y_min]),
                 j0s[mask] / x,
                 np.array([y_max])))
            
            result0 = np.zeros(len(yb)-1, dtype=np.complex128)
            result1 = np.zeros(len(yb)-1, dtype=np.complex128)
            result2 = np.zeros(len(yb)-1, dtype=np.complex128)

            # Loop over subintervals
            
            for index in np.arange(len(yb)-1):
                
                y_vector = np.linspace(yb[index], yb[index+1], N+1)
                dy = (yb[index+1] - yb[index])/N
            
                integral0 = np.zeros_like(y_vector, dtype=np.complex128)
                integral1 = np.zeros_like(y_vector, dtype=np.complex128)
                integral2 = np.zeros_like(y_vector, dtype=np.complex128)

                # Loop over y-axis points in the subinterval
                
                for m, y in enumerate(y_vector):

                    rp = rpIII_jit(y, omega, omega0, zjkref, kD, es)
                    
                    integral0[m] = np.exp(-y) * scipy.special.j0(y * x) * rp
                    integral1[m] = y * integral0[m]
                    integral2[m] = y * integral1[m]

                # Sum with Newton-Cotes weights
                
                result0[index] = dy * (an * integral0).sum()  
                result1[index] = dy * (an * integral1).sum()
                result2[index] = dy * (an * integral2).sum()

            K0[j,k] = result0.sum() / zjkref**1
            K1[j,k] = result1.sum() / zjkref**2
            K2[j,k] = result2.sum() / zjkref**3
            
    return K0, K1, K2

@jit(nopython=True)
def KmatrixIV_jit(sj, rk):
    """The unitless Green's function matrices for an image charge."""

    K0 = complex(1,0) * np.zeros((len(sj),len(rk)))
    K1 = complex(1,0) * np.zeros((len(sj),len(rk)))
    K2 = complex(1,0) * np.zeros((len(sj),len(rk)))

    for k, rke in enumerate(rk):
        
        # location of image charge
        rkei = rke.copy()
        rkei[2] = -1 * rkei[2]
        
        for j, sje in enumerate(sj):
            
            # shorthand
            Rinv = np.power((sje - rkei).T @ (sje - rkei), -1/2)
            
            K0[j,k] = complex( 1,0) * Rinv
            K1[j,k] = complex( 1,0) * (sje[2] + rke[2]) *  np.power(Rinv, 3)
            K2[j,k] = complex(-1,0) * (np.power(Rinv, 3) - 3 * np.power(sje[2] + rke[2], 2) * np.power(Rinv, 5))
            
    return K0, K1, K2

class twodimCobject():

    def __init__(self, sample):
        """Here sample is a SampleModel3Jit or SampleModel4Jit object."""

        self.sample = sample

        self.Vr = ureg.Quantity(1, 'V')
        self.zr = ureg.Quantity(1, 'nm')
        
    @property
    def cG(self):
        return (qe / (4 * np.pi * epsilon0 * self.Vr * self.zr)).to('dimensionless').magnitude
    
    @property
    def cGinv(self):
        return 1/self.cG

    def addsphere(self, h, N, M, theta_start=-np.pi/2, theta_stop=3*np.pi/2, theta_endpoint=False):
        """Model a sphere of radius $r$ above a ground plane, with a tip-sample
        separation of $h$.  Create image-charge points $r_j$ and voltage-test 
        points $r_k$.  The $N$ image-charge points are placed along a vertical
        line extending from $h + 0.1 r$ to $h + 1.90 r$. The $M$ voltage-test 
        points are located uniformly around the sphere, starting at the south 
        pole, $\theta = -\pi/2$, and rotating counter clockwise. Initialize the
        values of the image charges at 1.0.  Creates two arrays: 
        (a) self.sj, the voltage-test points, and (b) self.rk, the image-charge
        points, with the coordinates in nanometers. 
        """

        # read from sample.cantilever object
        r = ureg.Quantity(self.sample.cantilever.R, 'm').to('nm').magnitude

        # convert to nm and strip units
        h = h.to('nm').magnitude
        
        # charge locations
        delta_array = np.linspace(start=-0.90, stop=0.90, endpoint=True, num=N)
        self.rk = np.array([[0, 0, h + r + r * delta] for  delta in delta_array])

        # voltage-test locations
        theta_array = np.linspace(start=theta_start, stop=theta_stop, endpoint=theta_endpoint, num=M)
        self.sj = np.array([[r * np.cos(theta), 0, h + r + r * np.sin(theta)] for theta in theta_array])
        
        # save these
        self.info = {'type': 'sphere', 
                     'r [nm]': r, 
                     'h [nm]': h, 
                     'N': N,
                     'M': M }

        self.title1 = f'sphere, $r$ = {r:0.1f} nm, $h$ = {h:0.1f} nm, $N$ = {N:d} image charges, $M$ = {M:d} test points'
        self.title2 = ''

        # initialize the results, useful for plotting
        self.results = {
            'alpha': 0, 
            'q': np.ones(N),
            'S': np.ones(N),
            'Sinv': np.ones(N),
            'cn': 0, 
            'V': np.zeros(M)}  

    def addtip(self, h):
        """Model a cone-sphere tip above a ground plane.  The tip-sample
        separation is $h$.  The tip radius is $r$, the cone length is $L$,
        and the cone angle is $theta$ (read from self.sample.cantilever). 
        The $N_z$ image-charge points and the $N_r$ voltage test points 
        are placed following Xu and coworkers, Xu, J.; Li, J.; Li, W. 
        Calculating Electrostatic Interactions in Atomic Force Microscopy 
        with Semiconductor Samples. *AIP Advances* 2019, 9(10): 105308, 
        https://doi.org/10.1063/1.5110482."""
        
 
        # convert h to nm and strip units
        h = h.to('nm').magnitude
        
        # read from sample.cantilever object
        r = ureg.Quantity(self.sample.cantilever.R, 'm').to('nm').magnitude  # unitless, nm
        L = ureg.Quantity(self.sample.cantilever.L, 'm').to('nm').magnitude  # unitless, nm
        theta = ureg.Quantity(self.sample.cantilever.angle, 'degree')        # keep units 
        
        # convert to radians 
        thetar = theta.to('radian').magnitude

        dz = r**2/(2 * (r + h))
        d2 = h + r * (1 - np.sin(thetar))
        
        Nt = 3
        Nc = int(np.floor(np.sqrt((L - r) / dz)))
        Nz = Nt + Nc
        
        rk = np.zeros((Nz, 3))
        rk[0,:] = np.array([0, 0, np.sqrt(2 * r * h + h**2)])
        rk[1,:] = np.array([0, 0, h + r - dz])
        rk[2,:] = np.array([0, 0, h + r])
        for k in np.arange(3, Nz):
            rk[k,0] = 0
            rk[k,1] = 0
            rk[k,2] = rk[k-1,2] + dz * (2 * k - 5)
        
        self.rk = rk

        Nr = Nz
        sj = np.zeros((Nr, 3))
        sj[0,:] = np.array([0, 0, h])
        sj[1,:] = np.array([r * np.sin((np.pi/2 - thetar)/2), 0, h + r * (1 - np.cos((np.pi/2 - thetar)/2))])
        sj[2,:] = np.array([r * np.cos(thetar), 0, h + r * (1 - np.sin(thetar))])
        for k in np.arange(3, Nz):
            sj[k,2] = (rk[k,2] + rk[k-1,2])/2
            sj[k,1] = 0
            sj[k,0] = r * np.cos(thetar) + (sj[k,2] - d2) * np.tan(thetar)

        self.sj = sj
        
        # save these
        self.info = {'type': 'sphere-tipped cone', 
                     'r [nm]': r, 
                     'h [nm]': h, 
                     'L [nm]': L, 
                     'theta [degree]': theta.to('degree').magnitude, 
                     'N': Nz,
                     'M': Nr}
        
        self.title1 = f'sphere-tipped cone, $r$ = {r:0.1f} nm, $h$ = {h:0.1f} nm, ' \
            f'$L$ = {L:0.1f} nm, $\\theta$ = {theta:0.1f}, $N_z$ = {Nz:d}, $N_r$ = {Nr:d}'
        self.title2 = ''
        
        # initialize the results, useful for plotting
        self.results = {
            'alpha': 0, 
            'q': np.ones(Nz),
            'S': np.ones(Nz),
            'Sinv': np.ones(Nz),
            'cn': 0, 
            'V': np.zeros(Nr)}

    def solve(self, omega, alpha=0.):
        """Solve for the charges.  The parameter $\alpha$ is used to filter
        the singular values in the inverse.  The parameter omega is the unitless
        cantilever frequency in rad/s.       
        """

        C = Cmatrix_jit(self.sj, self.rk)

        if self.sample.type == 4:
            
            K0, K1, K2 = KmatrixIV_jit(self.sj, self.rk)
            
        elif self.sample.type == 3:

            j0s = scipy.special.jn_zeros(0,100.)
            an, _ = scipy.integrate.newton_cotes(20, 1)
            
            args = {'omega': omega, 
                'omega0': self.sample.omega0,
                'kD': self.sample.kD, 
                'es': self.sample.epsilon_s, 
                'sj': self.sj, 
                'rk': self.rk, 
                'j0s': j0s, 
                'an': an}
        
            K0, K1, K2 = KmatrixIII_jit(**args)
            
        else:

            raise Exception("unknown sample type")
               
        G0 = C - K0
        
        U, S, VT = np.linalg.svd(G0, full_matrices=False)
        filt = np.diag(np.power(S, 2)/(np.power(S, 2) + alpha**2))
        Sinv = filt * np.diag(np.power(S, -1))
        G0inv = VT.T @ Sinv @ U.T

        self.results['S'] = S                      # unitless
        self.results['Sinv'] = np.diagonal(Sinv)   # unitless
        self.results['cn'] = S.max()/S.min()       # unitless
        
        IdN = np.ones(self.info['N']).T
        IdM = np.ones(self.info['M'])

        Q = self.cGinv * complex(0,1) * G0inv @ IdM
        V = -1 * complex(0,1) * self.cG * G0 @ Q
  
        self.results['q'] = np.imag(Q) # units of qe 
        self.results['V'] = np.real(V) # units of Vr

        Vrms = np.std(V - np.ones_like(V))
        
        self.results['Vrms [ppm]'] = 1e6 * np.real(Vrms) # units of Vr
        
        L0 = IdN @ G0inv @ IdM
        L1 = -2 * IdN @ G0inv @ K1 @ G0inv @ IdM
        L2 = 4 * IdN @ (G0inv @ K2 @ G0inv + 2 * G0inv @ K1 @ G0inv @ K1 @ G0inv) @ IdM

        self.L0 = L0
        self.L1 = L1
        self.L2 = L2

        return L0, L1, L2
    
    def plot(self, N=0, M=0):
        """Plot, from left to right, (a) the voltage test points and the computed 
        image charges, (b) the relative voltage error around the object in parts per
        million, (c) image charge value versus position, and (d) singular values 
        for the response-function matrix."""

        if N == 0:
            N = self.info['N']

        if M == 0:
            M = self.info['M']
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(8.00, 2.5))
        
        fig.suptitle(self.title1 + '\n' + self.title2, fontsize=10)
    
        cmap = plt.get_cmap('RdBu')
        ax1.scatter(self.rk[0:N,0], self.rk[0:N,2], 
            marker='.', c=self.results['q'][0:N], cmap=cmap, 
            alpha=0.5, edgecolors='face',
            vmin=-max(abs(self.results['q'][0:N])), 
            vmax=max(abs(self.results['q'][0:N])))
        ax1.scatter(self.sj[0:M,0], self.sj[0:M,2], marker='.')
        ax1.set_xlabel('$x$ [nm]')
        ax1.set_ylabel('$z$ [nm]')
        ax1.axis('equal')

        ax2.plot(self.rk[0:N,2], self.results['q'][0:N], '.-')
        ax2.set_xlabel('$(r_j)_{z}$ [nm]')
        ax2.set_ylabel('$q/q_{e}$')
        ax2.set_title(r''.format(), fontsize=10)

        V = self.results['V'][0:M]
        ax3.plot(1e6 * (V - np.ones_like(V)), '.-')
        ax3.set_xlabel('index')
        ax3.set_ylabel(r'$\delta V/V_0$ [ppm]')
        
        ax4.plot(self.results['S'][0:N], label='$\Lambda_k$')
        ax4.plot(self.results['Sinv'][0:N], label=r'${\mathrm{filt}}(\Lambda_k^{-1})$')
        ax4.set_xlabel('index $k$')
        ax4.set_ylabel('SVD')
        ax4.set_yscale('log')
        ax4.legend(fontsize=6, frameon=False)
        
        fig.tight_layout()
        
        return fig

    def properties_dc(self, alpha=0.):
        """Compute the cantilever force, friction, and frequency shift when a 
        DC voltage is applied to the cantilever.  The parameter alpha is used to filter
        the singular values in the inverse.  The results are stored in self.results, 
        which is a dictionary with the following keys:

        - 'L0dc', 'L1dc', 'L2dc': the unitless response functions at DC
        - 'L0ac', 'L1ac', 'L2ac': the unitless response functions at the cantilever frequency
        - 'C0 [aF]', 'C1 [pF/m]', 'C2 [mF/m^2]': the capacitance and its derivatives
        - 'Fdc [pN]': the DC force on the cantilever
        - 'gamma [pN s/m]': the friction coefficient
        - 'F1(1) [pN/nm]', 'F1(2) [pN/nm]': two contributions to the force derivative
        - 'Delta f dc ss [Hz]': the steady-state frequency shift
        - 'Delta f dc dyn [Hz]': the dynamic frequency shift
        - 'Delta f dc [Hz]': the total frequency shift

        The results are in the units of the ureg module, which is a wrapper around
        the Pint unit system.

        """

        # Lambda values at 0 frequency and the cantilever frequency
        
        L0dc, L1dc, L2dc = self.solve(0., alpha)
        L0ac, L1ac, L2ac = self.solve(self.sample.cantilever.omega_c, alpha)

        for key, val in zip(
            ['L0dc', 'L1dc', 'L2dc', 'L0ac', 'L1ac', 'L2ac'],
            [L0dc, L1dc, L2dc, L0ac, L1ac, L2ac]):

            self.results[key] = val

        # Capacitance and derivatives
        
        c0 = self.cGinv * (qe / self.Vr)
        C0 = (c0 / self.zr**0) * np.imag(complex(0,1) * L0dc) 
        C1 = (c0 / self.zr**1) * np.imag(complex(0,1) * L1dc) 
        C2 = (c0 / self.zr**2) * np.imag(complex(0,1) * L2dc)
        
        for key, values, unit in zip(
            ['C0 [aF]', 'C1 [pF/m]', 'C2 [mF/m^2]'],
            [C0, C1, C2],
            ['aF', 'pF/m', 'mF/m^2']):

            self.results[key] = values.to(unit).magnitude

        # Shorthand
        
        V0 = ureg.Quantity(self.sample.cantilever.V_ts, 'V')
        wc = ureg.Quantity(self.sample.cantilever.omega_c, 'Hz')
        fc = ureg.Quantity(self.sample.cantilever.f_c, 'Hz')
        kc = ureg.Quantity(self.sample.cantilever.k_c, 'N/m')
        
        # Other properties

        c1 = 2 * np.pi * epsilon0 * V0**2
        Fdc = c1 * np.imag(complex(0,1) * L1dc) 
        self.results['Fdc [pN]'] = Fdc.to('pN').magnitude

        c2 = (4 * np.pi * epsilon0 * V0**2) / (8 * wc * self.zr)
        gamma = c2 * np.real(complex(0,1) * L2ac) * -1 # <== jam99: check this sign
        self.results['gamma [pN s/m]'] = gamma.to('pN s/m').magnitude

        c3 = (4 * np.pi * epsilon0 * V0**2) / (2 * self.zr)
        F11 =  1 * c3 * np.imag(complex(0,1) * L2dc)
        F12 = -1 * c3 * np.imag(complex(0,1) * L2ac)
        self.results['F1(1) [pN/nm]'] = F11.to('pN/nm').magnitude
        self.results['F1(2) [pN/nm]'] = F12.to('pN/nm').magnitude

        c4 = - fc / (2 * kc) 
        dfstat = c4 * F11
        self.results['Delta f dc ss [Hz]'] = dfstat.to('Hz').magnitude

        c5 = fc / (8 * kc)
        dfdyn = c5 * (F11 + F12)
        self.results['Delta f dc dyn [Hz]'] = dfdyn.to('Hz').magnitude
        
        self.results['Delta f dc [Hz]'] = (dfstat + dfdyn).to('Hz').magnitude

    def properties_ac(self, omega_m, alpha=0.):
        """Compute the cantilever force and steady-state frequency shift when an 
        AC voltage is applied to the cantilever, with omega_m the voltage oscillation
        frequency. The parameter alpha is used to filter the singular values in 
        the inverse.  The results are stored in self.results, which is a dictionary
        with the following keys:
        
        - 'L0wmp', 'L1wmp', 'L2wmp': the unitless response functions at +omega_m
        - 'L0wmm', 'L1wmm', 'L2wmm': the unitless response functions at -omega_m
        - 'Fac [pN]': the steady-state force on the cantilever
        - 'Delta f ac [Hz]': the steady-state frequency shift of the cantilever

        The results are in the units of the ureg module, which is a wrapper around
        the Pint unit system.`
        
        """

        # Lambda values at plus and minus omega_m

        L0wmp, L1wmp, L2wmp = self.solve( omega_m, alpha)
        L0wmm, L1wmm, L2wmm = self.solve(-omega_m, alpha)

        for key, val in zip(
            ['L0wmp', 'L1wmp', 'L2wmp', 'L0wmm', 'L1wmm', 'L2wmm'],
            [L0wmp, L1wmp, L2wmp, L0wmm, L1wmm, L2wmm]):

            self.results[key] = val

        # Shorthand
        
        V0 = ureg.Quantity(self.sample.cantilever.V_ts, 'V')
        fc = ureg.Quantity(self.sample.cantilever.f_c, 'Hz')
        kc = ureg.Quantity(self.sample.cantilever.k_c, 'N/m')
        
        # Properties

        c1 = 0.5 * np.pi * epsilon0 * V0**2
        Fac = c1 * np.imag(complex(0,1) * (L1wmp + L1wmm)) 
        self.results['Fac [pN]'] = Fac.to('pN').magnitude

        c2 = - 0.25 * (fc * np.pi * epsilon0 * V0**2) / (kc * self.zr)
        dfac = c2 * (np.imag(complex(0,1) * (L2wmp + L2wmm)))
        self.results['Delta f ac [Hz]'] = dfac.to('Hz').magnitude

    def properties_am(self, omega_m, omega_am, alpha=0.):
        """Compute the cantilever force and steady-state frequency shift when an 
        amplitude-modulated AC voltage is applied to the cantilever, with omega_m 
        the voltage oscillation frequency and omega_am the amplitude oscillation
        frequency. The parameter alpha is used to filter the singular values in 
        the inverse.  The results are stored in self.results, which is a dictionary
        including the following keys:
        
        - 'Fam [pN]': the steady-state force on the cantilever
        - 'Delta f am [Hz]': the steady-state frequency shift of the cantilever

        The results are in the units of the ureg module, which is a wrapper around
        the Pint unit system.`
        
        """

        # Lambda values at six frequencies: 
        #
        #   +/- omega_m
        #   +/- (omega_m + omega_am)
        #   +/- (omega_m - omega_am)

        L0a, L1a, L2a = self.solve( omega_m, alpha)
        L0b, L1b, L2b = self.solve(-omega_m, alpha)
        L0c, L1c, L2c = self.solve( omega_m + omega_am, alpha)
        L0d, L1d, L2d = self.solve(-omega_m - omega_am, alpha)
        L0e, L1e, L2e = self.solve( omega_m - omega_am, alpha)
        L0f, L1f, L2f = self.solve(-omega_m + omega_am, alpha)


        for key, val in zip(
            ['L0a', 'L1a', 'L2a',
             'L0b', 'L1b', 'L2b',
             'L0c', 'L1c', 'L2c',
             'L0d', 'L1d', 'L2d',
             'L0e', 'L1e', 'L2e',
             'L0f', 'L1f', 'L2f'],
            [L0a, L1a, L2a, 
             L0b, L1b, L2b, 
             L0c, L1c, L2c, 
             L0d, L1d, L2d, 
             L0e, L1e, L2e, 
             L0f, L1f, L2f]):

            self.results[key] = val

        # Shorthand
        
        V0 = ureg.Quantity(self.sample.cantilever.V_ts, 'V')
        fc = ureg.Quantity(self.sample.cantilever.f_c, 'Hz')
        kc = ureg.Quantity(self.sample.cantilever.k_c, 'N/m')
        
        # Properties

        c1 = 2 * np.pi * epsilon0 * V0**2
        Fam = c1 * np.imag(complex(0,1) * (L1a/16 + L1b/16 + L1c/64 + L1d/64 + L1e/64 + L1f/64)) 
        self.results['Fam [pN]'] = Fam.to('pN').magnitude

        c2 = - (fc * np.pi * epsilon0 * V0**2) / (kc * self.zr)
        dfam= c2 * (np.imag(complex(0,1) * (L2a/16 + L2b/16 + L2c/64 + L2d/64 + L2e/64 + L2f/64)))
        self.results['Delta f am [Hz]'] = dfam.to('Hz').magnitude        