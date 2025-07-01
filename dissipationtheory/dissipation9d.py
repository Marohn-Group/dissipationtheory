# dissipationtheory8c.py
# John A. Marohn (jam99@cornell.edu)
# Created 2025-06-18

import numpy as np
import matplotlib.pylab as plt

from dissipationtheory.constants import ureg, epsilon0, qe

class twodimCobject():

    def __init__(self, sample):
        """Here sample is a SampleModel1Jit, SampleModel2Jit, 
        SampleModel3Jit, or SampleModel4Jit object."""

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
        
        # write h to sample.cantilever.object
        # jam99 -- this doesn't exist anymore
        # self.sample.cantilever.d = h.to('m').magnitude

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

    def properties(self):

        # Lambda values at 0 frequency and the cantilever frequency
        
        L0dc, L1dc, L2dc = self.solve(0.)
        L0ac, L1ac, L2ac = self.solve(self.sample.cantilever.omega_c)

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
        gamma = c2 * np.real(complex(0,1) * L2ac)
        self.results['gamma [pN s/m]'] = gamma.to('pN s/m').magnitude

        c3 = (4 * np.pi * epsilon0 * V0**2) / (2 * self.zr)
        F11 =  1 * c3 * np.imag(complex(0,1) * L2dc)
        F12 = -1 * c3 * np.imag(complex(0,1) * L2ac)
        self.results['F1(1) [pN/nm]'] = F11.to('pN/nm').magnitude
        self.results['F1(2) [pN/nm]'] = F12.to('pN/nm').magnitude

        c4 = - fc / (2 * kc) 
        dfstat = c4 * F11
        self.results['Delta f stat [Hz]'] = dfstat.to('Hz').magnitude

        c5 = fc / (8 * kc)
        dfdyn = c5 * (F11 + F12)
        self.results['Delta f dyn [Hz]'] = dfdyn.to('Hz').magnitude
        
        self.results['Delta f [Hz]'] = (dfstat + dfdyn).to('Hz').magnitude