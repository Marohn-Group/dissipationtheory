import numpy as np
from dissipationtheory.constants import ureg, epsilon0, qe

class twodimCobject():

    def addsphere(self, r, h, Nz, Nr):
        """Model a sphere of radius $r$ above a ground plane, with a tip-sample
        separation of $h$.  Create image-charge points $r_j$ and voltage-test 
        points $r_k$.  The $N_z$ image-charge points are placed along a vertical
        line extending from $h + 0.1 r$ to $h + 1.90 r$. The $N_r$ voltage-test 
        points are located uniformly around the sphere, starting at the south 
        pole, $\theta = -\pi/2$, and rotating counter clockwise. Initialize the
        values of the image charges at 1.0."""

        # convert to nm and strip units
        h = h.to('nm').magnitude
        r = r.to('nm').magnitude

        # charge locations
        delta_array = np.linspace(start=-0.90, stop=0.90, endpoint=True, num=Nz)
        self.rj = np.array([[0, h + r + r * delta] for delta in delta_array])

        # voltage-test locations
        theta_array = np.linspace(start=-np.pi/2, stop=3*np.pi/2, endpoint=False, num=Nr)
        self.rk = np.array([[r * np.cos(theta), h + r + r * np.sin(theta)] for theta in theta_array])
        
        # save these
        self.info = {'type': 'sphere', 'r': r, 'h': h, 'Nz': Nz, 'Nr': Nr }
        self.title1 = f'sphere, $r$ = {r:0.1f} nm, $h$ = {h:0.1f} nm, $N_z$ = {Nz:d}, $N_r$ = {Nr:d}'
        self.title2 = ''

        # initialize the results, useful for plotting
        self.results = {
            'alpha': 0, 
            'q': np.ones(Nz),
            'S': np.ones(Nz),
            'Sinv': np.ones(Nz),
            'cn': 0, 
            'V': np.zeros(Nr),
            'C0': ureg.Quantity(0, 'F'),
            'C1': ureg.Quantity(0, 'F/m'),
            'C2': ureg.Quantity(0, 'F/m^2')}

    def addtip(self, r, h, L, theta):
        """Model a cone-sphere tip above a ground plane.  The tip-sample
        separation is $h$.  The tip radius is $r$, the cone length is $L$,
        and the cone angle is $theta$. The $N_z$ image-charge points and
        the $N_r$ voltage test points are placed following Xu and coworkers, 
        Xu, J.; Li, J.; Li, W. Calculating Electrostatic Interactions in 
        Atomic Force Microscopy with Semiconductor Samples. *AIP Advances* 
        2019, 9(10): 105308, https://doi.org/10.1063/1.5110482."""

        # convert to nm and strip units
        h = h.to('nm').magnitude
        r = r.to('nm').magnitude 
        L = L.to('nm').magnitude

        # convert to radians
        thetar = np.radians(theta)

        dz = r**2/(2 * (r + h))
        d2 = h + r * (1 - np.sin(thetar))
        
        Nt = 3
        Nc = int(np.floor(np.sqrt((L - r) / dz)))
        Nz = Nt + Nc
        
        rj = np.zeros((Nz, 2))
        rj[0,:] = np.array([0, np.sqrt(2 * r * h + h**2)])
        rj[1,:] = np.array([0, h + r - dz])
        rj[2,:] = np.array([0, h + r])
        for k in np.arange(3, Nz):
            rj[k,0] = 0
            rj[k,1] = rj[k-1,1] + dz * (2 * k - 5)
        
        self.rj = rj

        Nr = Nz
        rk = np.zeros((Nr, 2))
        rk[0,:] = np.array([0, h])
        rk[1,:] = np.array([r * np.sin((np.pi/2 - thetar)/2), h + r * (1 - np.cos((np.pi/2 - thetar)/2))])
        rk[2,:] = np.array([r * np.cos(thetar), h + r * (1 - np.sin(thetar))])
        for k in np.arange(3, Nz):
            rk[k,1] = (rj[k,1] + rj[k-1,1])/2 
            rk[k,0] = r * np.cos(thetar) + (rk[k,1] - d2) * np.tan(thetar)

        self.rk = rk
        
        # save these
        self.info = {'type': 'sphere-tipped cone', 'r': r, 'h': h, 'L': L, 'theta': theta, 'Nz': Nz, 'Nr': Nr}
        self.title1 = f'sphere-tipped cone, $r$ = {r:0.1f} nm, $h$ = {h:0.1f} nm, ' \
            f'$L$ = {L:0.1f} nm, $\\theta$ = {theta:0.1f} deg, $N_r$ = {Nr:d}'
        self.title2 = ''

        # initialize the results, useful for plotting
        self.results = {
            'alpha': 0, 
            'q': np.ones(Nz),
            'S': np.ones(Nz),
            'Sinv': np.ones(Nz),
            'cn': 0, 
            'V': np.zeros(Nr),
            'C0': ureg.Quantity(0, 'F'),
            'C1': ureg.Quantity(0, 'F/m'),
            'C2': ureg.Quantity(0, 'F/m^2')}        
        
    def response_metal(self):
        """Creates the response function, the matrix you multiply the charges
        by to get the voltages.  The model below assumes a metallic plane
        at $z = 0$, which gives rise to a set of images charges.  For charges
        in units of $q_e$, the electronic charge, the voltages will be in 
        units of volts.  In the code below, 

            $r_k$ -- voltage-test points
            $r_j$ -- image-charge points
            
        """

        # shorthand
        rk = self.rk
        
        # reverse the z-coordinates to get the locations of the image charges
        rjp = self.rj
        rjm = np.array([self.rj[:,0], -self.rj[:,1]]).T
        
        # unit matrices
        Idk = np.ones_like(rk[:,0])
        Idj = np.ones_like(rjp[:,0])

        # locations of positive and minus (i.e. image) charges
        
        dxp = np.outer(rk[:,0], Idj.T) - np.outer(Idk, rjp[:,0].T)
        dzp = np.outer(rk[:,1], Idj.T) - np.outer(Idk, rjp[:,1].T)
        
        dxm = np.outer(rk[:,0], Idj.T) - np.outer(Idk, rjm[:,0].T)
        dzm = np.outer(rk[:,1], Idj.T) - np.outer(Idk, rjm[:,1].T)

        # compute unitless constant
        V0 = ureg.Quantity(1, 'V')
        x0 = ureg.Quantity(1, 'nm')
        kR = (qe / (4 * np.pi * epsilon0 * V0 * x0)).to('').magnitude

        # response functions
        
        self.R = kR * (np.power(dxp**2 + dzp**2, -1/2) 
                     - np.power(dxm**2 + dzm**2, -1/2))

        self.R1 = kR * 2 * dzm * np.power(dxm**2 + dzm**2, -3/2)

        self.R2 = kR * (4 * np.power(dxm**2 + dzm**2, -3/2)
                          - 12 * dzm**2 * np.power(dxm**2 + dzm**2, -5/2))
        
        # initial voltage guess
        self.results['V'] = np.dot(self.R, self.results['q'])

        # re-use these internal variables
        # locations of positive and minus (i.e. image) charges
        
        dxp = np.outer(rjp[:,0], Idj.T) - np.outer(Idj, rjp[:,0].T)
        dzp = np.outer(rjp[:,1], Idj.T) - np.outer(Idj, rjp[:,1].T)
        
        dxm = np.outer(rjp[:,0], Idj.T) - np.outer(Idj, rjm[:,0].T)
        dzm = np.outer(rjp[:,1], Idj.T) - np.outer(Idj, rjm[:,1].T)        
    
        
        # response functions
        # keep only the reaction term in H0
        
        with np.errstate(divide='ignore'):
            
            self.H0 = - kR * np.power(dxm**2 + dzm**2, -1/2)
            
            self.H1 = kR * 2 * dzm * np.power(dxm**2 + dzm**2, -3/2)

            self.H2 = kR * (4 * np.power(dxm**2 + dzm**2, -3/2)
                              - 12 * dzm**2 * np.power(dxm**2 + dzm**2, -5/2))
            
        # remove inf elements
        
        self.H0[np.isinf(self.H0)] = 0
        self.H1[np.isinf(self.H1)] = 0
        self.H2[np.isinf(self.H2)] = 0
        
    def solve(self, alpha=0.):
        """Solve for the charges.  The parameter $\alpha$ is used to filter
        the singular values in the inverse.         
        """

        U, S, VT = np.linalg.svd(self.R, full_matrices=False)

        filt = np.diag(np.power(S, 2)/(np.power(S, 2) + alpha**2))
        Sinv = filt * np.diag(np.power(S, -1)) 
        Rinv = np.dot(np.dot(VT.T, Sinv), U.T)
              
        IdNr = np.ones(self.info['Nr'])
        q0 = np.dot(Rinv, IdNr)
        q1 = - np.dot(np.dot(np.dot(Rinv, self.R1), Rinv), IdNr) 
        q2 = - np.dot(np.dot(np.dot(Rinv, self.R2), Rinv), IdNr) \
             + 2 * np.dot(np.dot(Rinv, np.dot(self.R1, np.dot(Rinv, np.dot(self.R1, Rinv)))), IdNr)
        
        # save for diagnosis
        self.Rinv = Rinv       
            
        # constants
        V0 = ureg.Quantity(1, 'V')
        x0 = ureg.Quantity(1, 'nm')
        kc0 = (qe / V0).to('aF')
        kc1 = (kc0 / x0).to('F/m')
        kc2 = (kc1 / x0).to('F/m^2')
        
        # derived quantities
        
        self.results['S'] = S                    # unitless
        self.results['Sinv'] = np.diagonal(Sinv) # unitless
        self.results['cn'] = S.max()/S.min()     # unitless
        self.results['q'] = q0                   # units of qe
        self.results['C0'] = kc0 * q0.sum()      # units of F
        self.results['C1'] = kc1 * q1.sum()      # units of F/m
        self.results['C2'] = kc2 * q2.sum()      # units of F/m^2
        self.results['V'] =  np.dot(self.R, q0)  # units of V
        
        # more derived quantities
        
        self.results['c0'] = q0
        self.results['c1'] = q1
        self.results['c2'] = q2

        # recompute the plotting title string
        self.title2 = r'$\alpha$ = {:0.2e}, cn = {:0.2e}, $C_0$ = {:0.4f} aF, $C_1$ = {:0.2e} F/m, $C_2$ = {:0.2e} F/m$^2$'.format(
            self.results['alpha'],
            self.results['cn'], 
            self.results['C0'].to('aF').magnitude,
            self.results['C1'].to('F/m').magnitude,
            self.results['C2'].to('F/m^2').magnitude)
    
    def plot(self, Nj=0, Nr=0):
        """Plot, from left to right, (a) the voltage test points and the computed 
        image charges, (b) the relative voltage error around the object in parts per
        million, (c) image charge value versus position, and (d) singular values 
        for the response-function matrix."""

        if Nj == 0:
            Nj = self.info['Nz']

        if Nr == 0:
            Nr = self.info['Nr']
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(8.00, 2.5))
        
        fig.suptitle(self.title1 + '\n' + self.title2, fontsize=10)
    
        cmap = plt.get_cmap('RdBu')
        ax1.scatter(self.rj[0:Nj,0], self.rj[0:Nj,1], 
            marker='.', c=self.results['q'][0:Nj], cmap=cmap, 
            alpha=0.5, edgecolors='face',
            vmin=-max(abs(self.results['q'][0:Nj])), 
            vmax=max(abs(self.results['q'][0:Nj])))
        ax1.scatter(self.rk[0:Nr,0], self.rk[0:Nr,1], marker='.')
        ax1.set_xlabel(r'$\rho$ [nm]')
        ax1.set_ylabel('$z$ [nm]')
        ax1.axis('equal')

        ax2.plot(self.rj[0:Nj,1], self.results['q'][0:Nj], '.-')
        ax2.set_xlabel('$(r_j)_{z}$ [nm]')
        ax2.set_ylabel('$q/q_{e}$')
        ax2.set_title(r''.format(), fontsize=10)
        
        ax3.plot(self.results['V'][0:Nr], '.-')
        ax3.set_xlabel('index')
        ax3.set_ylabel(r'$V$ [V]')
        
        ax4.plot(self.results['S'][0:Nj], label='$\Lambda_k$')
        ax4.plot(self.results['Sinv'][0:Nj], label=r'${\mathrm{filt}}(\Lambda_k^{-1})$')
        ax4.set_xlabel('index $k$')
        ax4.set_ylabel('SVD')
        ax4.set_yscale('log')
        ax4.legend(fontsize=6, frameon=False)
        
        fig.tight_layout()
        
        return fig
    
    def forceA(self, Vts=ureg.Quantity(1, 'V')):
        """Compute the force using the standard capacitance-derivative 
        formula.  Return the force in pN, F1, and the force relative
        to $\pi epsilon_0 V^2}$, F2."""
        
        F1 = (0.5 * self.results['C1'] * Vts**2).to('pN').magnitude
        F2 = ((0.5 * self.results['C1'])/(np.pi * epsilon0)).to('').magnitude
        
        return F1, F2
    
    def forceB(self, Vts=ureg.Quantity(1, 'V')):
    
        V = Vts.to('V').magnitude
        const = (qe * ureg.Quantity(1, 'V') / ureg.Quantity(1, 'nm'))
        
        c0 = V * self.results['c0']
        c1 = V * self.results['c1']
        c2 = V * self.results['c2']
        
        H0 = self.H0
        H1 = self.H1
        H2 = self.H2
        
        Idj = np.ones_like(self.rj[:,0])
         
        f1 = -0.5 * np.dot(c0.T, np.dot(H1, c0))
        
        F1 = ureg.Quantity(const * f1, 'pN')
        F2 = (F1/(np.pi * epsilon0 * Vts**2)).to('').magnitude
        
        return F1.to('pN').magnitude, F2
    
    def forceC(self, Vts=ureg.Quantity(1, 'V')):
    
        V = Vts.to('V').magnitude
        const = (qe * ureg.Quantity(1, 'V') / ureg.Quantity(1, 'nm'))
        
        c0 = V * self.results['c0']
        c1 = V * self.results['c1']
        c2 = V * self.results['c2']
        
        H0 = self.H0
        H1 = self.H1
        H2 = self.H2
        
        Idj = np.ones_like(self.rj[:,0])
        
        f1 = -0.5 * (np.dot(c1.T, np.dot(H0, c0)) 
                   + np.dot(c0.T, np.dot(H1, c0))
                   + np.dot(c0.T, np.dot(H0, c1))) + V * np.dot(Idj.T, c1) 
        
        F1 = ureg.Quantity(const * f1, 'pN')
        F2 = (F1/(np.pi * epsilon0 * Vts**2)).to('').magnitude
        
        return F1.to('pN').magnitude, F2    