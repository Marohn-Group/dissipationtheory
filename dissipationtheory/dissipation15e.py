# dissipationtheory14e.py
# Author: John A. Marohn (jam99@cornell.edu)
# Date: 2025-10-09
# Summary: Convert the torch GPU code from dissipation14e.py into vmapped torch GPU code.

import torch
import numpy as np
import matplotlib.pylab as plt

from dissipationtheory.constants import ureg, qe, epsilon0

from pint import set_application_registry
set_application_registry(ureg)

def get_device(verbose=False):
    """Select the device to use for tensor computations.
    Try CUDA/NVIDIA, then MPS, then CPU."""

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            if verbose:
                print("Using MPS (Apple Silicon GPU)")
        else:
            device = torch.device("cpu")
            if verbose:
                print("Using CPU")

    return device


@torch.compile
def C(sj, rk):
    return 1 / torch.linalg.vector_norm(sj - rk)


Cmatrix = torch.vmap(torch.vmap(C, (None, 0)), (0, None))

@torch.compile
def mysech(x):
    
    return 2 * torch.exp(-x)/(1 + torch.exp(- 2 * x))

@torch.compile
def mycsch(x):
    
    return 2 * torch.exp(-x)/(1 - torch.exp(- 2 * x))

def rpI(
    y: torch.tensor, 
    omega: float, 
    omega0: float, 
    zr: float, 
    kD: float,
    hs: float,
    es: complex,
    ed: complex
) -> torch.complex128:
    """Fresnel coefficient for Sample III object:

        cantilever | vacuum gap | semiconductor (semi-infinite)

    In the code below, `y` is the unitless integration variable.
    """

    Omega = omega / omega0
    
    theta1 = complex(1,0) * torch.sqrt(
        y**2 * (hs / (zr * 1.0e-9) )**2 + (hs * kD)**2 * (1/es + complex(0,1) * Omega))
    
    theta2 = complex(1,0) * y * hs / (zr * 1.0e-9)

    k_over_eta = y / torch.sqrt(
        y**2 + (zr * 1e-9 * kD) ** 2 * (1 / es + complex(0, 1) * Omega)
    )

    p0 = 1 + complex(0,1) * es * Omega
    p1 = k_over_eta / (es * p0)
    p2 = - Omega**2 / (p0 * p0)
    p3 = complex(0,1) * Omega / (ed * p0)
    p4 = complex(0,1) * Omega * k_over_eta / (es * p0**2) 
    p5 = - k_over_eta**2 / (es**2 * p0**2)
    p6 = complex(0,1) * Omega / p0
    p7 = 1 / ed

    t1 = p1 / torch.tanh(theta1)
    n1 = p2 * torch.tanh(theta1) * torch.tanh(theta2) + p3 * torch.tanh(theta1) + p4
    n2 = - 2 * p4 * mysech(theta2) * mysech(theta1)
    n3 = p5 * torch.tanh(theta2) * mysech(theta1) * mycsch(theta1)
    d1 = p6 * torch.tanh(theta1) + torch.tanh(theta2) * (p1 + p7 * torch.tanh(theta1))

    theta_norm = t1 + (n1 + n2 + n3)/d1
    rp = (1 - theta_norm) / (1 + theta_norm)
 
    return rp


@torch.compile
def rp_I_integrator(
    omega: float,
    omega0: float,
    kD: float,
    hs: float,
    es: complex,
    ed: complex,
    sj: torch.tensor,
    rk: torch.tensor,
    dh: torch.tensor,
    pts: int,
    device: torch.device,
) -> torch.tensor:

    y_min = 2.0e-9
    y_max = 2.0e1

    w = torch.linspace(
        torch.log(torch.tensor(y_min)),
        torch.log(torch.tensor(y_max)),
        pts,
        device=device,
    )

    sj = sj + dh
    rk = rk + dh

    zjkref = sj[2] + rk[2]
    x = torch.sqrt((sj[0] - rk[0]) ** 2 + (sj[1] - rk[1]) ** 2) / zjkref

    t0 = torch.exp(w)
    t1 = torch.exp(w - t0)
    t2 = torch.special.bessel_j0(t0 * x)
    t3 = rpI(t0, omega, omega0, zjkref, kD, hs, es, ed)
    
    I0 = t1 * t2 * t3
    I1 = t0 * I0
    I2 = t0 * I1

    K0 = torch.trapezoid(y=I0, x=w) / zjkref**1
    K1 = torch.trapezoid(y=I1, x=w) / zjkref**2
    K2 = torch.trapezoid(y=I2, x=w) / zjkref**3

    return [K0, K1, K2]  # Works if this is a list but NOT a torch.tensor()

# map over sj, rk
# 
# rp_I_integrator arguments: omega, omega0, kD, hs, es, ed, sj, rk, dh, pts, device <== 11 arguments
#
#                        rk: None, None, None, None, None, None, None,    0, None, None, None
#                        sj: None, None, None, None, None, None,    0, None, None, None, None

KmatrixI = \
torch.vmap(
    torch.vmap(
        rp_I_integrator,
        in_dims=(None, None, None, None, None, None, None, 0, None, None, None),
        out_dims=0,
    ),
    in_dims=(None, None, None, None, None, None, 0, None, None, None, None),
    out_dims=0
)

# map over h, omega, sj, rk
# 
# rp_I_integrator arguments: omega, omega0, kD, hs, es, ed, sj, rk, dh, pts, device <== 11 arguments
#
#                        rk: None, None, None, None, None, None, None,    0, None, None, None
#                        sj: None, None, None, None, None, None,    0, None, None, None, None 
#                     omega:    0, None, None, None, None, None, None, None, None, None, None 
#                        dh: None, None, None, None, None, None, None, None,    0, None, None 
# 

KmatrixIhw = \
torch.vmap(
    torch.vmap(
        torch.vmap(
            torch.vmap(
                rp_I_integrator,
                in_dims=(None, None, None, None, None, None, None, 0, None, None, None),
                out_dims=0,
            ),
            in_dims=(None, None, None, None, None, None, 0, None, None, None, None),
            out_dims=0,
        ),
        in_dims=(0, None, None, None, None, None, None, None, None, None, None),
        out_dims=0,
    ),
    in_dims=(None, None, None, None, None, None, None, None, 0, None, None),
    out_dims=0
)

@torch.compile
def rpIII(
    y: torch.tensor, 
    omega: float, 
    omega0: float, 
    zr: float, 
    kD: float,
    es: complex
) -> torch.complex128:
    """Fresnel coefficient for Sample III object:

        cantilever | vacuum gap | semiconductor (semi-infinite)

    In the code below, `y` is the unitless integration variable.
    """

    Omega = omega / omega0
    k_over_eta = y / torch.sqrt(
        y**2 + (zr * 1e-9 * kD) ** 2 * (1 / es + complex(0, 1) * Omega)
    )

    p0 = 1 + complex(0, 1) * es * Omega
    p1 = k_over_eta / (es * p0)
    p6 = complex(0, 1) * Omega / p0

    theta_norm = p6 + p1
    rp = (1 - theta_norm) / (1 + theta_norm)

    return rp

@torch.compile
def rp_III_integrator(
    omega: float,
    omega0: float,
    kD: float,
    es: complex,
    sj: torch.tensor,
    rk: torch.tensor,
    dh: torch.tensor,
    pts: int,
    device: torch.device,
) -> torch.tensor:

    y_min = 2.0e-9
    y_max = 2.0e1

    w = torch.linspace(
        torch.log(torch.tensor(y_min)),
        torch.log(torch.tensor(y_max)),
        pts,
        device=device,
    )

    sj = sj + dh
    rk = rk + dh

    zjkref = sj[2] + rk[2]
    x = torch.sqrt((sj[0] - rk[0]) ** 2 + (sj[1] - rk[1]) ** 2) / zjkref

    t0 = torch.exp(w)
    t1 = torch.exp(w - t0)
    t2 = torch.special.bessel_j0(t0 * x)
    t3 = rpIII(t0, omega, omega0, zjkref, kD, es)

    I0 = t1 * t2 * t3
    I1 = t0 * I0
    I2 = t0 * I1

    K0 = torch.trapezoid(y=I0, x=w) / zjkref**1
    K1 = torch.trapezoid(y=I1, x=w) / zjkref**2
    K2 = torch.trapezoid(y=I2, x=w) / zjkref**3

    return [K0, K1, K2]  # Works if this is a list but NOT a torch.tensor()


# map over sj, rk

KmatrixIII = torch.vmap(
    torch.vmap(
        rp_III_integrator,
        in_dims=(None, None, None, None, None, 0, None, None),
        out_dims=0,
    ),
    in_dims=(None, None, None, None, 0, None, None, None),
    out_dims=0,
)


# map over h, omega, sj, rk

KmatrixIIIhw = \
torch.vmap(
    torch.vmap(
        torch.vmap(
            torch.vmap(
                rp_III_integrator,
                in_dims=(None, None, None, None, None, 0, None, None, None),
                out_dims=0,
            ),
            in_dims=(None, None, None, None, 0, None, None, None, None),
            out_dims=0,
        ),
        in_dims=(0, None, None, None, None, None, None, None, None),
        out_dims=0,
    ),
    in_dims=(None, None, None, None, None, None, 0, None, None),
    out_dims=0
)

class twodimCobject():

    def __init__(self, sample, device):
        """Here sample is a SampleModel3 object."""

        self.sample = sample
        self.device = device

        self.Vr = ureg.Quantity(1, 'V')
        self.zr = ureg.Quantity(1, 'nm')

        self.results = {}
        self.results['Vts [V]'] = self.sample.cantilever.V_ts.to('V').magnitude
        self.keys = ['Vts [V]']

        self.alpha = 0.0  # regularization parameter, unitless
        self.pts = 3000   # number of points used for numerical integration, unitless

        self.results['regularization parameter alpha'] = self.alpha
        self.results['integration points'] = self.pts
        self.keys += ['regularization parameter alpha', 'integration points']
        
    def print_results(self):
        
        df = pd.DataFrame.from_dict(
            self.results,
            orient='index',
            columns=['value'])
        
        try:
            display(df)
        except:
            print(df)
        
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
        r = self.sample.cantilever.R.to('nm').magnitude

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

        # Initialize these results, useful for plotting

        results = {
            'alpha': self.alpha, 
            'q': np.ones(N),
            'S': np.ones(N),
            'Sinv': np.ones(N),
            'cn': 0, 
            'V': np.zeros(M)}

        self.results.update(results)
        
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
        r = self.sample.cantilever.R.to('nm').magnitude   # unitless, nm
        L = self.sample.cantilever.L.to('nm').magnitude   # unitless, nm
        theta = self.sample.cantilever.angle.to('degree') # keep units 
        
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
        
        # Initialize these results, useful for plotting

        results = {
            'alpha': self.alpha, 
            'q': np.ones(Nz),
            'S': np.ones(Nz),
            'Sinv': np.ones(Nz),
            'cn': 0, 
            'V': np.zeros(Nr)}

        self.results.update(results)

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
        
    def set_alpha(self, alpha):
        """Set the regularization parameter alpha.  This is a unitless number."""
        
        self.alpha = alpha
        self.results['alpha'] = alpha

    def set_integration_points(self, pts):
        """Set the number of breakpoints to use in the numerical integration."""
        
        self.pts = pts
        self.results['integration points'] = pts

    def solve(self, dh_tensor, omega_tensor):
        
        sj = torch.as_tensor(self.sj.astype(np.float32), device=self.device)
        rk = torch.as_tensor(self.rk.astype(np.float32), device=self.device)
        
        # The Coulomb matrix doesn't depend on tip-sample 
        # separation or frequency

        Ctensor = Cmatrix(sj, rk) 
        
        if self.sample.type == 3:
            
            omega0 = self.sample.omega0.to('Hz').magnitude
            kD = self.sample.kD.to('1/m').magnitude
            es = self.sample.epsilon_s.to('').magnitude
            
            K0tensor, K1tensor, K2tensor = KmatrixIIIhw(omega_tensor, omega0, kD, es, sj, rk, dh_tensor, self.pts, self.device)

        elif self.sample.type == 1:
        
            omega0 = self.sample.omega0.to('Hz').magnitude
            kD = self.sample.kD.to('1/m').magnitude
            hs = self.sample.h_s.to('m').magnitude
            es = self.sample.epsilon_s.to('').magnitude            
            ed = self.sample.epsilon_d.to('').magnitude
            
            K0tensor, K1tensor, K2tensor = KmatrixIhw(omega_tensor, omega0, kD, hs, es, ed, sj, rk, dh_tensor, self.pts, self.device)

        else:
            
            raise Exception("unknown sample type")
                
        C = Ctensor.cpu().numpy()
        K0 = K0tensor.cpu().numpy() 
        K1 = K1tensor.cpu().numpy()
        K2 = K2tensor.cpu().numpy()
        
        return C, K0, K1, K2

def reduce(C, K0, K1, K2, alpha, cGinv, cG):
    
    m, n, M, N = K0.shape
    
    L0 = np.zeros((m,n), dtype=np.complex64)
    L1 = np.zeros((m,n), dtype=np.complex64)
    L2 = np.zeros((m,n), dtype=np.complex64)
    Q = np.zeros((m,n,N), dtype=np.complex64)    
    Vrms = np.zeros((m,n), dtype=np.float32)
    Vones = np.ones(M, dtype=np.float32)
 
    G0 = C - K0
    
    for index1 in np.arange(m):
        
        for index2 in np.arange(n):

            U, S, VT = np.linalg.svd(G0[index1,index2,:,:], full_matrices=False)

            filt = np.diag(np.power(S, 2)/(np.power(S, 2) + alpha**2))
            Sinv = filt * np.diag(np.power(S, -1))
            G0inv = VT.T @ Sinv @ U.T

            IdN = np.ones(N).T
            IdM = np.ones(M)

            L0[index1,index2] = IdN @ G0inv @ IdM
            L1[index1,index2] = -2 * IdN @ G0inv @ K1[index1,index2,:,:] @ G0inv @ IdM
            L2[index1,index2] = 4 * IdN @ (G0inv @ K2[index1,index2,::] @ G0inv 
                                + 2 * G0inv @ K1[index1,index2,:,:] @ G0inv @ K1[index1,index2,:,:] @ G0inv) @ IdM     

            Q[index1,index2] = cGinv * complex(0,1) * G0inv @ IdM
            V = -1 * complex(0,1) * cG * G0[index1,index2,:,:] @ Q[index1,index2]

            Vrms[index1,index2] = np.std(V - Vones)
        
    return L0, L1, L2, Vrms, Q

def compute_blds_ac(obj, dh__torch, omega_m__torch):
    
    Cp, K0p, K1p, K2p = obj.solve(dh__torch,      omega_m__torch)
    Cm, K0m, K1m, K2m = obj.solve(dh__torch, -1 * omega_m__torch)
    
    L0p, L1p, L2p, _, _ = reduce(Cp, K0p, K1p, K2p, obj.alpha, obj.cGinv, obj.cG)
    L0m, L1m, L2m, _, _ = reduce(Cm, K0m, K1m, K2m, obj.alpha, obj.cGinv, obj.cG)
    
    V0 = obj.sample.cantilever.V_ts.to('V')
    fc = obj.sample.cantilever.f_c.to('Hz')
    kc = obj.sample.cantilever.k_c.to('N/m')
    
    c1 = 0.5 * np.pi * epsilon0 * V0**2
    Fac = c1 * np.imag(complex(0,1) * (L1p + L1m))     
    
    c2 = - 0.25 * (fc * np.pi * epsilon0 * V0**2) / (kc * obj.zr)
    dfac = c2 * (np.imag(complex(0,1) * (L2p + L2m)))
    
    return Fac.to('pN').magnitude, dfac.to('Hz').magnitude

def compute_blds_am(obj, dh__torch, omega_m__torch, omega_am__torch):

    # Lambda values at six frequencies: 
    #
    #   +/- omega_m
    #   +/- (omega_m + omega_am)
    #   +/- (omega_m - omega_am)
    
    C = {}
    K0, K1, K2 = {}, {}, {}
    L0, L1, L2 = {}, {}, {}
    
    omega_list__torch = [ omega_m__torch,
                         -omega_m__torch,
                          omega_m__torch + omega_am__torch,
                         -omega_m__torch - omega_am__torch,
                          omega_m__torch - omega_am__torch,
                         -omega_m__torch + omega_am__torch]
    
    keys = ['a', 'b', 'c', 'd', 'e', 'f']
    
    for (key, omega) in zip(keys, omega_list__torch):
        
        C[key], K0[key], K1[key], K2[key] = obj.solve(dh__torch, omega)
        L0[key], L1[key], L2[key], _, _ = reduce(C[key], K0[key], K1[key], K2[key], obj.alpha, obj.cGinv, obj.cG)
    
    V0 = obj.sample.cantilever.V_ts.to('V')
    fc = obj.sample.cantilever.f_c.to('Hz')
    kc = obj.sample.cantilever.k_c.to('N/m')
    
    c1 = 2 * np.pi * epsilon0 * V0**2
    Fam = c1 * np.imag(complex(0,1) * (L1['a']/16 + L1['b']/16 + L1['c']/64 + L1['d']/64 + L1['e']/64 + L1['f']/64)) 

    c2 = - (fc * np.pi * epsilon0 * V0**2) / (kc * obj.zr)
    dfam= c2 * (np.imag(complex(0,1) * (L2['a']/16 + L2['b']/16 + L2['c']/64 + L2['d']/64 + L2['e']/64 + L2['f']/64)))
    
    return Fam.to('pN').magnitude, dfam.to('Hz').magnitude

def comparetwoK(a, b, label='K'):
    
    for idx, (Ka, Kb) in enumerate(zip(a,b)):

        err_real = (Ka.real-Kb.real)/Ka.real        
        print('Re[{:}[{:d}]] {:+0.9e} vs {:+0.9e}, relative error = {:+3.2e}'.format(
            label, idx, Ka.real, Kb.real, err_real))

    print("")
    for idx, (Ka, Kb) in enumerate(zip(a,b)):
        err_imag = (Ka.imag-Kb.imag)/Ka.imag
        print('Im[{:}[{:d}]] {:+0.9e} vs {:+0.9e}, relative error = {:+3.2e}'.format(
            label,idx, Ka.imag, Kb.imag, err_imag))   

def plotme(h, omega, df, ylabel, title):
    
    fig, axs = plt.subplots(
        ncols=len(h),
        nrows=1,
        sharey=True,
        sharex=True,
        figsize=(2.25 * len(h), 3.25)) 

    axs[0].set_ylabel(ylabel)
    for index, h_ in enumerate(h):
        
        axs[index].semilogx(omega, np.abs(df[index,:]), '-')
        axs[index].set_xlabel('$\omega_{\mathrm{m}}$ [Hz]')
        axs[index].set_title('$h = ${:0.0f} nm'.format(h_))
    
    fig.suptitle(title)
    fig.subplots_adjust(wspace=0.2)
    fig.tight_layout() 

    return fig

if __name__ == "__main__":

    from dissipationtheory.dissipation9a import CantileverModel, SampleModel3
    from dissipationtheory.dissipation9b import SampleModel3Jit

    device = get_device(verbose=True)

    cantilever = CantileverModel(
        f_c=ureg.Quantity(62, "kHz"),
        k_c=ureg.Quantity(2.8, "N/m"),
        V_ts=ureg.Quantity(1, "V"),
        R=ureg.Quantity(60, "nm"),
        angle=ureg.Quantity(20, "degree"),
        L=ureg.Quantity(1000, "nm"),
    )

    sample3 = SampleModel3(
        cantilever=cantilever,
        epsilon_s=ureg.Quantity(complex(3, 0), ""),
        sigma=ureg.Quantity(1e-6, "S/m"),
        rho=ureg.Quantity(1e21, "1/m^3"),
        z_r=ureg.Quantity(1, "nm"),
    )

    sample3_jit = SampleModel3Jit(**sample3.args())

    # 1D torch array of frequencies

    omega_m__torch = torch.as_tensor(
        np.logspace(start=3,stop=6,num=40).astype(np.float32), 
        device=device)

    omega_am__torch = torch.as_tensor(250.0, device=device)

    # 2D torch array of tip displacement vectors

    dh__torch = torch.zeros((6, 3), device=device)

    dh__torch[:,2] = torch.as_tensor(
        np.linspace(0, 50, 6).astype(np.float32), 
        device=device)

    # Numpy arrays of z height and frequency for plotting

    h =  100 + dh__torch[:,2].cpu().numpy()
    omega_m = omega_m__torch.cpu().numpy() 

    # Cantilver at height 100 nm

    obj = twodimCobject(sample3, device)
    obj.addsphere(ureg.Quantity(100,'nm'), 21, 24)
    obj.set_alpha(1.0e-6)
    obj.set_integration_points(21 * 15)

    Fac, dfac = compute_blds_ac(obj, dh__torch, omega_m__torch)

    label_df = 'frequency shift $|\Delta f_{\mathrm{ac}} (\omega_{\mathrm{m}})|$ [Hz]'
    label_F = 'force $|F_{\mathrm{ac}} (\omega_{\mathrm{m}})|$ [pN]'

    _ = plotme(h, omega_m, Fac, label_F, 'voltage modulation: ac')
    _ = plotme(h, omega_m, dfac, label_df, 'voltage modulation: ac')    


    Fam, dfam = compute_blds_am(obj, dh__torch, omega_m__torch, omega_am__torch)

    _ = plotme(h, omega_m, Fam, label_F, 'voltage modulation: ac + am')
    _ = plotme(h, omega_m, dfam, label_df, 'voltage modulation: ac + am')    

    print("Please close the plot windows to exit the program.")
    plt.show()