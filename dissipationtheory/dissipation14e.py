# dissipationtheory14e.py
# Author: John A. Marohn (jam99@cornell.edu)
# Date: 2025-09-23
# Summary: Convert the compiled CPU code from dissipation13e.py into torch GPU code.

import torch
import numpy as np
import scipy
import matplotlib.pylab as plt
import pandas as pd
import concurrent
import multiprocess as mp
import multiprocessing as mpg

from dissipationtheory.constants import ureg, qe, epsilon0
from dissipationtheory.dissipation9a import CantileverModel, SampleModel3
from dissipationtheory.dissipation9b import SampleModel3Jit, integrand3jit

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


def rpIII(
    y: torch.tensor, omega: float, omega0: float, zr: float, kD: float, es: complex
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


def rp_III_integrator(
    omega: float,
    omega0: float,
    kD: float,
    es: complex,
    sj: torch.tensor,
    rk: torch.tensor,
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


KmatrixIII = torch.vmap(
    torch.vmap(
        rp_III_integrator,
        in_dims=(None, None, None, None, None, 0, None, None),
        out_dims=0,
    ),
    in_dims=(None, None, None, None, 0, None, None, None),
    out_dims=0,
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

    def solve(self, omega):
        """Solve for the charges.  The parameter $\alpha$ is used to filter
        the singular values in the inverse.  The parameter omega is the unitless
        cantilever frequency in rad/s.       
        """
        
        sj = torch.as_tensor(self.sj.astype(np.float32), device=self.device)
        rk = torch.as_tensor(self.rk.astype(np.float32), device=self.device)
        
        Ctensor = Cmatrix(sj, rk)
        
        if self.sample.type == 3:
            
            omega0 = self.sample.omega0.to('Hz').magnitude
            kD = self.sample.kD.to('1/m').magnitude
            es = self.sample.epsilon_s.to('').magnitude
            
            K0tensor, K1tensor, K2tensor = KmatrixIII(omega, omega0, kD, es, sj, rk, self.pts, self.device)
            
        else:
            
            raise Exception("unknown sample type")
            
        K0 = K0tensor.cpu().numpy() 
        K1 = K1tensor.cpu().numpy()
        K2 = K2tensor.cpu().numpy()
        C = Ctensor.cpu().numpy()
        
        G0 = C - K0
        alpha = self.alpha
    
        U, S, VT = np.linalg.svd(G0, full_matrices=False) # svd not implemented yet in torch for M1
        
        filt = np.diag(np.power(S, 2)/(np.power(S, 2) + alpha**2))
        Sinv = filt * np.diag(np.power(S, -1))
        G0inv = VT.T @ Sinv @ U.T
        
        self.results['S'] = S                # unitless
        self.results['Sinv'] = np.diag(Sinv) # unitless
        self.results['cn'] = S.max()/S.min() # unitless
        
        IdN = np.ones(self.info['N']).T
        IdM = np.ones(self.info['M'])
        
        L0 = IdN @ G0inv @ IdM
        L1 = -2 * IdN @ G0inv @ K1 @ G0inv @ IdM
        L2 = 4 * IdN @ (G0inv @ K2 @ G0inv + 2 * G0inv @ K1 @ G0inv @ K1 @ G0inv) @ IdM

        Q = self.cGinv * complex(0,1) * G0inv @ IdM
        V = -1 * complex(0,1) * self.cG * G0 @ Q

        self.results['q'] = np.imag(Q) # units of qe 
        self.results['V'] = np.real(V) # units of Vr
                
        Vrms = np.std(V - np.ones_like(V))
        
        self.results['Vrms [ppm]'] = 1e6 * np.real(Vrms) # units of Vr        
        
        return L0, L1, L2

class twodimCobjectExperimental(twodimCobject):

    def properties_blds_mockup1(self, moniker_list, omega_list):

        mpg.set_start_method("fork", force=True) # works in fork mode!

        L = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:

            futures = []

            for omega in omega_list:
                future = executor.submit(self.solve, omega)
                futures.append(future)

            for moniker, future in zip(moniker_list, futures):
                L[moniker] = future.result()

        return L

    def properties_blds_mockup2(self, omega):

        def worker(obj, x, queue):
            queue.put((x, obj.solve(x)))

        mp.set_start_method('spawn', force=True)

        q = mp.Queue()
        p = mp.Process(target=worker, args=(self, omega, q))
        p.start()
        p.join()

        if not q.empty():
            return(q.get())
        else:  
            return("queue is empty")

    def properties_blds_mockup3(self, moniker_list, omega_list):

        mpg.set_start_method("spawn", force=True) # crashes in fork mode

        L = {}

        with concurrent.futures.ProcessPoolExecutor() as executor:

            futures = []

            for omega in omega_list:
                future = executor.submit(self.solve, omega)
                futures.append(future)

            for moniker, future in zip(moniker_list, futures):
                L[moniker] = future.result()

        return L

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

if __name__ == "__main__":

    from dissipationtheory.dissipation13e import Cmatrix_jit, KmatrixIII_jit
    from dissipationtheory.dissipation13e import twodimCobject as twodimCobjectPrior

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
        epsilon_s=ureg.Quantity(complex(20, 0), ""),
        sigma=ureg.Quantity(1e-6, "S/m"),
        rho=ureg.Quantity(1e19, "1/m^3"),
        z_r=ureg.Quantity(1, "nm"),
    )

    sample3_jit = SampleModel3Jit(**sample3.args())

    loc1_nm = np.array([[0, 0, 20], [0, 0, 30]], dtype=np.float64)
    loc2_nm = np.array([[0, 20, 20], [0, 30, 30]], dtype=np.float64)

    loc1_m = 1e-9 * loc1_nm
    loc2_m = 1e-9 * loc2_nm

    omega = 1e5 * 2 * np.pi

    params3_jit = {
        "integrand": integrand3jit,
        "sample": sample3_jit,
        "omega": omega,
        "location1": loc1_m,
        "location2": loc2_m,
    }

    j0s = scipy.special.jn_zeros(0, 100.0)
    an, _ = scipy.integrate.newton_cotes(20, 1)

    args = {
        "omega": omega,
        "omega0": params3_jit["sample"].omega0,
        "kD": params3_jit["sample"].kD,
        "es": params3_jit["sample"].epsilon_s,
        "sj": loc1_nm,
        "rk": loc2_nm,
        "j0s": j0s,
        "an": an,
        "verbose": False,
        "breakpoints": 15,
    }

    omega0 = params3_jit["sample"].omega0
    kD = params3_jit["sample"].kD
    es = params3_jit["sample"].epsilon_s
    sj = torch.as_tensor(loc1_nm.astype(np.float32), device=device)
    rk = torch.as_tensor(loc2_nm.astype(np.float32), device=device)
    pts = 3000

    # Compare results computed using the new GPU code and old CPU code
    # 
    # 1. the Coulomb matrix


    Ctorch = Cmatrix(sj, rk)
    C0jit = Cmatrix_jit(loc1_nm, loc2_nm)

    print("C matrix test pass?", np.allclose(C0jit, Ctorch.cpu()))

    # 2. The (K0, K1, K2) matrices

    K0torch, K1torch, K2torch = KmatrixIII(omega, omega0, kD, es, sj, rk, pts, device)
    K0jit, K1jit, K2jit = KmatrixIII_jit(**args)

    print(
        "Kn matrices test pass?",
        np.allclose(K0jit, K0torch.cpu()),
        np.allclose(K1jit, K1torch.cpu()),
        np.allclose(K2jit, K2torch.cpu()),
    )

    # Compute (L0, L1, L2) 
    # 
    # 1. using the new GPU code and

    obj1 = twodimCobject(sample3, device)
    obj1.addsphere(ureg.Quantity(100,'nm'), 21, 21)
    obj1.set_alpha(1.0e-6)
    obj1.set_integration_points(21 * 15)
    obj1.solve(omega=1.0e5)
    print("Vrms = {:0.1e} V".format(1e-6 * obj1.results['Vrms [ppm]']))
    _ = obj1.plot()

    plt.show(block=False)

    # 2. using the old CPU code,

    obj2 = twodimCobjectPrior(sample3_jit)
    obj2.addsphere(ureg.Quantity(100,'nm'), 21, 21)
    obj2.set_alpha(1.0e-6)
    obj2.set_breakpoints(15)
    obj2.solve(omega=1.0e5)
    print("Vrms = {:0.1e} V".format(1e-6 * obj2.results['Vrms [ppm]']))
    _ = obj2.plot()

    plt.show(block=False)

    # and show that the (L0, L1, L2) values agree 
    # to within 0.1 percent, or a part per thousand.

    print(
        "(L0, L1, L2) test pass?",
        np.allclose(
           np.array(obj1.solve(omega=1.0e5)),
           np.array(obj2.solve(omega=1.0e5)),
           rtol=1e-3)
    )

    print("Please close the plot windows to exit the program.")
    plt.show()
    