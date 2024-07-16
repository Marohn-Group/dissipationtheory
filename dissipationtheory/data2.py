import numpy as np
import matplotlib.pyplot as plt

from dissipationtheory.data import BLDSData
from dissipationtheory.constants import ureg
from dissipationtheory.dissipation import blds_perpendicular_jit
from dissipationtheory.dissipation2 import theta1norm_jit
from lmfit import Model, Parameters d

class BLDSData2(BLDSData):

    def fitfunc(self, x, separation, conductivity, density):
        """A function used internally to fit the BLDS spectrum to theory.
        
        :param np.array x: modulation-frequency data
        :param float separation: tip-sample separation [nm]
        :param float conductivity: sample conductivity [:math:`10^{-8} \\mathrmm{S}/\\mathrm{m}`]
        :param float density: sample charge density [:math:`10^{21} \\mathrm{m}^{-3}`]

        The conductivity and charge density are in units of :math:`10^{-8} \\mathrmm{S}/\\mathrm{m}` and
        :math:`10^{21} \\mathrm{m}^{-3}`, respectively.  This is so that the parameters passed to the curve-fitting
        function are likely to be between about 0.001 and 1000. 
        """

        self.sample_jit.cantilever.d = separation * 1e-9
        self.sample_jit.sigma = conductivity * 1e-8
        self.sample_jit.rho = density * 1e21
        
        omega_m = ureg.Quantity(x, 'Hz')
        blds = ureg.Quantity(np.zeros_like(x), 'Hz')
        for index, omega_ in enumerate(omega_m):
                blds[index] = blds_perpendicular_jit(
                    theta1norm_jit,
                    self.sample_jit,
                    omega_).to('Hz')
        
        return abs(blds.to('Hz').magnitude)

    def fitguess(self, separation, conductivity, density):
        """Create an initial guess for curve fitting.
        The same initial guess is used at each light intensity.
        
        :param float separation: tip-sample separation [nm]
        :param float conductivity: sample conductivity [:math:`10^{-8} \\mathrmm{S}/\\mathrm{m}`]
        :param float density: sample charge density [:math:`10^{21} \\mathrm{m}^{-3}`]
        """

        self.guess = True
        self.separation = separation
        self.sigma = conductivity
        self.rho = density

        for key in self.database.keys():
            self.database[key]['y_calc'] = self.fitfunc(self.database[key]['x'], separation, conductivity, density)
            
    def fit(self):
        """For each BLDS spectrum, find an optimimum tip-sample separation, conductivity, and charge density."""

        # Set up the fit
        
        self.fitted=True
        gmodel = Model(self.fitfunc)

        pars= Parameters() 
        pars.add('separation', value=self.separation, min=5, max=5000)
        pars.add('conductivity', value=self.sigma, min=1e-4, max=1e4)
        pars.add('density', value=self.rho, min=1e-4, max=1e4)

        # Loop over all keys and fit the blds data
        
        for key in self.database.keys():

            print('fitting dataset {:}'.format(key))

            result = gmodel.fit(
                self.database[key][self.plotkey],
                x=self.database[key]['x'],
                params=pars)
            
            self.database[key]['result'] = result
            self.database[key]['y_calc'] = result.best_fit
            self.database[key]['values'] = {
                'separation': ureg.Quantity(result.params['separation'].value, 'nm'),
                'conductivity': ureg.Quantity(1e-8 * result.params['conductivity'].value, 'S/m'),
                'density': ureg.Quantity(1e21 * result.params['density'].value, '1/m^3')}            
            self.database[key]['stderr'] = {
                'separation': ureg.Quantity(result.params['separation'].stderr, 'nm'),
                'conductivity': ureg.Quantity(1e-8 * result.params['conductivity'].stderr, 'S/m'),
                'density': ureg.Quantity(1e21 * result.params['density'].stderr, '1/m^3')}
       
        # Create conductivity dictionary containing intensities and conductivity (values and error bars) 
        # with units. Compute a conductivity error bar by propagating error
        
        I_val = ureg.Quantity(np.zeros(len(self.database)), 'mW/cm^2')
        
        for index, key in enumerate(self.database.keys()):
        
            I_val[index] =  ureg.Quantity(self.database[key]['I [mW/cm^2]'], 'mW/cm^2')
            
            sigma_val = self.database[key]['values']['conductivity']
            sigma_err = self.database[key]['stderr']['conductivity']

            rho_val = self.database[key]['values']['density']        
            rho_err = self.database[key]['stderr']['density']

       # Create a dictionary of separation, conductivity, and density.
       # There is some rearranging to do, from a list of items with units, 
       # to a numpy array with one unit for the whole array.
  
        for finding in ['separation', 'conductivity', 'density']:
            
            unit = self.database[key]['values'][finding].units
            y = np.array([self.database[key]['values'][finding].to(unit).magnitude for key in self.database.keys()])
            yerr = np.array([self.database[key]['stderr'][finding].to(unit).magnitude for key in self.database.keys()])
            
            self.findings[finding] = {
                'x': I_val,
                'y': ureg.Quantity(y, unit),
                'yerr': ureg.Quantity(yerr, unit)
            }

def plotBLDSfindings(D):
    """Plot the fit results: calculated conductivity and best-fit separation, conductivity, and charge denstiy versus light intensity."""

    fig, axes = plt.subplots(figsize=(3*2.3, 2.50), ncols=3)
    opts = dict(marker='o', mfc='w', ms=4, capsize=3, linestyle='none')
    
    for index, finding in enumerate(['conductivity', 'density', 'separation']):
    
        if finding == 'conductivity':
            ylabel=r'$\sigma$ [$\mu$S/m]'
            yunit='uS/m'
            ydiv=1
        elif finding == 'separation':
            ylabel=r'$d$ [nm]'
            yunit='nm'
            ydiv=1
        elif finding == 'density':
            ylabel=r'$\rho$ [$10^{16}$ cm$^{-3}$]'
            yunit='1/cm^3'
            ydiv=1e16
            
        axes[index].errorbar(
            D.findings[finding]['x'].to('mW/cm^2').magnitude, 
            D.findings[finding]['y'].to(yunit).magnitude/ydiv,
            yerr=D.findings[finding]['yerr'].to(yunit).magnitude/ydiv,
            **opts)
    
        # axes[index].set_xscale('log')
        axes[index].set_ylabel(ylabel)
        axes[index].set_xlabel(r'$I_{h \nu}$ [mW/cm$^2$]')
        axes[index].grid()
    
    fig.tight_layout()
    
    return fig