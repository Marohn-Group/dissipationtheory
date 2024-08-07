import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dissipationtheory.constants import ureg, qe
from lmfit import Model, Parameters
from dissipationtheory.capacitance import C2SphereConefit
from dissipationtheory.capacitance import FrictionCoefficientThickfit
from dissipationtheory.dissipation import blds_perpendicular_jit
from dissipationtheory.dissipation import theta1norm_jit
import os
import pandas as pd

def pdf_to_dict_with_units(pdf):
    """This utility function can be applied to a pandas dataframe (pdf) object whose keys contain strings like 'distance [nm]' which contain data descriptor, distance, and a unit, nm.  This utility function turns the dataframe into a dictionary.  The dictionary key in the above example would be 'distance' and the dictionary values would be ureg.Quantity(np.array(<distance data>),'nm'), that is, a numpy array of data with units (implemented using the `pint` package)."""

    datadict = {}
    for key in pdf.keys():

        loc1 = key.find('[')
        loc2 = key.find(']')

        if loc1 > 0: # skip over entries without a unit
            
            new_key = key[0:loc1].strip()
            new_unit = key[loc1:loc2].strip('[').strip(']')
            new_unit = new_unit.replace('pNs', 'pN s') # a kludge

            datadict[new_key] = ureg.Quantity(np.array(pdf[key]), new_unit) 
            
    return datadict

class DissipationData(object):
    """An object for reading in and manipulating Marohn-group 'ringdown' data, stored as a ``csv`` file.  It is assumed that the ``csv`` file has the following data headings.
    
    * Distance from Surface [nm]
    * Curvature [Hz/V^2]
    * Curvature std [Hz/V^2]
    * Surface Potential [V]
    * Surface Potential std [V]
    * Dissipation Amplitude [pNs/(V^2 m)]
    * Dissipation Amplitude std [pNs/(V^2 m)]
    * Dissipation Ringdown [pNs/(V^2 m)]
    * Dissipation Ringdown std [pNs/(V^2 m)]
    * C2 [F/m^2],Electric Field Fluctuation [V^2/m^2 Hz]
    * Ringdown Electric Field Fluctuation [V^2/m^2 Hz]

    """

    def __init__(self, datadict, key, filename):
        """Initialize the object by providing an initial data dictionary, datadict (which can be a empty dictionary {}); a key string describing the dataset; and the filename of the csv file to read data from.  Example:

            gold = DissipationData({}, '0.00 V', '20230929-ringdown-gold_data_summary.csv')
        
        """

        self.datadict = datadict
        self.datadict[key] = pdf_to_dict_with_units(pd.read_csv(filename))

    def add_cantilever(self, cantilever):
        """Add cantilever information to the data dictionary.  Example: 
        
            gold.add_cantilever(
                {'f_0': ureg.Quantity(60457,'Hz'), 
                'k_0': ureg.Quantity(2.8,'N/m')})
        
        """
        self.cantilever = cantilever

    def C2_compute(self):
        """Use the cantilever spring constant and resonance frequency to convert the frequency parabola curvature, in Hz/V^2, to capacitance second derivative, in units of mF/m^2.  """


        prefactor = (4 * self.cantilever['k_0'] / self.cantilever['f_0'])
        
        for key in self.datadict.keys():
    
            self.datadict[key]['C2'] = (prefactor * self.datadict[key]['Curvature']).to('mF/m^2')
            self.datadict[key]['C2 std'] = (prefactor * self.datadict[key]['Curvature std']).to('mF/m^2')

    def C2_fit(self):
        """Fit the capacitance second derivative to the model of a sphere plus a cone above a ground plane."""

        gmodel = Model(C2SphereConefit)
        
        pars = Parameters()
        pars.add('radius', value=55.0, min=0.1, max=100.)
        pars.add('theta', value=15.0, min=1.0, max=45.)

        self.result = {}
        for key in self.datadict.keys():
    
            self.result[key] = gmodel.fit(
                self.datadict[key]['C2'].to('mF/m^2').magnitude, 
                params=pars, 
                height=self.datadict[key]['Distance from Surface'].to('nm').magnitude,
                weights=1/(self.datadict[key]['C2 std'].to('mF/m^2').magnitude))

    def C2_fit_plot(self, xscale='log', yscale='log', name=None):
        """Plot the capacitance second-derivative fit."""

        fig, axes = plt.subplots(
            figsize=(3.25,4.00),
            nrows=2,
            sharex=True, 
            gridspec_kw={'height_ratios': [1, 3]})

        for key in self.datadict.keys():
        
            x1 = self.datadict[key]['Distance from Surface'].to('nm').magnitude
            y1 = self.datadict[key]['C2'].to('mF/m^2').magnitude
            dy1 = self.datadict[key]['C2 std'].to('mF/m^2').magnitude
    
            r_avg = self.result[key].result.params['radius'].value
            r_err = self.result[key].result.params['radius'].stderr
            theta_avg = self.result[key].result.params['theta'].value
            theta_err = self.result[key].result.params['theta'].stderr
            
            y2 = C2SphereConefit(x1, r_avg, theta_avg)

            opts = dict(marker='o', mfc='w', ms=4, capsize=3, linestyle='none')

            axes[1].set_xscale(xscale)
            axes[1].set_yscale(yscale)
            axes[1].errorbar(x1, y1, yerr=dy1, **opts, label=key)
            axes[1].plot(x1, y2, '-')
            
            axes[0].set_yscale('linear')
            axes[0].errorbar(x1, (y2 - y1)/dy1, yerr=dy1, **opts, label=key)
            
        axes[0].set_ylabel('norm. resid.')
        axes[1].set_xlabel('tip-sample separation $h$ [nm]')
        axes[1].set_ylabel('$C_2$ [mF/m$^2$]')
        
        text1 = r"sphere $r = {:0.2f} \pm {:0.2f}$".format(r_avg, r_err)
        text2 = r"cone $\theta = {:0.2f} \pm {:0.2f}$".format(theta_avg, theta_err)
        text = text1 + '\n' + text2
        
        axes[1].text(0.95, 0.95, text, fontsize=10, 
            transform=axes[1].transAxes,
            horizontalalignment='right',
            verticalalignment='top')
        
        fig.align_ylabels()
        plt.tight_layout()

        if name is not None:

            fig.savefig(name + '.pdf')
            fig.savefig(name + '.png', dpi=300)
        
        plt.show()
                
    def dissipation_thickfit(self):
        """Fit the dissipation data to Marohn's formula for a semi-infinite, high dielectric constant sample."""

        gmodel = Model(FrictionCoefficientThickfit)

        pars = Parameters()
        pars.add('a', value=0.1, min=0, max=100)
        pars.add('R', value=self.cantilever['R'].to('nm').magnitude, vary=False)
        pars.add('fc', value=self.cantilever['f_0'].to('Hz').magnitude, vary=False)

        self.result = {}
        for key in self.datadict.keys():
    
            self.result[key] = gmodel.fit(
                self.datadict[key]['Dissipation Amplitude'].to('(pN s)/(V^2 m)').magnitude, 
                params=pars, 
                height=self.datadict[key]['Distance from Surface'].to('nm').magnitude,
                weights=1/(self.datadict[key]['Dissipation Amplitude std'].to('(pN s)/(V^2 m)').magnitude))
 
    def dissipation_thickfit_plot(self, xscale='log', yscale='linear', name=None):
        """Plot the result of `dissipation_thickfit`."""

        fig, axes = plt.subplots(
            figsize=(3.25,4.00),
            nrows=2,
            sharex=True, 
            gridspec_kw={'height_ratios': [1, 3]})

        for key in self.datadict.keys():
        
            x1 = self.datadict[key]['Distance from Surface'].to('nm').magnitude
            y1 = self.datadict[key]['Dissipation Amplitude'].to('(pN s)/(V^2 m)').magnitude
            dy1 = self.datadict[key]['Dissipation Amplitude std'].to('(pN s)/(V^2 m)').magnitude
    
            a_avg = self.result[key].result.params['a'].value
            a_err = self.result[key].result.params['a'].stderr
            
            y2 = FrictionCoefficientThickfit(
                height=x1, 
                a=a_avg, 
                R=self.cantilever['R'].to('nm').magnitude, 
                fc=self.cantilever['f_0'].to('Hz').magnitude)

            opts = dict(marker='o', mfc='w', ms=4, capsize=3, linestyle='none')

            axes[1].set_xscale(xscale)
            axes[1].set_yscale(yscale)
            axes[1].errorbar(x1, y1, yerr=dy1, **opts, label=key)
            axes[1].plot(x1, y2, '-')
            
            axes[0].set_yscale('linear')
            axes[0].errorbar(x1, (y2 - y1)/dy1, yerr=dy1, **opts, label=key)
            
        axes[0].set_ylabel('norm. resid.')
        axes[1].set_xlabel('tip-sample separation $h$ [nm]')
        axes[1].set_ylabel('$\gamma^V_{\perp}$ [pN s/(V$^2 m)]')
        
        text = r"$a = {:0.6f} \pm {:0.6f}$".format(a_avg, a_err)
        
        axes[1].text(0.95, 0.95, text, fontsize=10, 
            transform=axes[1].transAxes,
            horizontalalignment='right',
            verticalalignment='top')
        
        fig.align_ylabels()
        plt.tight_layout()

        if name is not None:

            fig.savefig(name + '.pdf')
            fig.savefig(name + '.png', dpi=300)
        
        plt.show()       


    def dissipation_plot(self, xscale='log', yscale='linear', name=None):
        """Plot the dissipation versus height."""

        fig, axs = plt.subplots(1, 2, figsize=(6.50, 3.25), sharey=True, sharex=True)

        if xscale == 'log':
            axs[0].set_xscale('log')
        
        if yscale == 'log':
            axs[0].set_yscale('log')

        opts = dict(marker='o', mfc='w', ms=4, capsize=3, linestyle='none')

        for key in self.datadict.keys():
        
            x = self.datadict[key]['Distance from Surface'].to('nm').magnitude
            y1 = self.datadict[key]['Dissipation Amplitude'].to('pN s/(V^2 m)').magnitude
            y1err = self.datadict[key]['Dissipation Amplitude std'].to('pN s/(V^2 m)').magnitude
            
            y2 = self.datadict[key]['Dissipation Ringdown'].to('pN s/(V^2 m)').magnitude
            y2err = self.datadict[key]['Dissipation Ringdown std'].to('pN s/(V^2 m)').magnitude
        
            axs[0].errorbar(x, y1, yerr=y1err, **opts, label=key)
            axs[1].errorbar(x, y2, yerr=y2err, **opts, label=key)
        
        axs[0].set_ylabel('friction coefficient $\gamma$ [pNs/(V$^2$ m)]')
        axs[0].set_xlabel('tip-sample separation $h$ [nm]')
        axs[1].set_xlabel('tip-sample separation $h$ [nm]')
        axs[0].legend()
        axs[1].legend()

        if name is not None:

            fig.savefig(name + '.pdf')
            fig.savefig(name + '.png', dpi=300)
        
        plt.tight_layout()

class BLDSData(object):
    """An object for reading in and manipulating Marohn-group BLDS spectroscopy data, stored as a ``tsv`` file.
    It is assumed that the ``tsv`` file has the following data headings.

    * Modulation Frequency [Hz]
    * Mean Freq [Hz]
    * X-Channel [Hz]
    * Y-Channel [Hz]

    For each BLDS spectrum, find a best-fit tip-sample separation, mobility, and charge density.
    """

    def __init__(self, THIS, filepath, database, sample_jit):
        """Initialize the data structure.

        :param string THIS: the name of the current jupyter notebook (to be prepended to a figure filename)
        :param list filepath: a list of strings describing the file path; see the example below 
        :param dictionary database: a dictionary of dictionaries specifying the filenames and light intensities; see the example below
        :param SampleModel1Jit sample_jit: 

        Example filepath ::

            ['~','Dropbox','EFM_Data_workup','pm6-y6-paper-blds-data','pm6-y6','ito','pm6-y6-ito-2']
    
        Example database ::

            database = {}
            database['A'] = {'filename': '230531-085452-BLDS-pm6-y6-3-dark.tsv', 'I [mW/cm^2]' : 0}
            database['B'] = {'filename': '230531-085907-BLDS-pm6-y6-3-50mA.tsv', 'I [mW/cm^2]' : 0.84}
        
        Data from all the files are read into an internally-stored database at initialization time.  Example sample_jit ::


            sample_jit = SampleModel1Jit(
                cantilever = CantileverModelJit(
                    f_c = 75e3, 
                    V_ts = 1.0,
                    R = 30e-9,
                    d = 200e-9),
                h_s = 110e-9, 
                epsilon_s = complex(3.4, 0),
                mu = 1e-8,
                rho = 1e21,
                epsilon_d = complex(1e6, 0),
                z_r = 110e-9
            )

        During curve fitting, the tip-sample separation ``d``, charge mobility ``mu``, and charge density ``rho`` are varied.
        """

        self.THIS = THIS
        self.database = database
        self.sample_jit = sample_jit
        self.findings = {}
        self.plotkey = None
        self.guess = False
        self.fitted = False

        for key in database.keys():
    
            path = os.path.join(os.path.join(*filepath), database[key]['filename'])
            df = pd.read_csv(path, sep='\t')
        
            x = df['Modulation Frequency [Hz]'].to_numpy()
            y1 = df['Mean Freq [Hz]'].to_numpy()
            y2r = df['X-Channel [Hz]'].to_numpy()
            y2i = df['Y-Channel [Hz]'].to_numpy()
            y2m = np.sqrt(y2r**2 + y2i**2)
        
            self.database[key]['x'] = x
            self.database[key]['abs(y1)'] = abs(y1)
            self.database[key]['y2m'] = y2m

    def plotdata(self, plotkey='abs(y1)'):
        """Plot the BLDS spectra -- one plot per light intensity, arranged horizontally with a common y-axis.
        This function returns the plot, for further customization or saving. 
        If the data has been fit before, or if an initial guess was provided, the calculated BLDS spectrum will be plotted also.
         
        :param string plotkey: either 'abs(y1)' (default) or 'y2m'
        """

        self.plotkey = plotkey
        fig, axes = plt.subplots(
            figsize=(len(self.database) * 2.00 + 1.00, 2.50),
            ncols=len(self.database),
            sharey=True)

        for index, key in enumerate(self.database.keys()):

            x = self.database[key]['x']
            y = self.database[key][plotkey]
            
            lbl = r'$I_{h \nu}$ = ' + '{:0.1f} mW/cm$^2$'.format(self.database[key]['I [mW/cm^2]'])
            axes[index].set_title(lbl, fontsize=12)
            axes[index].semilogx(x, y, marker = 'o', fillstyle='none', linestyle='none')
            axes[index].set_xticks([1e3,1e5])
            axes[index].set_xlabel('$\omega_{\mathrm{m}}$ [rad/s]')
            axes[index].grid()
            
            if self.guess or self.fitted:
                axes[index].semilogx(self.database[key]['x'], self.database[key]['y_calc'], '-')
                
        axes[0].set_ylabel('|$\Delta f_{\mathrm{BLDS}}$| [Hz]')
        fig.tight_layout()
        
        return fig
        
    def fitfunc(self, x, separation, mobility, density):
        """A function used internally to fit the BLDS spectrum to theory.
        
        :param np.array x: modulation-frequency data
        :param float separation: tip-sample separation [nm]
        :param float mobility: sample mobility [:math:`10^{-8} \\mathrmm{m}^2 \\mathrm{V}^{-1} \\mathrm{s}^{-1}`]
        :param float density: sample charge density [:math:`10^{21} \\mathrm{m}^{-3}`]

        The mobility and charge density are in units of :math:`10^{-8} \\mathrmm{m}^2 \\mathrm{V}^{-1} \\mathrm{s}^{-1}` and
        :math:`10^{21} \\mathrm{m}^{-3}`, respectively.  This is so that the parameters passed to the curve-fitting
        function are likely to be between about 0.001 and 1000. 
        """

        self.sample_jit.cantilever.d = separation * 1e-9
        self.sample_jit.mu = mobility * 1e-8
        self.sample_jit.rho = density * 1e21
        
        omega_m = ureg.Quantity(x, 'Hz')
        blds = ureg.Quantity(np.zeros_like(x), 'Hz')
        for index, omega_ in enumerate(omega_m):
                blds[index] = blds_perpendicular_jit(theta1norm_jit, self.sample_jit, omega_).to('Hz')
        
        return abs(blds.to('Hz').magnitude)

    def fitguess(self, separation, mobility, density):
        """Create an initial guess for curve fitting.
        The same initial guess is used at each light intensity.
        
        :param float separation: tip-sample separation [nm]
        :param float mobility: sample mobility [:math:`10^{-8} \\mathrmm{m}^2 \\mathrm{V}^{-1} \\mathrm{s}^{-1}`]
        :param float density: sample charge density [:math:`10^{21} \\mathrm{m}^{-3}`]
        """

        self.guess = True
        self.separation = separation
        self.mobility = mobility
        self.mobility = density

        for key in self.database.keys():
            self.database[key]['y_calc'] = self.fitfunc(self.database[key]['x'], separation, mobility, density)

    def fit(self):
        """For each BLDS spectrum, find an optimum tip-sample separation, mobility, and charge density."""

        # Set up the fit
        
        self.fitted=True
        gmodel = Model(self.fitfunc)

        pars= Parameters() 
        pars.add('separation', value=self.separation, min=50, max=500)
        pars.add('mobility', value=self.mobility, min=0.01, max=100)
        pars.add('density', value=self.mobility, min=0.01, max=100)

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
                'mobility': ureg.Quantity(1e-8 * result.params['mobility'].value, 'm^2/(V s)'),
                'density': ureg.Quantity(1e21 * result.params['density'].value, '1/m^3')}            
            self.database[key]['stderr'] = {
                'separation': ureg.Quantity(result.params['separation'].stderr, 'nm'),
                'mobility': ureg.Quantity(1e-8 * result.params['mobility'].stderr, 'm^2/(V s)'),
                'density': ureg.Quantity(1e21 * result.params['density'].stderr, '1/m^3')}
       
        # Create conductivity dictionary containing intensities and conductivity (values and error bars) 
        # with units. Compute a conductivity error bar by propagating error
        
        I_val = ureg.Quantity(np.zeros(len(self.database)), 'mW/cm^2')
        cond_val = ureg.Quantity(np.zeros(len(self.database)), 'mS/m')
        cond_err = ureg.Quantity(np.zeros(len(self.database)), 'mS/m')
        
        for index, key in enumerate(self.database.keys()):
        
            I_val[index] =  ureg.Quantity(self.database[key]['I [mW/cm^2]'], 'mW/cm^2')
            
            mu_val = self.database[key]['values']['mobility']
            rho_val = self.database[key]['values']['density']
            
            cond_val[index] = qe * mu_val * rho_val
        
            mu_err = self.database[key]['stderr']['mobility']
            rho_err = self.database[key]['stderr']['density']
            
            cond_err[index] = cond_val[index] * np.sqrt((mu_err/mu_val)**2 + (rho_err/rho_val)**2)

        self.findings['conductivity'] = {'x': I_val, 'y': cond_val, 'yerr': cond_err}

       # Create a dictionary of separation, mobility, and density.
       # There is some rearranging to do, from a list of items with units, 
       # to a numpy array with one unit for the whole array.
  
        for finding in ['separation', 'mobility', 'density']:
            unit = self.database[key]['values'][finding].units
            y = np.array([self.database[key]['values'][finding].to(unit).magnitude for key in self.database.keys()])
            yerr = np.array([self.database[key]['stderr'][finding].to(unit).magnitude for key in self.database.keys()])
            
            self.findings[finding] = {
                'x': I_val,
                'y': ureg.Quantity(y, unit),
                'yerr': ureg.Quantity(yerr, unit)
            }

    def plotfindings(self):
        """Plot the fit results: calculated conductivity and best-fit separation, mobility, and charge denstiy versus light intensity."""

        fig, axes = plt.subplots(figsize=(10.0, 2.50), ncols=4)
        opts = dict(marker='o', mfc='w', ms=4, capsize=3, linestyle='none')
        
        for index, finding in enumerate(self.findings.keys()):
        
            if finding == 'conductivity':
                ylabel=r'$\sigma$ [$\mu$S/m]'
                yunit='uS/m'
                ydiv=1
            elif finding == 'separation':
                ylabel=r'$d$ [nm]'
                yunit='nm'
                ydiv=1
            elif finding == 'mobility':
                ylabel=r'$\mu$ [$10^{-5}$ cm$^2$/(V s)]'
                yunit='cm^2/(V s)'
                ydiv=1e-5
            elif finding == 'density':
                ylabel=r'$\rho$ [$10^{16}$ cm$^{-3}$]'
                yunit='1/cm^3'
                ydiv=1e16
                
            axes[index].errorbar(
                self.findings[finding]['x'].to('mW/cm^2').magnitude, 
                self.findings[finding]['y'].to(yunit).magnitude/ydiv,
                yerr=self.findings[finding]['yerr'].to(yunit).magnitude /ydiv,
                **opts)
        
            # axes[index].set_xscale('log')
            axes[index].set_ylabel(ylabel)
            axes[index].set_xlabel(r'$I_{h \nu}$ [mW/cm$^2$]')
            axes[index].grid()
        
        fig.tight_layout()
        
        return fig

    def printfindings(self):
        """Print out useful results, like the dark conductivity."""

        print("dark conductivity = {:5.1f} +/- {:5.1f} uS/m".format(
            self.findings['conductivity']['y'][0].to('uS/m').magnitude,
            self.findings['conductivity']['yerr'][0].to('uS/m').magnitude))