import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dissipationtheory.constants import ureg
from lmfit import Model, Parameters
from dissipationtheory.capacitance import C2SphereConefit
from dissipationtheory.capacitance import FrictionCoefficientThickfit

def pdf_to_dict_with_units(pdf):
    """This utility function can be applied to a pandas dataframe object whose keys contain strings like 'distance [nm]' which contain data descriptor, distance, and a unit, nm.  This utility function turns the dataframe into a dictionary.  The dictionary key in the above example would be 'distance' and the dictionary values would be ureg.Quantity(np.array(<distance data>),'nm'), that is, a numpy array of data with units (implemented using the `pint` package)."""

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
    """An object for reading in and manipulating Marohn-group 'ringdown' data, stored as a csv file.  It is assumed that the `csv` file has the following data headings.
    
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