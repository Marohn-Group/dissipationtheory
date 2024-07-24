import numpy as np
import matplotlib.pyplot as plt

from dissipationtheory.data import BLDSData
from dissipationtheory.constants import ureg
from dissipationtheory.dissipation import blds_perpendicular_jit
from dissipationtheory.dissipation2 import theta1norm_jit
from lmfit import Model, Parameters
import h5py
from freqdemod.demodulate import Signal

class BLDSData2(BLDSData):
    """An object for reading in and manipulating Marohn-group BLDS spectroscopy data, stored as a ``tsv`` file.
    It is assumed that the ``tsv`` file has the following data headings.

    * Modulation Frequency [Hz]
    * Mean Freq [Hz]
    * X-Channel [Hz]
    * Y-Channel [Hz]

    For each BLDS spectrum, find a best-fit tip-sample separation, conductivity, and charge density.
    """

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
        """For each BLDS spectrum, find an optimum tip-sample separation, conductivity, and charge density."""

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
    """Plot the fit results: calculated conductivity and best-fit separation, conductivity, and charge density versus light intensity."""

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

class BLDSdataRT(object):
    """An object to work up cantilever displacement versus time data for a BLDS 
    experiment using a software lock-in.
    """

    def __init__(self, filepath):
        """Load the `.h5` file.  Save the data keys, create an array of 
        modulation frequency (in Hz and rad/s), and determine the time step 
        and number of points in the control transient.
        """
        
        self.path = filepath
        self.f = h5py.File(filepath, mode='r')
        self.keys = [key for key in self.f['data'].keys() if key != '0'] # does not include the control experiment
        self.f_m = np.array([float(key) for key in self.keys])           # does non include the control experiment
        self.omega_m = 2 * np.pi * self.f_m
        self.am = self.f['waveformParams'].attrs['amFreq']
        self.dt = 1/self.f['acq'].attrs['SampleRate']
        self.Nt = len(np.array(self.f['data']['0']))       
    
    def __repr__(self):
        """Print out the filename, time step, duration of each transient, and 
        report a summary of the modulation frequencies.  Assume the last modulation
        frequency is the built-in control at zero frequency.
        """
        
        str = "file = {:} \n".format(self.path.split('/')[-1])
        str = str + "time step = {:0.2f} us\n".format(1e6 * self.dt)
        str = str + "no. pts each = {:d}\n".format(self.Nt)
        str = str + "duration each = {:0.2f} ms\n".format(1e3 * self.dt * self.Nt)
        str = str + "am modulation freq = {:0.2f} Hz\n".format(self.am)
        str = str + "modulation frequencies\n"
        str = str + "  {:d} pts plus 1 pt at 0 Hz\n".format(len(self.f_m)) 
        str = str + "  {:0.2e} to {:0.2e} Hz\n".format(min(self.f_m), max(self.f_m)) 
        str = str + "  {:0.2e} to {:0.2e} rad/s\n".format(min(self.omega_m), max(self.omega_m))
        return str

    def _convert(self, signal_adc):
        """Convert an analog-to-digital converter signal to nanometers. To convert the adc signal to volts, 
        I am following the directions from Chris Petroff in the file 
        `~/Dropbox/EFM_Data_workup/2024-06-05--cap339--software-lockin--PM6-Y6/README`.
        """

        c1 = self.f['acq'].attrs['SampleOffset']
        c2 = self.f['acq'].attrs['SampleResolution']
        c3 = self.f['photodetectorChan'].attrs['InputRange'] / 2000
        c4 = self.f['photodetectorChan'].attrs['DcOffset']

        signal_volts = (c1 - signal_adc) /  c2 * c3 + c4

        c5 = self.f['bldsMetadata'].attrs['Interferometer nm-to-volt [nm/V]']
        
        signal_nm = signal_volts * c5
    
        return signal_nm
    
    def _extract_freq_vs_time(self, key):
        """Use the `freqdemod` package to extract the cantilever frequency versus time.
        The time axis is stored in self.T and the frequency axis is stored in self.S.  
        If you want ot use different demodulation parameters, rewrite this function.
        """

        # Convert the adc signal from unitless adc values to nanometers
        
        signal_adc = np.array(self.f['data'][key])
        signal_nm = self._convert(signal_adc)

        # Use the freqdemod package
        
        self.s = Signal()
        self.s.load_nparray(signal_nm, "x", "nm", self.dt)

        self.s.time_mask_binarate("middle")  # Pull out the middle section
        self.s.time_window_cyclicize(0.1E-3) # Force the data to start and end at zero
        self.s.fft()                         # Fourier transform the data
        self.s.freq_filter_Hilbert_complex() # Take the complex Hilbert transform
        self.s.freq_filter_bp(5.00)          # Apply a 5 kHz wide bandpass filter
        self.s.time_mask_rippleless(1.5E-3)  # Set up a filter to remove ripple
        self.s.ifft()                        # Inverse Fourier transform the data
        self.s.fit_phase(50E-6)              # Fit the phase vs time data

        self.T = np.array(self.s.f['workup/fit/x'])        
        self.S = np.array(self.s.f['workup/fit/y'])

    def _FT_freq_vs_time(self):
        """Compute the Fourier transform of the delta frequency vs. time signal, 
        i.e. self.S - self.S.mean() vs self.T.  The Fourier transform has units of Hz.
        """

        dT = self.T[1] - self.T[0]
        dS = self.S - self.S.mean()
        self.F = np.fft.fftshift(np.fft.fftfreq(len(dS), dT))
        self.dS_FT = (2 / len(dS)) * np.fft.fftshift(np.fft.fft(dS))

    def _freq_vs_time_lock_in(self, target):
        """The complex Fourier component nearest the target frequency."""

        idx = np.abs(B.F - target).argmin() 
        return (self.F[idx], self.dS_FT[idx])

    def spectra(self):
        """Saves two spectra (1) self.favg, the difference between the average cantilever frequency with the oscillating 
        voltage present and a dc voltage present and (2) self.BLDS, the complex Fourier component of the cantilever 
        nearest the am modulation frequency.  The frequency axis, in rad/s, is stored in self.omega_m."""

        # First, the control experiment

        self._extract_freq_vs_time('0')
        self.fc = self.S.mean()

        # Next, loop over the datasets

        self.favg = np.zeros_like(self.omega_m)
        self.BLDS = np.zeros_like(self.omega_m)

        for index, key in enumerate(self.keys):
            
            self._extract_freq_vs_time(key)
            self.favg[index] = self.S.mean() - self.fc
            
            self._FT_freq_vs_time()
            _, self.BLDS[index] = self._freq_vs_time_lock_in(self.am)