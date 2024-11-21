<!---
pandoc table-of-contents.md -o table-of-contents.html --css pandoc.css -s --mathjax --metadata title="Table of Contents" && open table-of-contents.html
-->

- author: John A. Marohn
- created: 2024-10-29
- last updated: 2024-10-29

In the `dissipationtheory/development` folder there are a number of *studies* --- Jupyter notebooks in which I experimented with calculation and data-analysis code.  The studies have names like `dissipation-theory--Study-1.ipynb` and so forth.  Here I summarize these studies.  

# Studies

- **Study 1** ([html](dissipation-theory--Study-1.html)).  Explore Loring's recently revised theory for friction over metals.

- **Study 2** ([html](dissipation-theory--Study-2.html)).  Explore Lekkala and Loring's theory for friction over a semiconductor backed by a metal.

- **Study 3** ([html](dissipation-theory--Study-3.html)).  Try to reproduce Figure 7(b) and Figure 9(b) from Lekkala2012.

- **Study 4** ([html](dissipation-theory--Study-4.html)). Look at ways to cut down code redundancy in the code in Study 3 ([html](dissipation-theory--Study-3.html)). It would greatly simplify my code if the sample object would update the diffusion length automatically when the mobility or cantilever period was updated.

- **Study 5** ([html](dissipation-theory--Study-5.html)). Look at ways to cut down code redundancy in the code in Study 4 ([html](dissipation-theory--Study-4.html)).

- **Study 6** ([html](dissipation-theory--Study-6.html)).  Reproduce the friction $\gamma_{\parallel}$ versus charge density $\rho$ plots in Lekkala2013 Figure 7(b) using the functions in `dissipationtheory.dissipation.py`. The computation is carried out over 40 charge densities, for 4 charge mobilities, and using Model I and Model II in the paper. The computation in done two ways -- in pure Python and in `numba`-compiled Python. The computation takes approximately 2.4 min in pure Python and 0.6 s in `numba`-compiled Python.

- **Study 7** ([html](dissipation-theory--Study-7.html)).  A computation of friction was carried out over 40 charge densities and 5 values of the imaginary dielectric constant $\epsilon_{\mathrm{s}}^{\prime \prime}$, using a `numba`-compiled Python version of Model II in the paper. The computation takes less than a second.

  In the paper, the tip-sample separation $d$ was not specified. A first computation was done with $d = 300 \: \mathrm{\mu m}$, but the agreement with Lekkala was poor. The tip-sample separation was varied between 30 nm and 300 nm in 1 nm increments and the sum-squared deviation $\chi^2$ between the computed dissipation and Lekkala's dissipation calculated for the $\epsilon_{\mathrm{s}}^{\prime \prime} = 0$ case. The $\chi^2$ was minimized at an optimal tip-sample separation of $d_{\mathrm{opt}} = 101.0 \: \mathrm{nm}$. Computations performed with $d = d_{\mathrm{opt}}$ were in very good agreement with Lekkala over a range of $\epsilon_{\mathrm{s}}^{\prime \prime}$ values.

  Instead of minimizing $\chi^2$ using the $\epsilon_{\mathrm{s}}^{\prime\prime} = 0$ dataset, minimize it over all five data sets. Moreover, minimize the relative error, not the absolute error. This gives all the curves equal weight, even though they have quite different peak $\gamma_{\perp}$ values. The $\chi^2$ is now minimized at an optimal tip-sample separation of $d_{\mathrm{opt}} = 100.0 \: \mathrm{nm}$. Computations are are essentially unchanged.
  
- **Study 8** ([html](dissipation-theory--Study-8.html)).  Examine the unitless 0th and 1st derivatives of the sphere capacitance.

- **Study 9** ([html](dissipation-theory--Study-9.html)).  Reproduce the friction $\gamma_{\perp}$ versus charge density $\rho$ plots in Lekkala2013 Figure 9(b) using functions in `dissipationtheory.dissipation.py`. Add in a low-density approximation.

- **Study 10** ([html](dissipation-theory--Study-10.html)).  Plot the friction $\gamma_{\perp}$ versus height $h$ expected for a thin organic semiconductor sample backed by a metal.

- **Study 11** ([html](dissipation-theory--Study-11.html)).  Calculate dissipation versus height over a sample with properties similar to Rachael's PM6:Y6 and compare to Marohn's analytical expression derived in the thick-sample limit.

- **Study 12** ([html](dissipation-theory--Study-12.html)).  Calculate dissipation versus charge density for a representive perovskite sample.

- **Study 13** ([html](dissipation-theory--Study-13.html)).  Plot $a_{\mathrm{max}}$ contours in the $(\epsilon^{\prime}_{\mathrm{s}}, \epsilon^{\prime\prime}_{\mathrm{s}})$ plane, keeping in mind that $\epsilon^{\prime\prime}_{\mathrm{s}}$ is negative.

- **Study 14** ([html](dissipation-theory--Study-14.html)).  Calculate dissipation $\gamma_{\perp}$ versus charge density $\rho$ for a representive perovskite sample. Compare the exact result to the low-density exapansion and an analytical expression for $\gamma_{\perp}^{\mathrm{max}}$. Both these approximations are only valid in the infinite-sample limit, so use Model 2 with the dielectric overlayer thickness set to zero. Compare the infinite-sample Model 2 result to a finite-sample Model 1 result, to check the validity of the infinite-sample approximation used to obtain an analytical result for the height dependence of the $\gamma_{\perp}$.

- **Study 15** ([html](dissipation-theory--Study-15.html)).  Calculate dissipation versus height over a sample with properties similar to Rachael's PM6:Y6 and compare to Marohn's analytical expression derived in the thick-sample limit. This notebook recalculates the successful fit in Study 11 ([html](dissipation-theory--Study-11.html)) and further explores the result.

- **Study 16** ([html](dissipation-theory--Study-16.html)). Update the fits in Study 15 ([html](dissipation-theory--Study-15.html)) using a slightly different consensus sample mobility.

- **Study 17** ([html](dissipation-theory--Study-17.html)).  Test drive the new BLDS code in `dissipationtheory.dissipation`.

- **Study 18** ([html](dissipation-theory--Study-18.html)).  Test drive the new LDS code in `dissipationtheory.dissipation`.

- **Study 19** ([html](dissipation-theory--Study-19.html)).  Test drive the new `BLDSData` object in `dissipationtheory.data`.

- **Study 20** ([html](dissipation-theory--Study-20.html)).  Explore how the BLDS signal depends on mobility and charge density.  Plot the $B^{(2)}(\omega_{\mathrm{m}}=0)$ integral versus charge density $\rho$, for selected dielectric constants.  Generate a range of conductivities $\sigma_0$ and compute the associated roll-off frequency $\omega_0 = \sigma_0/\epsilon_0$.  Plot the BLDS spectrum for various values of the charge density $\rho$, with the mobility $\mu$ fixed. Plot the low-frequency limit of the BLDS frequency shift versus the charge density, with the mobility fixed.

- **Study 21** ([html](dissipation-theory--Study-21.html)).  I have created a new module, `dissipation2.py`, in which the semiconductor properties are input in terms of conductivity and charge density instead of mobility and charge density. Explore the new module.

- **Study 22** ([html](dissipation-theory--Study-22.html)).  I have created a new data-analysis module, `data2.py`, in which the BLDS spectrum is fit to extact charge conductivity and charge density. Use the module to analyze representative BLDS spectra.  Fit representative PM6:Y6 BLDS spectra.

- **Study 23** ([html](dissipation-theory--Study-23.html)).  Plot cantilever friction and BLDS signal versus conductivity and charge density.

- **Study 24** ([html](dissipation-theory--Study-24.html)).  Plot cantilever friction versus conductivity, similar to Lekkala 203 Figures 7(b) and 9(b), but for (1) perpendicular friction and (2) plotted versus conductivity and not charge density.

- **Study 25** ([html](dissipation-theory--Study-25.html)).  For a chosen set of cantilever and sample parameters, plot versus charge density $\rho$ the

  - low-frequency BLDS frequency shift $|\Delta f_{\mathrm{BLDS}}(0)|$ and
  - cantilever friction $\gamma_{\mathrm{\perp}}$.

  Plot these quantities versus, respectively, the unitless parameters
  
  - $(h/\lambda_{\mathrm{D}})^2$, with $h$ the tip-sample separation and $\lambda_{\mathrm{D}}$ the Debye length, and
  - $\omega_0/(\epsilon_{\mathrm{s}}^{\prime} \omega_{\mathrm{c}})$, 
  
  with $\omega_0 = \sigma/\epsilon_0$, $\sigma$ the conductivity, $\epsilon_{\mathrm{s}}^{\prime}$ the real part of the dielectric constant, and $\omega_{\mathrm{c}}$ the cantilever resonance frequency.
Compare Marohn's numerical calcualtions to Loring's low-density and high-density approximations for $|\Delta f_{\mathrm{BLDS}}(0)|$.

- **Study 26** ([html](dissipation-theory--Study-26.html)).  Loring's clear, revised equations are writtien in terms of a new unitless integral $K$, not a correlation function $C$ (Leakkala) or a response function $R$ (Marohn). The code in `dissipation.py` and `dissipation2.py` writes the friction and BLDS frequency shift in terms of a correlation function $C$ introduced in Leakkala's 2013 JCP paper. In `dissipation3.py` I rewrite the `dissipation.py` code in terms of Loring's unitless $K$ integral, elimating extranneous factors of $k_{\mathrm{b}} T$. In `dissipation3.py` I follow the convention of `dissipation2.py`, where the sample's dependent variables are conductivity $\sigma$ and charge denstiy $\rho$, not the convention in `dissipation.py`, where the sample's dependent variable are conductivity $\sigma$ and mobility $\mu$.

- **Study 27** ([html](dissipation-theory--Study-27.html)). The pure-Python objects `CantileverModel`, `SampleModel1`, and `SampleModel2` offer a better way to input simulation parameters, because you can input parameters with units, but the pure-Python computations are painfully show. Work out how to pass parameters from `CantileverModel` to `CantileverModelJit`, from `SampleModel1` to `SampleModel1Jit`, and from `SampleModel2` to `SampleModel2Jit`. We can now enter parameters using the pure-Python objects, then transfer the parameters to the `numba/jit` objects for fast computations.

- **Study 28** ([html](dissipation-theory--Study-28.html)).  Redo Study 26 ([html](dissipation-theory--Study-26.html)) using `numba/jit`-accelerated functions coded in `dissipation3.py`. The `numba/jit` functions are 100 to 200 times faster than their pure-Python counterparts run in Study 26 ([html](dissipation-theory--Study-26.html)), enabling us to plot here a more detailed BLDS spectrum and explore friction and the zero-frequency limit of the BLDS spectrum as a function of many more conductivity points.

- **Study 29** ([html](dissipation-theory--Study-29.html)).  I have rewritten the code in `dissipation4.py` to require the user to explicitly input the tip charge’s $z$ location. In this notebook I test drive the new code. Check that, for an “infinitely thick” sample, the BLDS frequency shift at $\omega_{\text{m}} = 0$ agrees with Loring’s $\rho \rightarrow 0$ and $\rho \rightarrow \infty$ limiting expressions.  The numerical calculation of the low-frequency limit of the BLDS spectrum agrees with the Loring's expansions at both low denstity and high density.  To reach agreement, it is important that the sample be very thick --- 100 times the tip-sample separation.

- **Study 30** ([html](dissipation-theory--Study-30.html)).  Compute the capacitance of a sphere over a metal plane numerically using the method outlined by Xu and co-workers. See Xu, J.; Li, J.; Li, W. Calculating Electrostatic Interactions in Atomic Force Microscopy with Semiconductor Samples. AIP Advances (2019) 9(10): 105308,  [doi:10.1063/1.5110482](https://doi.org/10.1063/1.5110482).