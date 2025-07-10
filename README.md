<!--
pandoc README.md -o README.html --css pandoc.css -s --mathjax --metadata title="README for dissipationtheory package" && open README.html 
-->

# dissipationtheory

Compute atomic force microscope cantilever dissipation and frequency noise over metals and dielectrics.  Compute cantilever capacitance using a sphere plus a cone model.

## Theory

This package implements models of tip-sample capacitance, friction, frequency noise, and frequency shift published in the following manuscripts.

1. [**Cherniavskaya2003feb**] Cherniavskaya, O.; Chen, L.; Weng, V.; Yuditsky, L.; Brus, L. E. Quantitative Noncontact Electrostatic Force Imaging of Nanocrystal Polarizability. *J. Phys. Chem. B* **2003**, *107 (7)*, 1525. https://doi.org/10.1021/jp0265438. Figure 4 is helpful.  Equation 19 (cone-plane capacitance second derivative).

2. [**Kuehn2006apr**] Kuehn, S.; Loring, R. F.; Marohn, J. A. Dielectric Fluctuations and the Origins of Noncontact Friction. *Phys. Rev. Lett.* **2006**, *96 (15)*, 156103. https://doi.org/10.1103/PhysRevLett.96.156103.

3. [**Yazdanian2008jun**] Yazdanian, S. M.; Marohn, J. A.; Loring, R. F. Dielectric Fluctuations in Force Microscopy: {N}oncontact Friction and Frequency Jitter. *J. Chem. Phys.* **2008**, *128 (22)*, 224706. https://doi.org/10.1063/1.2932254.

4. [**Yazdanian2009jun**] Yazdanian, S. M.; Hoepker, N.; Kuehn, S.; Loring, R. F.; Marohn, J. A. Quantifying Electric Field Gradient Fluctuations over Polymers Using Ultrasensitive Cantilevers. *Nano Lett.* **2009**, *9 (6)*, 2273. https://doi.org/10.1021/nl9004332.  Supplement to: Quantifying Electric Field Gradient Fluctuations over Polymers Using Ultrasensitive Cantilevers. *Nano Lett.* **2009**, *9 (10)*, 3668. https://doi.org/10.1021/nl901850b.

5. [**Hoepker2011dec**] Hoepker, N.; Lekkala, S.; Loring, R. F.; Marohn, J. A. Dielectric Fluctuations over Polymer Films Detected Using an Atomic Force Microscope. *J. Phys. Chem. B* **2011**, *115 (49)*, 14493. https://doi.org/10.1021/jp207387d.  Equations 4 (frequency shift) and 22 (cone-plane capacitance).  The paper directs you to the supplement for the cone-plane capacitance, but the information is not there (sorry).

6. [**Hoepker2013jan**] Hoepker, N.  Fluctuations near Thin Films of Polymers, Organic Photovoltaics, and Organic Semiconductors Probed by Electric Force Microscopy.  PhD thesis, Cornell University **2013**. http://hdl.handle.net/1813/33910.  Equations 2.49 (cone-plane capacitance) and equations 2.51 and 2.52 (sphere-plane capacitance).

7. [**Lekkala2012sep**] Lekkala, S.; Hoepker, N.; Marohn, J. A.; Loring, R. F. Charge Carrier Dynamics and Interactions in Electric Force Microscopy. *J. Chem. Phys.* **2012**, *137 (12)*, 124701. https://doi.org/10.1063/1.4754602.

8. [**Lekkala2013nov**] Lekkala, S.; Marohn, J. A.; Loring, R. F. Electric Force Microscopy of Semiconductors: Theory of Cantilever Frequency Fluctuations and Noncontact Friction. *J. Chem. Phys.* **2013**, *139 (18)*, 184702. https://doi.org/10.1063/1.4828862.

We plan to incorporate equations from the following two papers in the near future.

9. [**Loring2022sep**] Loring, R. F. Noncontact Friction in Electric Force Microscopy over a Conductor with Nonlocal Dielectric Response. *J. Phys. Chem. A* **2022**, *126 (36)*, 6309. https://doi.org/10.1021/acs.jpca.2c04428.

10. [**Loring2023jul**] Loring, R. F. Voltage Fluctuations and Probe Frequency Jitter in Electric Force Microscopy of a Conductor. *J. Chem. Phys.* **2023**, *159 (4)*, 044703. https://doi.org/10.1063/5.0160556.


## Installation

## Install the development version

### Via poetry

I am using the poetry tool ([link](https://python-poetry.org/)) for dependency management and packaging.  So install the poetry tool.  I usually run the conda python distribution at the command line, deactivate conda before installing poetry.

```
$ conda deactivate
$ curl -sSL https://install.python-poetry.org | python3 -
$ open ${HOME}/.bash_profile
```

and in the `.bash_profile` file add

```
export PATH="/Users/jam99/.local/bin:$PATH"
```

As follows, run the `.bash_profile` file to update the path.  This will reactivate conda, so deactivate it again.  We should now be able to run poetry at the command line.

```
$ source ${HOME}/.bash_profile
$ conda deactivate
$ poetry --version
Poetry (version 1.4.2)
```

There is as yet no python available at the command line.  Next create a `pyproject.toml` file that will specify the virtual environment (i.e., tells poetry what version of python to run and what packages to load).  To create the `pyproject.toml` file I ran `poetry init` and answered the questions.  Afterwards, invoke the virtual environment as follows.  Verify that python is now running at the command line.

```
$ poetry shell
$ python --version
Python 3.8.5
```

While python is installed, none of the packages are:

```
$ python -c "import numpy as np"
ModuleNotFoundError: No module named 'numpy'
```

To install the packages run

```
$ poetry install --all-extras
```

and be prepared to wait while the many dependent packages are installed.  We now have a `poetry.lock` file present.  To run the unit tests,

```
$ poetry run pytest
```

After updating the package dependencies in `pyproject.toml`, I'll run

```
$ poetry lock
$ poetry install
```

Rememember to deactivate any conda environments before you start, to avoid errors resulting from package collisions.  If you are coming back to work on the package, the working commands are

```
$ conda deactivate
$ poetry lock
$ poetry install --all-extras
$ poetry run pytest
```

Note: I upgraded the python version in the `.toml` file from 3.8 to 3.9 but then got an error trying to launch `poetry shell`.  First I have to run

```
$ poetry env use 3.9
```

which sets up a new virtual environment.  Now I can proceed with adding a package that requires python 3.9, like sphinx

```
$ poetry add sphinx
$ poetry lock
$ poetry install --all-extras
$ poetry run pytest
```


### Adding and running jupyter

To install,

```
$ poetry add -D jupyter
$ jupyter notebook
```

Because I want to run the notebook in MS Visual Studio Code, I'll need to create a named Jupyter kernel based on this poetry environment.

```
$ poetry run ipython kernel install --user --name=dissipationtheory
Installed kernelspec dissipationtheory in /Users/jam99/Library/Jupyter/kernels/dissipationtheory
```

### Via conda

Alternatively

```
$ conda create -n dissipationtheory python=3.8
$ pip install .[test]
$ python -m pytest
```

### Total lines of code 

The lines of code in the repository can be viewed by running

```
git ls-files *.py | xargs wc -l
```

### Upgrading poetry

To implement the bessel function `scipy.special.j0` in a jit-compiled function, the numba-scipy package is required.  This could in principle be added using `poetry add numba-scipy` bit this adding procedure could not resolve the environment (after 20 minutes).

In principle, after exiting poetry and conda, I should be able to run `poetry self update` to update poetry.  However, I got the error "The current project's Python requirement (3.8.5) is not compatible with some of the required packages Python requirement".  Poetry requires a Python version <4.0 and >=3.9.
So let me create a conda environment with Python 3.10 as follows.

```
$ conda deactivate
$ conda create -n py310 python=3.10 
```

This installation takes a while.  Now activate the new Python environment and try to update poetry

```
$ conda activate py310
$ poetry self update
```

This updating fails with the error "The current project's Python requirement (3.8.5) is not compatible ...".  Exercise the "nuclear option" by re-installing poetry as follows.

```
$ curl -sSL https://install.python-poetry.org | python3 -
$ poetry --version
Poetry (version 2.1.3)
$ which poetry
/Users/jam99/.local/bin/poetry
$ which python
/Users/jam99/opt/anaconda3/envs/py310/bin/python
```

Poetry no longer has a `shell` command. Ugh. Install all packages, update the lock, and activate the resulting environment following directions [here](https://python-poetry.org/docs/managing-environments/).  I have listed `scipy` as a package in the `.toml` file, so if Python can import `scipy` then I have entered the virtual environment ok.

```
$ poetry install --all-extras
$ poetry lock
$ $ poetry env activate
source /Users/jam99/Library/Caches/pypoetry/virtualenvs/dissipationtheory-Uvi85QQO-py3.10/bin/activate
$ eval $(poetry env activate)
$ python -c "import scipy"
```

Now try adding the `numba-scipy` package.  Frustratingly, this addition fails.  It is not clear from the convoluted error message what the problem is.  Remove the `scipy` line from the `pyproject.toml` file, run `poetry install --all-extras` and `poetry lock`.  Surprisingly, the `scipy` package is still installed.

```$ poetry show | grep "scipy"
scipy 1.15.3
```

Running the command 

```
poetry show --tree | more
```

is helpful.  I can see that the `scipy` package is required by `freqdemod` and `lmfit`.  I hypothesize that this is why poetry did not uninstall `scipy` when asked.  Let me try adding `numba-scipy` by more carefully specifying the Python version.  In the `.toml` file, specify the version of Python as follows:

```
python = "3.10.2"
```

Now run `poetry install --all-extras`, `poetry lock`, and try

```
$ poetry add numba-scipy
```

Happily, the environment resolves in 1.1s! The `scipy` package is downgraded from version 1.15.3 to 1.10.1 and the `numba-scipy` package 0.4.0 is installed.  I get the message "Writing lock file" and I can see that a line

```
numba-scipy = "^0.4.0"
```

has been added to the `pyproject.toml` file.  Finally, update Jupyter

```
$ poetry run ipython kernel install --user --name=dissipationtheory
```

The new procedure for booting poetry is 

```
$ conda deactivate
$ eval $(poetry env activate)
```