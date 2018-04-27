phc_mode_clustering [![DOI](https://zenodo.org/badge/128033169.svg)](https://zenodo.org/badge/latestdoi/128033169)
==============================

This is the Python code publication supplement to the manuscript entitled
  > “Machine learning classification for field distributions of photonic modes”

by
  > C. Barth¹ and C. Becker¹
  
affiliated with
  > ¹Helmholtz-Zentrum Berlin für Materialien und Energie, Albert-Einstein-Str. 15, 12489 Berlin, Germany.

The manuscript is published as a preprint on ArXiv:
  > ArXiv e-prints (2018), [arXiv:1803.08290](http://arxiv.org/abs/1803.08290) [physics.optics].

This code is accompanied by (and partly depends on) the data publication
  > Barth, Carlo; Becker, Christiane (2018): *Supplement to: Machine learning classification for field distributions of photonic modes.* HZB Data Service. http://doi.org/10.5442/ND000002


**Notes**
  - A script for automatic download and verification of the data is included in this module. See the usage notes below.
  - The data publication DOI will only be resolvable after the manuscript has been officially published. Until then a fallback URL will be used internally.

## Installation

This code can only be cloned as is, either directly from Github or via the version archived by Zenodo (you can find the DOI for the archived version in the final publication).

See [requirements.txt](requirements.txt) for a list of Python modules on which it depends. This is basically the PyData stack, so e.g. users of anaconda may need minimal installations to get started.

## Usage

### Getting the data

  > **Caution, the size of the data is >100GB. Make that you have enough disk space and bandwidth/time.**

The [src](src) module comes with a command-line [script to download](src/data/make_dataset.py) and verify the complete data. You can see the syntax like this:

    >> python src/data/make_dataset.py --help
    Usage: make_dataset.py [OPTIONS]
    
      Downloads and verifies the raw data needed for the processing scripts and
      saves it to ../raw.
    
    Options:
      --full_checksum    Use full SHA256 checksums instead of reduced ones.
      --print_checksums  Print-out the actual checksums instead of checking.
      --help             Show this message and exit.

That is, simply calling the script like

    >> python src/data/make_dataset.py
    
will download the data and verify it using reduced checksums. Use the `full_checksum` flag if you prefer a complete verification, but note that it will take a while due t the large file sizes. If you have the [tqdm](https://github.com/tqdm/tqdm) module installed, the download will also display a progress bar and the estimated remaining time.

The data will be downloaded to [data/raw](data/raw). This location is hard-coded, so that the data must not be moved in any way.

### Testing

A dummy data set is included into the repository under [data/raw_dummy](data/raw_dummy), so that the proper environment and the code itself can be tested without having downloaded the full data.  Usage of the dummy data set can be invoked by calling

```python
tools.set_dummy_mode(True)
```

before any data loading executed. You can run a small test suite that will verify the basic functionality of the code by calling

    >> python src/tests/test_base.py

### Using the module

Once you are set up, you can add the parent folder to your Python path and import the module in the usual way, for example

```python
import sys
sys.path.insert(0, "path/to/phc_mode_clustering")
from src import tools, in_out, visualize as vis
```

The **[How to Perform the Clustering](notebooks/How%20to%20Perform%20the%20Clustering.ipynb)** notebook will introduce you to the module, demonstrate a complete clustering procedure and show some basic plotting. It also demonstrates persistent storage of the clustering results, including the models itself.

The **[Loading and Plotting of Stored Data](notebooks/Loading%20and%20Plotting%20of%20Stored%20Data.ipynb)** demonstrates loading of stored clustering data, using a small example data set distributed with this package. It moreover shows some advanced plotting to achieve results similar to those shown in the main publication.


## Database details

The abstract in the [data publication](http://doi.org/10.5442/ND000002) gives a general description of the database structure. In addition, the following table gives descriptions for the columns in the `parameters_and_results.h5/data` table.

Key | Description
:---| -----------
`AccumulatedCPUTime` | Accumulated CPU time, including post processes
`AccumulatedTotalTime` | Accumulated total (wall) time, including post processes
`CpuPerUnknown` | CPU time per unknown fraction
`CpuTime...` | Various CPU time metrics of the solver
`E_1/2` | Integrated field energy enhancement in the superspace volume V_sup of the computational domain (including the hole) for TE/TM polarization
`E_norm` | Energy of the incident plane wave in the superspace volume V_sup, used as a normalization constant in the calculation of E_+
`FEDegree{N}_Percentage` | Percentage of patches with a finite element degree (i.e. polynomial degree of the ansatz functions) of N, for N in [0...10]
`Level` | Refinement level
`SystemMemory_GB` | Consumed memory during the simulation in GB
`TotalMemory_GB` | Total memory of the solve
`TotalTime...` | Various wall time metrics of the solver
`Unknowns` | Number of unknowns (degrees of freedom) in the FEM simulation
`a_1/2_by_p_in` | Absorption in the superspace volume V_sup for TE/TM polarization, normalized to the incident power
`conservation1/2` | Conservation metric Reflectance + Transmittance + Absorption  in the superspace volume V_sup for TE/TM polarization, used as a convergence/validity estimator
`d` | Center diameter of the holes
`e_11...e_24` | Field energy for a specific polarization and domain (format: `e_{polarization}{domain}`.
`fem_degree_max` | The maximum FEM degree used in the adaptive approach
`h` | Height, i.e. extent in *z*-direction, of the slab
`h_sub` | Height, i.e. extent in *z*-direction of the substrate material
`h_sup` | Height, i.e. extent in *z*-direction of the superstrate material
`mat_phc_k...mat_sup_n` | Refractive index (`n` = real part, `k` = imaginary part) for the three domain (subspace, PhC, superspace)
`max_sl_circle` | Maximum side length of the circle in the non-extruded (2D) layout
`max_sl_polygon` | Maximum side length of the polygon in the non-extruded (2D) layout
`max_sl_z_slab` | Maximum side length in *z*-direction for the slab
`max_sl_z_sub` | Maximum side length in *z*-direction for the subspace
`max_sl_z_sup` | Maximum side length in *z*-direction for the superspace
`p` | Pitch, i.e. lattice constant of the hexagonal lattice
`phi` | Polar angle of the direction of incident light
`precision_field_energy` | `Precision` parameter in the `Scattering->Accuracy` section, controlling the numerical accuracy of the near field
`r_1/2` | Reflectance for TE/TM polarization
`t_1/2` | Transmittance for TE/TM polarization
`theta` | Azimuthal angle of the direction of incident light
`vacuum_wavelength` | Vacuum wavelength of the incident light in meter


## Funding

The *German Federal Ministry of Education and Research* is acknowledged for
funding research activities  within the program NanoMatFutur (No. 03X5520)
which made this software project possible.


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


