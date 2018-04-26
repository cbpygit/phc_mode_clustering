phc_mode_clustering
==============================

This is the Python code publication supplement to the manuscript
  > C. Barth and C. Becker, “Machine learning classification for field distributions of photonic modes”, ArXiv e-prints (2018), arXiv:1803.08290 [physics.optics].

This code is accompanied by (and partly depends on) the data publication
  > Barth, Carlo; Becker, Christiane (2018): Supplement to: Machine learning classification for field distributions of photonic modes. HZB Data Service. http://doi.org/10.5442/ND000002

*Note that a script for automatic download and verification of the data is included in this module. See the usage notes below.*

At the time when this material was published, the authors are both affiliated with
  > Helmholtz-Zentrum Berlin für Materialien und Energie, Albert-Einstein-Str. 15, 12489 Berlin, Germany.

## Installation

This code can only be cloned as is. See [requirements.txt](requirements.txt) for a list of Python modules on which it depends. This is basically the PyData stack, so e.g. users of anaconda may need minimal installations to get started.

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

The **[How to Perform the Clustering]('How%20to%20Perform%20the%20Clustering.ipynb')** notebook will introduce you to the module, demonstrate a complete clustering procedure and show some basic plotting. It also demonstrates persistent storage of the clustering results, including the models itself.

The **[Loading and Plotting of Stored Data]('Loading%20and%20Plotting%20of%20Stored%20Data.ipynb')** demonstrates loading of stored clustering data, using a small example data set distributed with this package. It moreover shows some advanced plotting to achieve results similar to those shown in the main publication.


## Funding

The *German Federal Ministry of Education and Research* is acknowledged for
funding research activities  within the program NanoMatFutur (No. 03X5520)
which made this software project possible.


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


