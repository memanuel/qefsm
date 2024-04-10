# Introduction
This GitHub repository contains all the data and source code required to replicate the analysis and figures in our publication *Quantitative Local State of the Charge Mapping by Operando Electrochemical Fluorescence Microscopy in Porous Electrodes*.
These are the steps to replicate the calculations on your own computer.

### Clone the Repository from GitHub
    git clone https://github.com/memanuel/qefsm.git

### Create an Anaconda environment with the necessary packages
Install Anaconda if it is not already on your system. 
See https://conda.io/projects/conda/en/latest/user-guide/install/index.html

Then run these commands in a terminal if you are on a Windows platform:

    > cd \your\path\to\qefsm
    > conda env create -f src/environment_windows.yml
    > conda activate qefsm

If you are on a Linux platform, run these commands:

    $ cd /your/path/to/qefsm
    $ conda env create -f src/environment_linux.yml
    $ conda activate qefsm

This will create a new conda environment on your system named qefsm that includes all the packages required to run the Python programs.
This software has been tested on both Windows and Linux platforms. It has not been tested on a Mac platform.

### Run the Python scripts in order
    (qefsm) python src/clip_images.py
    (qefsm) python src/align_images.py
    (qefsm) python src/qsoc.py
    (qefsm) python src/make_plot_data.py

The script `qsoc.py` has several configurable flags at the beginning. It takes several minutes to generate all the plots depending on the speed of the computer. The calibration process can be run without rebuilding the plots by setting `make_plots_data` and `make_plots_model` to `False`.