"""
Generate small matlab data files (.mat) that will be used to generate figures for paper.
"""

import numpy as np
from scipy.io import savemat
from pathlib import Path
from utils import Quantiles, EchemData, ImageData, print_stars
from data import load_echem_data, load_image_data
from model import ImageModel, make_image_model

# **************************************************************************************************
def save_mat(ec: EchemData, m: ImageModel):
    """Save model output to MatLab .mat files"""

    # Shape of images when saving to disk
    # shape_plot = (m.m, m.n)
    # shape_imgs = (m.nc, m.nv, m.m, m.n)
    # shape_conc = (m.nc, m.nv, m.m, m.n, 3)

    # The output directory
    out_dir: Path = Path('calcs')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract the mean concentration of each species
    conc_mean: np.ndarray = np.zeros((m.nr, 3))
    for r in range (m.nr):
        # Concentrations for this reaction condition
        conc_r: np.ndarray = m.conc[r]
        # Valid pixels for this reaction condition
        ok_r: np.ndarray = m.ok[r]
        # Mean concentration for this reaction; mean taken only over valid pixels
        conc_mean[r, :] = np.mean(conc_r[ok_r], axis=0)
    
    # Extract named arrays for each species and reshape to (4, 7)
    conc_aqds_bar   = conc_mean[:, 0].reshape(m.shape_react)
    conc_h2aqds_bar = conc_mean[:, 1].reshape(m.shape_react)
    conc_qh_bar     = conc_mean[:, 2].reshape(m.shape_react)

    # Create a dictionary of arrays to save
    mdict: dict[str, np.ndarray] = {
        # The total concentration of AQDS in millimolar
        'T':            m.T.reshape(m.shape_react),
        # The applied plotential in millivolts
        'V':            m.V.reshape(m.shape_react),
        # The utilization for each reaction
        'U':            ec.U,
        # The mean image intensity
        'I_bar':        m.I_bar.reshape(m.shape_react),
        # The mean state of charge recovered from the model output
        'S_bar':        m.S_bar_pred.reshape(m.shape_react),
        # The mean concentration of AQDS (oxidized)
        'conc_aqds_bar'  :  conc_aqds_bar,
        # The mean concentration of H2AQDS (reduced)
        'conc_h2aqds_bar':  conc_h2aqds_bar,
        # The mean concentration of QH (dimer)
        'conc_qh_bar':  conc_qh_bar,
    }

    # Save the dictionary to a .mat file
    file_name: str = Path('calcs', 'plot_data.mat')
    savemat(file_name=file_name, mdict=mdict, format='5', do_compression=False, oned_as='column')

# **************************************************************************************************
def main():

    # Load all the electrochemical data
    ec: EchemData = load_echem_data()

    # Set quantiles for excluding outliers in the manually processed bright field image
    quantiles = Quantiles(lo=0.5500, hi=0.9950)

    # Strategy for populating mask on 4d data
    upsample_2d: bool = False

    # Load all the image data with manual processing
    im: ImageData = load_image_data(quantiles=quantiles, upsample_2d=upsample_2d)

    # Initialize an ImageModel object
    m: ImageModel = make_image_model(im=im, ec=ec)

    # Load the model if requested
    m.load_npy()
    print_stars()
    m.report_summary()
    print_stars()
    m.report_coeff()

    # Delegate to save_mat to save model output to MatLab .mat files
    save_mat(ec=ec, m=m)

# **************************************************************************************************
if __name__ == '__main__':
    main()
