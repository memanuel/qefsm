import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import Quantiles
from data import load_ex_situ_image, load_echem_images_raw, filter_data_2d

# **************************************************************************************************
def main():
    """
    Clip the pixel values in the raw TIF images to a sensible dynamic range.
    Also convert the original TIF images to PNG format.
    Apply minimal image processing to each image.  Find low and high percentiles of brightness.
    Clip the intensities and then rescale the image in this range.
    This is required because the original TIF images have a very small dynamic range,
    with ~96% of the pixels having a brightness of < 900 out of 65,536.
    The unprocessed TIF images appear as black squares to a human eye, and the image
    alignment feature detector does not run on them.
    The input files are located in data/tif
    The output files are located in data/png_clipped
    """

    # Load the raw data for the bright field image
    B_raw: np.ndarray = load_ex_situ_image()

    # Get shape of data
    shape = B_raw.shape
    m: int = shape[0]
    n: int = shape[0]

    # Load the raw electrochemical images
    I_raw: np.ndarray = load_echem_images_raw(m=m, n=n)

    # Get shape of I
    nc: int = I_raw.shape[0]
    nv: int = I_raw.shape[1]

    # Set quantiles for excluding outliers when converting from TIF to PNG
    quantiles = Quantiles(lo=0.02, hi=0.98)

    # Filter the bright field
    B: np.ndarray
    B, _ = filter_data_2d(data_in=B_raw, quantiles=quantiles)

    # Set colormap for the output PNG files; want grayscale b/c original image was one channel
    cmap: plt.cm = plt.cm.Greys_r

    # The output directory
    outdir: Path = Path('calcs', 'png_clipped')
    # Create output directory if necessary
    outdir.mkdir(parents=True, exist_ok=True)

    # Save bright field as a PNG file
    fname = Path(outdir, 'ex_situ.png')
    plt.imsave(fname, B, cmap=cmap)

    # Filter the electrochemical intensity one image at a time

    # Iterate over concentrations
    for ic in range(nc):
        # Iterate over voltages
        for iv in range(nv):
            # The raw image slice
            I_slc_raw: np.ndarray = I_raw[ic, iv]
            # Filter the raw slice
            I_slc: np.ndarray
            I_slc, _ = filter_data_2d(data_in = I_slc_raw, quantiles=quantiles)
            # Save this echem intensity slice to a PNG file
            fname: str = Path(outdir, f'conc{ic:d}_pot{iv:d}.png')
            plt.imsave(fname, I_slc, cmap=cmap)

# **************************************************************************************************
if __name__ == '__main__':
    main()
