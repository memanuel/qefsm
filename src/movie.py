import numpy as np
from scipy.interpolate import PchipInterpolator
from dataclasses import dataclass
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import Quantiles, PixelMask, EchemData, ImageData, make_default_pixel_mask_2d, print_stars
from data import load_image, load_images, filter_data_2d, load_echem_data, load_image_data
from model import ImageModel, make_image_model
from equilibrium import y2conc, conc2s
from plot import plot_image_mask

# **************************************************************************************************
# Keywords for matplotlib plotting
mpl_kwargs = {
    'dpi': 600,
    'format': 'png',
    'bbox_inches': 'tight'
}

# Set matplotlib default font
font = {
    'weight' : 'normal', 
    'size'   : 20
}

# Set default font in matpltlib
mpl.rc('font', **font)

# Use latex in plots
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Normalization for absolute intensity plots
vmin_abs: float = 0.0
vmax_abs: float = 2500.0

# Normalization for SOC plots
vmin_soc: float = 0.0
vmax_soc: float = 1.0

# **************************************************************************************************
@dataclass
class MovieData:
    """Class to bundle the experimental image data used to generate a movie"""
    # The raw bright field image before processing; shape (m, n) = (1024, 1024)
    B_raw: np.ndarray
    # The bright field image after processing; shape (m, n) = (1024, 1024)
    B: np.ndarray
    # The image of the experimental conditions at each frame; shape (nf, m, n)
    I: np.ndarray

    # These are simple, direct calculations from the data

    # The relative image intensity - image intensity / in situ intensity
    I_rel: np.ndarray
    # The adjusted image intensity - image intensity / optical factor
    I_adj: np.ndarray

    # Correlation between B and I
    corr_BI: np.ndarray

    # The brightest pixel in the in situ image
    # B_max: np.float64
    # The mean intensity in the in situ image - averaged over valid pixels only
    B_bar: np.float64
    # The PixelMask object with good, low and high pixels for the bright field; shape (m, n)
    mask_B: PixelMask
    # The PixelMask object with good, low and high pixels for the electrochemical images; shape (nc, nv, m, n)
    mask_I: PixelMask

    # The number of Frames
    nf: int
    # The number of rows
    m: int
    # The number of columns
    n: int

    def __init__(self, B_raw: np.ndarray, B: np.ndarray, I: np.ndarray, 
                 mask_B: PixelMask, mask_I: PixelMask):
        # Copy the arrays
        self.B_raw = B_raw
        self.B = B
        self.I = I
        self.mask_B = mask_B
        self.mask_I = mask_I

        # Shape of data
        self.nf = I.shape[0]
        self.m = I.shape[1]
        self.n = I.shape[2]

        # The mean valid pixel intensity
        self.B_bar: np.float64 = np.mean(B[self.mask_B.ok])
        # The maximum optical factor
        F_max: float = np.max(B[self.mask_B.ok]) / self.B_bar
        # Calculate the optical factor at each pixel
        self.F = np.clip(self.B / self.B_bar, 0.0, F_max)
        # Calculate the the adjusted intensity at each pixel
        self.I_adj = self.I / self.F.reshape((1, 1, self.m, self.n))

# **************************************************************************************************
def make_default_pixel_mask_3d(nf: int, m: int, n: int):
    """
    Create a PixelMask with default entries
    INPUTS:
        nf:     Number of frames
        m:      Number of rows
        n:      Number of columns
    """
    # The shape of the arrays
    shape: tuple[int, int, int] = (nf, m, n,)
    # Mask for pixels that are good - all of them
    ok: np.ndarray = np.ones(shape, dtype=np.bool_)
    # Mask for pixels that are too low - none
    lo: np.ndarray = np.zeros(shape, dtype=np.bool_)
    # Mask for pixels that are too high - none
    hi: np.ndarray = np.zeros(shape, dtype=np.bool_)
    # Wrap these into a PixelMask
    return PixelMask(ok=ok, lo=lo, hi=hi)

# **************************************************************************************************
def filter_data_3d(data_in: np.ndarray, quantiles: Quantiles):
    """
    Filter raw image data to a numpy array in the range of the specified quantiles
    Takes a 3D array of input data (e.g. the electrochemical intensity)
    INPUTS:
        data_in:        A numpy array of raw image data
        quantiles:      A single quantiles object with the fraction of data to be excluded at the 
                        low end (quantiles.lo) and high end (quantiles.hi) respectively
                        e.g., quantiles.lo = 0.01 excludes data below the first percentile
    RETURNS:
        data_out:       Numpy array of clipped data in the range [data_lo, data_hi]
        mask_out:       PixelMask indicating which pixels are good, low and high
    """

    # Get number of frames
    nf: int = data_in.shape[0]
    # Get number of rows
    m: int = data_in.shape[1]
    # Get number of columns
    n: int = data_in.shape[2]

    # Shape of the full 3d data and its masks
    shape_3d: tuple[int, int] = (nf, m, n,)

    # Initialize array to store the clipped data
    data_out: np.ndarray = np.zeros(shape_3d)
    # Mask for the full 4d data set
    mask_out: PixelMask = make_default_pixel_mask_3d(nf=nf, m=m, n=n)

    # The mask on one slice of electrochemical image data
    mask_slc: PixelMask = make_default_pixel_mask_2d(m=m, n=n)

    # Filter the pixels for each image slice separately
    for i in range(nf):
        # Delegate to filter_data_2d
        data_out[i], mask_slc = filter_data_2d(data_in=data_in[i], quantiles=quantiles)
        # Copy the mask data from the slice to the main mask
        mask_out.ok[i] = mask_slc.ok
        mask_out.lo[i] = mask_slc.lo
        mask_out.hi[i] = mask_slc.hi

    return data_out, mask_out

# **************************************************************************************************
def calc_movie_data(quantiles: Quantiles, t: int) -> MovieData:
    """
    Load all the data in this study into an MovieData object
    INPUTS:
        quantiles:          Quantiles of data to exlcude when processing each image slice
        t:                  Total concentration in millimolar
    """
    # Load the raw data for the ex situ image
    fname_b: str = f'movie/data/{t:d}mM_bright.tif'
    B_raw: np.ndarray = load_image(fname_b)

    # The processed bright field image
    B: np.ndarray
    # The mask of pixel categories for the bright field
    mask_B: PixelMask
    # Process image data for the bright field
    B, mask_B = filter_data_2d(data_in=B_raw, quantiles=quantiles)

    # Load the image data - use the aligned version
    fname: str = f'movie/data/{t:d}mM.tif'
    I_raw: np.ndarray = load_images(fname)
    
    # The processed electrochemical image
    I: np.ndarray
    # The mask of pixel categories for the electrochemical image
    mask_I: np.ndarray

    # Process the image data for the electrochemical experiments
    I, mask_I = filter_data_3d(data_in=I_raw, quantiles=quantiles)

    # Number of frames
    nf: int = I.shape[0]
    # Iterate over frames and logically combine masks
    for i in range(nf):
        # Pixels must be ok on B to be ok on I
        mask_I.ok[i] = mask_I.ok[i] & mask_B.ok
        # If a pixel is lo on B, it's lo on I; and a hi pixel on B cannot be lo on I
        mask_I.lo[i] = (mask_I.lo[i] | mask_B.lo) & ~mask_B.hi
        # If a pixel is hi on B, it's hi on I; and a lo pixel on B cannot be lo on I
        mask_I.hi[i] = (mask_I.hi[i] | mask_B.hi) & ~mask_B.lo

    # Wrap all the image data and return it
    md: MovieData =  MovieData(B_raw=B_raw, B=B, I=I, mask_B=mask_B, mask_I=mask_I)
    return md

# **************************************************************************************************
def plot_frame_abs(I: np.ndarray, mask: PixelMask, t: int, f: int):
    """
    Plot the absolute brightness of one frame
    INPUTS:
        I:          Array of image intensity for the reaction condition
        mask:       Pixel mask associated with image I
        t:          Total concentration in millimolar; for chart title and file name
        f:          Frame number being plotted
    """
    # Title and file name
    title = f'Brightness - {t:d} mM'
    fname = f'movie/frames_abs/{t:d}mM_f{f:03d}.png'
    # Normalization function
    norm: mpl.colors.Normalize = mpl.colors.Normalize(vmin=vmin_abs, vmax=vmax_abs)
    # Colormap
    cmap: mpl.cm = plt.cm.cool
    # Delegate to plot_image_mask
    plot_image_mask(data=I, mask=mask, title=title, fname=fname, norm=norm, cmap=cmap)

# **************************************************************************************************
def make_frames_abs(md: MovieData, t: int):
    """Generate all the frames of absolute intensity that are assembled into a movie"""

    # Status
    print(f'B.shape =', md.B.shape)
    print(f'I.shape =', md.I.shape)
    print(f'Rendering {md.nf} frames...')

    # Plot the frames
    for f in tqdm(range(md.nf)):
        # The image slice
        I: np.ndarray = md.I[f, :, :]
        # Plot this frame
        plot_frame_abs(I=I, mask=md.mask_B, t=t, f=f)

# **************************************************************************************************
def estimate_soc(md: MovieData, t: int) -> np.ndarray:
    """Estimate the SOC for all of the frames in the movie"""
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

    # Load the model
    m.load_npy()
    print_stars()
    m.report_coeff()

    # The index of this concentration
    r: int = np.argmax(ec.T == t)
    # print(f'Concentration index r={r:d} is ec.T[r]={ec.T[r]:.0f} mM')

    # The optical factor for this concentration
    F: np.ndarray = md.F[r]
    # Convert total concentration to float
    t: float = float(t)

    # Shape of plots
    shape_plot: tuple[int, int] = (md.m, md.n,)

    # Preallocate array for state of charge on all the frames
    S: np.ndarray = np.zeros((md.nf, md.m, md.n,))

    # Interpolator from relative reduced concentration to adjusted intensity I_adj at this concentration
    y2p: PchipInterpolator = m.func_tbl_y2p[t]
    # print(f'Looked up y2p interpolator')

    # Interpolator from adjusted intensity I_adj to relative reduced concentration at this concentration
    p2y: PchipInterpolator = m.func_tbl_p2y[t]
    # print(f'Looked up p2y interpolator')

    # Get the valid range of intensities that can be interpolated to SOC at this concentration
    P_min: float = y2p(0.0)
    P_max: float = y2p(1.0)
    # print(f'Got allowed range for P: {P_min:0.3f} <= P <= {P_max:0.3f}')

    # The image intensity input used to estimate the SOC
    for f in range(md.nf):
        I_data: np.ndarray = md.I[f]

        # Light production for this reaction on the this image slice, clipped to the valid range
        P: np.ndarray = np.clip(I_data / F, P_min, P_max).flatten()
        # print(f'Clipped P to valid range with shape {P.shape}')

        # The interpolated relative reduced concentration
        y: np.ndarray = p2y(P)
        # print(f'Estimated y with shape {y.shape}')

        # The relative concentrations of all three species; [X, Y, Z]
        conc = y2conc(y=y, t=t)
        # print(f'Estimated concentrations with shape {conc.shape}')

        # The state of charge on this frame
        S[f] = conc2s(conc=conc).reshape(shape_plot)

    return S

# **************************************************************************************************
def plot_frame_soc_v1(S: np.ndarray, mask: PixelMask, t: int, f: int):
    """
    Plot the SOC of one frame
    INPUTS:
        S:          Array of estimated state of charge
        mask:       Pixel mask associated with SOC
        t:          Total concentration in millimolar; for chart title and file name
        f:          Frame number being plotted
    """
    # Title and file name
    title = f'State of Charge - {t:d} mM'
    fname = f'movie/frames_soc/{t:d}mM_f{f:03d}.png'
    # Normalization function
    norm: mpl.colors.Normalize = mpl.colors.Normalize(vmin=vmin_soc, vmax=vmax_soc)
    # Colormap
    cmap: mpl.cm = plt.cm.cool
    # Delegate to plot_image_mask
    plot_image_mask(data=S, mask=mask, title=title, fname=fname, norm=norm, cmap=cmap)
    

# **************************************************************************************************
def plot_frame_soc(S: np.ndarray, mask: PixelMask, t: int, f: int):
    """
    Plot the SOC of one frame
    INPUTS:
        S:          Array of estimated state of charge
        mask:       Pixel mask associated with SOC
        t:          Total concentration in millimolar; for chart title and file name
        f:          Frame number being plotted
    """

    # Title and filename
    title = f'State of Charge - {t:d} mM'
    fname = f'movie/frames_soc/{t:d}mM_f{f:03d}.png'
    # Normalization function; always plot SOC in the range [0.0, 1.0]
    norm: mpl.colors.Normalize = mpl.colors.Normalize(vmin=vmin_soc, vmax=vmax_soc)
    # Colormap - use jet because I'm from the 1970s
    cmap: mpl.cm = plt.cm.jet

    # Build the plot axes
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    # Apply color map to SOC data
    X = cmap(S)
    # Keep only the three color channels (RGB); discard the transparency channel (A)
    X = X[:,:,0:3]

    # Get shape of masks
    m: int = mask.ok.shape[0]
    n: int = mask.ok.shape[1]
    X_shape = (m, n, 3)
    # Upsample the masks for low and high pixels to 3D
    mask_lo_3d = np.broadcast_to(mask.lo.reshape((m, n, 1)), X_shape)
    mask_hi_3d = np.broadcast_to(mask.hi.reshape((m, n, 1)), X_shape)
    
    # Apply the mask for dim pixels; color these black = (0.0, 0.0, 0.0) in RGB
    X[mask_lo_3d] = 0.0
    # Apply the mask for bright pixels; color these white = (1.0, 1.0, 1.0) in RGB
    X[mask_hi_3d] = 1.0

    # Plot the 2D data - it's already colored manually
    ax.imshow(X, cmap=cmap, vmin=0.0, vmax=1.0)
    # Add a colorbar
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))

    # Save the PNG file
    path = Path(fname)
    fig.savefig(path, **mpl_kwargs)
    plt.close(fig)

# **************************************************************************************************
def make_frames_soc(S: np.ndarray, mask: PixelMask, t: int) -> np.ndarray:
    """Build the frames for the SOC movie"""
    # Number of frames
    nf: int = S.shape[0]
    # Status message
    print(f'Rendering {nf:d} SOC frames...')
    # Plot the frames
    for f in tqdm(range(nf)):
        plot_frame_soc(S=S[f], mask=mask, t=t, f=f)

# **************************************************************************************************
def main():
    """Entry point for console program"""
    
    # The total concentration being plotted
    t: int = 20

    # Set quantiles for excluding outliers in the manually processed bright field image
    quantiles = Quantiles(lo=0.5500, hi=0.9950)

    # Load movie data
    md: MovieData = calc_movie_data(quantiles=quantiles, t=t)

    # Build the frames with aboslute brightness
    # make_frames_abs(md=md, t=t)

    # Estimate the SOC in the movie frames
    S: np.ndarray = estimate_soc(md=md, t=t)

    # SOC summary statistics
    s_bar = np.mean(S)
    s_min = np.min(S)
    s_max = np.max(S)

    # Report SOC summary for movie
    print(f'Summary of estimated state of charge:')
    print('S.shape =', S.shape)
    print(f'Mean: {s_bar:0.3f}')
    print(f'Min : {s_min:0.3f}')
    print(f'Max : {s_max:0.3f}')

    # Build the frames with SOC
    make_frames_soc(S=S, mask=md.mask_B, t=t)

# **************************************************************************************************
if __name__ == '__main__':
    main()
