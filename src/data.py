import numpy as np
from PIL import Image, ImageSequence
from sklearn.mixture import GaussianMixture

from utils import Quantiles, PixelMask, EchemData, ImageData, \
    make_default_pixel_mask_2d, make_default_pixel_mask_4d
from data_aug import calc_I_rel, calc_corr_B_I

# **************************************************************************************************
def load_image(fname: str) -> np.ndarray:
    """
    Load the data in a TIF image to a numpy array
    INPUTS:
        fname:      The name of the TIF file to load, e.g. 'data/Bright.tif'
    RETURNS:
        data:       Numpy array of raw image data
    """
    # Open the TIF image
    img: Image = Image.open(fname)
    # Convert the TIF image to a numpy array of doubles
    data: np.ndarray = np.array(img).astype(np.float64)

    return data

# **************************************************************************************************
def load_images(fname: str) -> np.ndarray:
    """
    Load the data in a multi-page TIF image to a numpy array
    INPUTS:
        fname:      The name of the multi TIF file to load, e.g. 'data/10mM.tif'
    RETURNS:
        data_raw:   A 3D array of images; shape is (k, m, n), indexing (page, row, col)
    """
    # Open the multi-page TIF image
    img: Image = Image.open(fname)

    # Convert the TIF to a list of numpy arrays, one for each page
    data_list: list = list()
    for page in ImageSequence.Iterator(img):
        arr = np.array(page).astype(np.float64)
        data_list.append(arr)
    # Convert the list of 2D arrays to a single 3D array
    data: np.ndarray = np.array(data_list)

    return data

# **************************************************************************************************
def load_echem_data() -> EchemData:
    """
    Load the electrochemical data for these experiments
    INPUTS:
        None
    Returns:
        T:      Total concentration of electroactive species in millimolar
        V:      Applied voltage vs. OCV in millivolts
        U:      Dimensionless fraction of current / flow of oxidized species x electron count
    """
    # Data type for electrochemical data files
    dtype: np.dtype = np.float64

    # Delimiter for electrochemical data files
    delimiter: str = ','

    # Load the total concentration of electroactive species (millimolar)
    T: np.ndarray = np.loadtxt('data/csv/total_concentration.csv', dtype=dtype, delimiter=delimiter)
    # Load the voltage
    V: np.ndarray = np.loadtxt('data/csv/voltage.csv', dtype=dtype, delimiter=delimiter)
    # Load the utilization (dimensionless)
    U: np.ndarray = np.loadtxt('data/csv/utilization.csv', dtype=dtype, delimiter=delimiter)
    # Reshape utilization to (nc, nv)
    U = np.transpose(U)

    # Wrap results into an EchemData object
    return EchemData(T=T, V=V, U=U)

# **************************************************************************************************
def load_ex_situ_image() -> np.ndarray:
    """Load the raw data for the ex situ image"""
    B_raw: np.ndarray = load_image(fname='data/tif/ex_situ.tif')
    return B_raw

# **************************************************************************************************
def load_echem_images_raw(m: int, n: int) -> np.ndarray:
    """
    Load the original image arrays for the electrochemical experiments excluding the ex situ
    INPUTS:
        m:      The number of rows in the images
        n:      The number of columns in the images
    """
    # There are four concentrations
    nc: int = 4
    # There are seven voltages
    nv: int = 7

    # Shape of combined image araray; components index (concentration, voltage, x, y)
    I_shape: tuple[int, int, int, int] = (nc, nv, m, n,)
    # Preallocate big array for the image data; size is (nc, np, m, n)
    I: np.ndarray = np.zeros(I_shape)

    # Iterate through the concentrations
    for ic in range(nc):
        # Iterate through the voltages
        for iv in range (nv):
            # The filename for this image
            fname = f'data/tif/conc{ic:d}_pot{iv:d}.tif'
            # Load data for each concentration into the slices of the combined array I
            I[ic, iv, :, :] = load_image(fname=fname)

    return I

# **************************************************************************************************
def load_echem_images_aligned(m: int, n: int) -> np.ndarray:
    """
    Load the aligned image arrays for the electrochemical experiments
    INPUTS:
        m:      The number of rows in the images
        n:      The number of columns in the images
    """
    # There are four concentrations
    nc: int = 4
    # There are seven voltages
    nv: int = 7

    # Shape of combined image araray; components index (concentration, voltage, x, y)
    I_shape: tuple[int, int, int, int] = (nc, nv, m, n,)
    # Preallocate big array for the image data; size is (nc, np, m, n)
    I: np.ndarray = np.zeros(I_shape)

    # Iterate through the concentrations
    for ic in range(nc):
        # Iterate through the voltages
        for iv in range (nv):
            # The filename for this image
            fname = f'calcs/png_aligned/conc{ic:d}_pot{iv:d}.png'
            # Load data for each concentration into the slices of the combined array I
            I[ic, iv, :, :] = load_image(fname=fname)

    return I

# **************************************************************************************************
def load_fibers_tc():
    """Load mask array of fibers as classified by Tom Cochard"""
    is_fluid: np.ndarray = load_image('data/tif/mask.tif').astype(np.bool_)
    is_fiber: np.ndarray = (~is_fluid)
    return is_fiber

# **************************************************************************************************
def locate_fibers(B: np.ndarray, fluid_prob_min: float, verbose: bool):
    """
    Locate the fibers in the ex situ image using a Gaussian mixture model
    INPUTS:
        B:                  Numpy array of pixel values in the ex situ image
        fluid_prob_min:     Minimum probability that cells labeled valid are classified as liquid
        verbose:            Print out weights, means, stdev and threshold to console?
    RETURNS:
        is_fiber:   Numpy array indicating whether each pixel is a fiber (1) or not (0)
    """
    # Flattened ex situ data; reshape to a column vector for GMM
    B_vec = B.reshape(-1, 1)

    # Fit a Gaussian mixture model with two categories
    gmm: GaussianMixture = GaussianMixture(n_components=2)
    gmm.fit(B_vec)

    # Predict the probabilities of the labels
    label_probs = gmm.predict_proba(B_vec)
    # Class with the largest mean brightness corresponds to fluid
    idx_fluid = np.argmax(gmm.means_.flatten())
    # These points will be labeled as fibers; put them in the same shape as B
    is_fluid: np.ndarray = (label_probs[:,idx_fluid] > fluid_prob_min).reshape(B.shape)
    # Points not classified as fluid are fibers
    is_fiber: np.ndarray = ~is_fluid

    # Check corner case where no fibers are tagged
    has_fibers: bool = np.sum(is_fiber) > 0

    # Threshold (in pixel brightness) associated with the fibers
    threshold: np.float64 = np.max(B[is_fiber]) if has_fibers else 0.0

    # Print results of mixture model if requested
    if verbose:
        print(f'\nGaussian mixture model to locate fibers in ex situ field image:')
        print('weights    = ', gmm.weights_.flatten())
        print('means      = ', gmm.means_.flatten())
        print('stdevs     = ', np.sqrt(gmm.covariances_.flatten()))
        print(f'threshold = {threshold:8.6f}')
        print(f'Fraction of fibers = {np.mean(is_fiber):8.6f}')

    return is_fiber

# **************************************************************************************************
def filter_data_2d(data_in: np.ndarray, quantiles: Quantiles):
    """
    Filter raw image data to a numpy array in the range of the specified quantiles
    Takes a single 2D array of input data (e.g. the ex situ image or one slice of echem)
    INPUTS:
        data_in:        A numpy array of raw image data
        quantiles:      A single quantiles object with the fraction of data to be excluded at the 
                        low end (quantiles.lo) and high end (quantiles.hi) respectively
                        e.g., quantiles.lo = 0.01 excludes data below the first percentile
    RETURNS:
        data_out:       Numpy array of clipped data in the range [data_lo, data_hi]
        mask:           PixelMask indicating which pixels are good, low and high
    """
    # Get low and high quantiles to filter image
    data_lo = np.quantile(data_in, quantiles.lo)
    data_hi = np.quantile(data_in, quantiles.hi)    

    # Mask of low pixels that are too dim
    mask_lo: np.ndarray = (data_in < data_lo)
    # Mask of high pixels that are too bright
    mask_hi: np.ndarray = (data_in > data_hi)
    # The remaining pixels are good
    mask_ok: np.ndarray = ~ (mask_lo | mask_hi)
    # Wrap the masks into a single PixelMask
    mask_out: PixelMask = PixelMask(ok=mask_ok, lo=mask_lo, hi=mask_hi)

    # Clip data in the range [data_lo, data_hi]
    data_out = np.clip(data_in, data_lo, data_hi)

    return data_out, mask_out

# **************************************************************************************************
def filter_data_region_2d(data_in: np.ndarray, mask_in: np.ndarray, quantiles: Quantiles):
    """
    Filter image data in the fluid region to a numpy array in the range of the specified quantiles
    INPUTS:
        data_in:    A numpy array of image data
        mask_in:    A numpy array of flags indicating which pixels are valid (1) vs. invalid (0)
        quantiles:  A single quantiles object with the fraction of data to be excluded at the 
                    low end (quantiles.lo) and high end (quantiles.hi) respectively
    RETURNS:
        data_out:   Numpy array of clipped data in the range [data_lo, data_hi]
        mask_out:   PixelMask describing pixels in the processed image
    """
    # Get low and high quantiles to filter image
    data_lo = np.quantile(data_in[mask_in], quantiles.lo)
    data_hi = np.quantile(data_in[mask_in], quantiles.hi)

    # Mask of NEW low pixels (too dim)
    mask_lo: np.ndarray = mask_in & (data_in < data_lo)
    # Mask of NEW high pixels (too bright)
    mask_hi: np.ndarray = mask_in & (data_in > data_hi)
    # Pixels are good if (1) they were in the original mask_in and (2) aren't low or high
    mask_ok: np.ndarray = mask_in & ~(mask_lo | mask_hi)
    # Wrap the masks into a single PixelMask
    mask_out: PixelMask = PixelMask(ok=mask_ok, lo=mask_lo, hi=mask_hi)

    # Clip data in the range [data_lo, data_hi]
    data_out = np.clip(data_in, data_lo, data_hi)

    return data_out, mask_out

# **************************************************************************************************
def filter_data_4d(data_in: np.ndarray, quantiles: Quantiles):
    """
    Filter raw image data to a numpy array in the range of the specified quantiles
    Takes a 4D array of input data (e.g. the electrochemical intensity)
    INPUTS:
        data_in:        A numpy array of raw image data
        quantiles:      A single quantiles object with the fraction of data to be excluded at the 
                        low end (quantiles.lo) and high end (quantiles.hi) respectively
                        e.g., quantiles.lo = 0.01 excludes data below the first percentile
    RETURNS:
        data_out:       Numpy array of clipped data in the range [data_lo, data_hi]
        mask_out:       PixelMask indicating which pixels are good, low and high
    """

    # Get number of concentrations
    nc: int = data_in.shape[0]
    # Get number of voltages
    nv: int = data_in.shape[1]
    # Get number of rows
    m: int = data_in.shape[2]
    # Get number of columns
    n: int = data_in.shape[3]

    # Shape of the full 4d data and its masks
    shape_4d: tuple[int, int, int, int] = (nc, nv, m, n,)

    # Initialize array to store the clipped data
    data_out: np.ndarray = np.zeros(shape_4d)
    # Mask for the full 4d data set
    mask_out: PixelMask = make_default_pixel_mask_4d(nc=nc, nv=nv, m=m, n=n)

    # The mask on one slice of electrochemical image data
    mask_slc: PixelMask = make_default_pixel_mask_2d(m=m, n=n)

    # Filter the pixels for each image slice separately
    for ic in range(nc):
        for iv in range(nv):
            # Delegate to filter_data_2d
            data_out[ic, iv], mask_slc = filter_data_2d(data_in=data_in[ic, iv], quantiles=quantiles)
            # Copy the mask data from the slice to the main mask
            mask_out.ok[ic, iv] = mask_slc.ok
            mask_out.lo[ic, iv] = mask_slc.lo
            mask_out.hi[ic, iv] = mask_slc.hi

    return data_out, mask_out

# **************************************************************************************************
def filter_data_4d_from_2d(data_in: np.ndarray, mask_2d: np.ndarray):
    """
    Filter 4d raw image data to a numpy array by upsampling a 2d mask of valid pixels.
    INPUTS:
        data_in:        A numpy array of raw image data
        mask_2d:        A PixelMask of 
    RETURNS:
        data_out:       Numpy array of clipped data in the range [data_lo, data_hi]
        mask_out:       PixelMask indicating which pixels are good, low and high
    """
    # Get number of concentrations
    nc: int = data_in.shape[0]
    # Get number of voltages
    nv: int = data_in.shape[1]
    # Get number of rows
    m: int = data_in.shape[2]
    # Get number of columns
    n: int = data_in.shape[3]

    # Shape of the full 4d data and its masks
    shape_4d: tuple[int, int, int, int] = (nc, nv, m, n,)
    # Shape for 2d data to be upsampled to 4d
    shape_up: tuple[int, int, int, int] = (1, 1, m, n)    

    # Initialize array to store the clipped data
    data_out: np.ndarray = np.zeros(shape_4d)
    # Mask for the full 4d data set
    mask_out: PixelMask = make_default_pixel_mask_4d(nc=nc, nv=nv, m=m, n=n)

    # Broadcast the 2D mask to the output mask
    mask_out.ok = np.broadcast_to(mask_2d.ok.reshape(shape_up), shape_4d)
    mask_out.lo = np.broadcast_to(mask_2d.lo.reshape(shape_up), shape_4d)
    mask_out.hi = np.broadcast_to(mask_2d.hi.reshape(shape_up), shape_4d)

    # The input data on one slice
    data_slc: np.ndarray = np.zeros((m, n))

    # Filter the pixels for each image slice separately
    for ic in range(nc):
        for iv in range(nv):
            # Get input data slice
            data_slc = data_in[ic, iv]
            # Get the minimum and maximum on valid pixels of this slice
            min_slc: np.float = np.min(data_slc)
            max_slc: np.float = np.max(data_slc)
            # Clip the data into the valid range and copy this to the output array
            data_out[ic, iv] = np.clip(data_slc, min_slc, max_slc)

    return data_out, mask_out

# **************************************************************************************************
def load_image_data(quantiles: Quantiles, upsample_2d: bool) -> ImageData:
    """
    Load all the image data in this study into an ImageData object
    INPUTS:
        quantiles:          Quantiles of data to exlcude when processing each image slice
        upsample_2d:        Flag indicating how to handle filtering
                            True:  upsample the 2D mask on the ex situ data to the 4d image data
                            False: filter each image slice independently with the input quantiles,
                                   then combine the filter logically with the ex situ filter
    """
    # Load the raw data for the ex situ image
    B_raw: np.ndarray = load_ex_situ_image()

    # Get array shape for each image
    m: int = B_raw.shape[0]
    n: int = B_raw.shape[1]

    # The processed bright field image
    B: np.ndarray
    # The mask of pixel categories for the bright field
    mask_B: PixelMask
    # Process image data for the bright field
    B, mask_B = filter_data_2d(data_in=B_raw, quantiles=quantiles)

    # Load the image data - use the aligned version
    I_raw: np.ndarray = load_echem_images_aligned(m=m, n=n)
    
    # The processed electrochemical image
    I: np.ndarray
    # The mask of pixel categories for the electrochemical image
    mask_I: np.ndarray

    # Process the image data with the strategy specified by upsample_2d
    
    if upsample_2d:
        # Filter the image data by upsampling the 2d mask
        I, mask_I = filter_data_4d_from_2d(data_in=I_raw, mask_2d=mask_B)
    else:
        # Process the image data for the electrochemical experiments
        I, mask_I = filter_data_4d(data_in=I_raw, quantiles=quantiles)

        # Policy decision: pixel categories on ex situ data B are respected on I

        # Number of concentrations
        nc: int = I.shape[0]
        # Number of voltages
        nv: int = I.shape[1]
        # Iterate over reaction conditions and logically combine masks
        for ic in range(nc):
            for iv in range(nv):
                # Pixels must be ok on B to be ok on I
                mask_I.ok[ic, iv] = mask_I.ok[ic, iv] & mask_B.ok
                # If a pixel is lo on B, it's lo on I; and a hi pixel on B cannot be lo on I
                mask_I.lo[ic, iv] = (mask_I.lo[ic, iv] | mask_B.lo) & ~mask_B.hi
                # If a pixel is hi on B, it's hi on I; and a lo pixel on B cannot be lo on I
                mask_I.hi[ic, iv] = (mask_I.hi[ic, iv] | mask_B.hi) & ~mask_B.lo

    # Wrap all the image data
    im: ImageData =  ImageData(B_raw=B_raw, B=B, I=I, mask_B=mask_B, mask_I=mask_I)

    # Update derived data that is not set by the ImageData initialization function

    # Relative image intensity
    im.I_rel: np.ndarray
    # Mask for ratio I_over_B
    im.mask_I_rel: PixelMask
    # Calculate the relative image intensity and its PixelMask; bind results to im
    im.I_rel, im.mask_I_rel = calc_I_rel(B=B, I=I, mask_B=mask_B, mask_I=mask_I)
    # Correlation between B and I; bind results to im
    im.corr_BI: np.ndarray = calc_corr_B_I(B=B, I=I, mask_B=mask_B, mask_I=mask_I)

    # Return the copy of the ImageData with the relative intensity and correlation
    return im
