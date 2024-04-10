import numpy as np
import scipy.stats
from dataclasses import dataclass

# **************************************************************************************************
@dataclass
class Quantiles:
    """Class to bundle a low and high quantile together as one argument"""
    # The quantile at the low end, e.g. 0.01 to exclude the first percentile
    lo: np.float64
    # The quantile at the high end, e.g. 0.99 to exclude the 99th percentile
    hi: np.float64

# **************************************************************************************************
@dataclass
class PixelMask:
    """Class for masks of pixel categories: good, low, high"""
    # Mask for pixels that are good
    ok: np.ndarray
    # Mask for pixels that are too low
    lo: np.ndarray
    # Mask for pixels that are too high
    hi: np.ndarray

# **************************************************************************************************
def make_default_pixel_mask_2d(m: int, n: int):
    """
    Create a PixelMask with default entries
    INPUTS:
        m:      Number of rows
        n:      Number of columns
    """
    # The shape of the arrays
    shape: tuple[int, int] = (m, n,)
    # Mask for pixels that are good - all of them
    ok: np.ndarray = np.ones(shape, dtype=np.bool_)
    # Mask for pixels that are too low - none
    lo: np.ndarray = np.zeros(shape, dtype=np.bool_)
    # Mask for pixels that are too high - none
    hi: np.ndarray = np.zeros(shape, dtype=np.bool_)
    # Wrap these into a PixelMask
    return PixelMask(ok=ok, lo=lo, hi=hi)

# **************************************************************************************************
def make_default_pixel_mask_4d(nc: int, nv: int, m: int, n: int):
    """
    Create a PixelMask with default entries
    INPUTS:
        nc:     Number of concentrations
        nv:     Number of voltages
        m:      Number of rows
        n:      Number of columns
    """
    # The shape of the arrays
    shape: tuple[int, int, int, int] = (nc, nv, m, n,)
    # Mask for pixels that are good - all of them
    ok: np.ndarray = np.ones(shape, dtype=np.bool_)
    # Mask for pixels that are too low - none
    lo: np.ndarray = np.zeros(shape, dtype=np.bool_)
    # Mask for pixels that are too high - none
    hi: np.ndarray = np.zeros(shape, dtype=np.bool_)
    # Wrap these into a PixelMask
    return PixelMask(ok=ok, lo=lo, hi=hi)

# **************************************************************************************************
def upsample_pixel_mask(mask_2d, nc: int, nv: int):
    """Upsample a PixelMask from 2d to 4d"""
    # Get shape of the 2D array
    m: int = mask_2d.ok.shape[0]
    n: int = mask_2d.ok.shape[1]
    # Shape of the 4d arrays
    shape_4d: tuple[int, int, int, int] = (nc, nv, m, n,)

    # Upsample the bright field pixel mask to 4d arrays aligned with I
    ok: np.ndarray = np.broadcast_to(mask_2d.ok.reshape(1, 1, m, n), shape_4d)
    lo: np.ndarray = np.broadcast_to(mask_2d.lo.reshape(1, 1, m, n), shape_4d)
    hi: np.ndarray = np.broadcast_to(mask_2d.hi.reshape(1, 1, m, n), shape_4d)
    return PixelMask(ok=ok, lo=lo, hi=hi)

# **************************************************************************************************
def calc_masked_ratio(num: np.ndarray, den: np.ndarray, mask_num: PixelMask, mask_den: PixelMask) \
        -> tuple[np.ndarray, PixelMask]:
    """
    Calculate the ratio of num / den and its associated PixelMask on two masked regions
    INPUTS:
        num:        Array for numerator of ratio
        den:        Array for denominator of ratio
        mask_num:   PixelMask for numerator of ratio
        mask_den:   PixelMask for denominator of ratio
    """
    # Create a PixelMask object for the ratio num / den
    ok: np.ndarray = mask_num.ok & mask_den.ok
    lo: np.ndarray = (mask_num.lo | mask_den.hi) & ~ok
    hi: np.ndarray = (mask_num.hi | mask_den.lo) & ~ok
    mask: PixelMask = PixelMask(ok=ok, lo=lo, hi=hi)

    # Initialize quotient to zeros
    Q: np.ndarray = np.zeros_like(num)
    # Take quotient on valid pixels only (avoid possible divide by zero)
    Q[mask.ok] = num[mask.ok] / den[mask.ok]
    # Fill in the low values
    Q[mask.lo] = np.min(Q[mask.ok])
    # Fill in the high values
    Q[mask.hi] = np.max(Q[mask.ok])

    return Q, mask

# **************************************************************************************************
@dataclass
class EchemData:
    """Class to bundle the experimental electochemical data for the SOC study"""
    # The total concentration of electroactive species in millimolar; shape (nc) = (4)
    T: np.ndarray
    # The applied voltage in millivolts; shape (nv) = (7)
    V: np.ndarray
    # The utilization; dimensionless; shape (nc, nv) = (4, 7)
    U: np.ndarray
    # The mean state of charge implied by the "simple model" U(x) = U_bar * (x / L); 
    # x_mid / L = 0.63 for the geometry of this experiment
    S_bar: np.ndarray

    # The number of concentrations
    nc: int
    # The number of voltages
    nv: int

    def __init__(self, T: np.ndarray, V: np.ndarray, U: np.ndarray):
        self.T = T
        self.V = V
        self.U = U
        self.S_bar = 0.63 * U
        self.nc = T.shape[0]
        self.nv = V.shape[0]

# **************************************************************************************************
@dataclass
class ImageData:
    """Class to bundle the experimental image data for the SOC study"""
    # The raw bright field image before processing; shape (m, n) = (1024, 1024)
    B_raw: np.ndarray

    # The bright field image after processing; shape (m, n) = (1024, 1024)
    B: np.ndarray
    # The image of the experimental conditions at each concentration and voltage; shape (nc, nv, m, n)
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
    # The PixelMask for relative intensity: 
    mask_I_rel: PixelMask

    # The number of concentrations
    nc: int
    # The number of voltages
    nv: int
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
        self.nc = I.shape[0]
        self.nv = I.shape[1]
        self.m = I.shape[2]
        self.n = I.shape[3]

        # The mean valid pixel intensity
        self.B_bar: np.float64 = np.mean(B[self.mask_B.ok])
        # The maximum optical factor
        F_max: float = np.max(B[self.mask_B.ok]) / self.B_bar
        # Calculate the optical factor at each pixel
        self.F = np.clip(self.B / self.B_bar, 0.0, F_max)
        # Calculate the the adjusted intensity at each pixel
        self.I_adj = self.I / self.F.reshape((1, 1, self.m, self.n))

        # Placeholder arrays for the relative intensities. These will be updated later.
        # Reason to do it this way: avoid a circular dependency between utils.py and data_aug.py
        self.I_rel = np.zeros_like(I)
        self.mask_I_rel = make_default_pixel_mask_4d(nc=self.nc, nv=self.nv, m=self.m, n=self.n)

        # Save placeholder array for the correlation
        self.corr_BI: np.ndarray = np.zeros((self.nc, self.nv))

# **************************************************************************************************
def report_summary_all(data: np.ndarray, quantiles: Quantiles, name: str, show_quantiles: bool):
    """Report summary statistics including quantiles for an array of image data with no mask"""

    # Calculate summary statistics
    mean: np.float64 = np.mean(data)
    std: np.float64  = np.std(data)
    min: np.float64  = np.min(data)
    max: np.float64  = np.max(data)
    count: np.int32  = np.size(data)

    # Report summary statistics
    print(f'\nSummary statistics:')
    print(f'Name  : {name:s}')
    print(f'Mean  : {mean:8.3f}')
    print(f'Std   : {std:8.3f}')
    print(f'Min   : {min:8.3f}')
    print(f'Max   : {max:8.3f}')
    print(f'Count : {count:d}')

    # Calculate and report quantiles if requested
    if show_quantiles:
        # Get low and high values at selected quantiles
        data_lo = np.quantile(data, quantiles.lo)
        data_hi = np.quantile(data, quantiles.hi)
        # Z scores of selected quantiles
        z_lo = (data_lo - mean) / std
        z_hi = (data_hi - mean) / std

        # Report selecteed quantiles and their z scores
        print(f'Selected quantiles:')
        print('  Quantile :  Value   : Z-score')
        print(f'{quantiles.lo:10.6f} : {data_lo:8.3f} : {z_lo:+8.3f}')
        print(f'{quantiles.hi:10.6f} : {data_hi:8.3f} : {z_hi:+8.3f}')

# **************************************************************************************************
def report_summary_mask(data: np.ndarray, mask: PixelMask, quantiles: Quantiles, 
                        name: str, show_quantiles: bool):
    """Report summary statistics for an array of processed image data"""

    # Alias the valid data and delegate to report_summary_all
    report_summary_all(data=data[mask.ok], quantiles=quantiles, name=name, show_quantiles=show_quantiles)

# **************************************************************************************************
def shifted_corr(X: np.ndarray, Y: np.ndarray, 
                 X_di: int, X_dj: int, Y_di: int, Y_dj: int, m: int, n: int) -> float:
    """
    Calculate the correlation between X and Y after applying a shift to X
    INPUTS:
        X:      Array of input values to be shifted
        Y:      Array of input values - not shifted
        X_di:   Shift applied to X - axis 0
        X_dj:   Shift applied to X - axis 1
        Y_di:   Shift applied to Y - axis 0
        Y_dj:   Shift applied to Y - axis 1
        m:      Size of shared data - axis 0
        n:      Size of shared data - axis 1
    """

    # Shifted X array
    X_shift = X[X_di:m+X_di, X_dj:n+X_dj]
    # Shifted Y array
    Y_shift = Y[Y_di:m+Y_di, Y_dj:n+Y_dj]
    # Calculate the correlation coefficient on flattened arrays
    corr, _ = scipy.stats.pearsonr(X_shift.flatten(), Y_shift.flatten())

    return corr

# **************************************************************************************************
def make_alignment_grid(B: np.ndarray, I: np.ndarray, sz: int):
    """
    Build a grid of correlations between the bright field and a selected electrochemical image.
    INPUTS:
        B:      Bright field image intensity; shape (m, n)
        I:      One experimental image; shape (m, n)
        sz:     Size of shifts considered in each direction.
                For example, if original images are (1024, 1024) and sz=12,
                Then B is restricted to B[12:1012, 12:1012] and the returned grid is 24x24
    RETURNS:
        grid:   Grid of correlations; shape (2*sz, 2*sz)
    """

    # Initialize grid of correlations
    grid_sz: int = 2* sz
    grid_shape: tuple[int, int] = (grid_sz, grid_sz)
    grid: np.ndarray = np.zeros(shape=grid_shape)

    # Size of the images limited to the overlapping region
    m: int = B.shape[0] - grid_sz
    n: int = B.shape[1] - grid_sz

    # The shift on B is always the same; only vary the shift on I
    # B_shift = B[sz:m+sz, sz:n+sz]
    Y_di: int = sz
    Y_dj: int = sz

    # Iterate through shifts on axis 0 (i):
    for i in range(grid_sz):
        # Iterate through shifts on azis 1 (j):
        for j in range(grid_sz):
            # Calculate the correlation with these shifts
            grid[i,j] = shifted_corr(X=I, Y=B, X_di=i, X_dj=j, Y_di=Y_di, Y_dj=Y_dj, m=m, n=n)

    return grid

# **************************************************************************************************
def print_stars():
    """Print a row of stars"""
    print('*' * 80)