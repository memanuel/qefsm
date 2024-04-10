import numpy as np
import scipy.stats

from utils import PixelMask, make_default_pixel_mask_2d, make_default_pixel_mask_4d, \
    calc_masked_ratio

# **************************************************************************************************
def calc_I_rel(B: np.ndarray, I: np.ndarray, mask_B: np.ndarray, mask_I: np.ndarray) \
    -> tuple[np.ndarray, PixelMask]:
    """
    Calculate the relative image intensity
    INPUTS:
        B:              Array of bright field intensities
        I:              Array of image intensities
        mask_B:         Mask for bright field intensity
        mask_I:         Mask for image intensity
    RETURNS:
        I_rel:          Array of relative image intensities (I / B)
        mask_I_rel:     PixelMask for the ratio
    """

    # Relative image intensity
    I_rel: np.ndarray = np.zeros_like(I)
    # Number of concentrations
    nc: int = I.shape[0]
    # Number of voltages
    nv: int = I.shape[1]
    # Number of rows
    m: int = I.shape[2]
    # Number of columns
    n: int = I.shape[3]

    # Mask for ratio I_over_B
    mask_I_rel: PixelMask = make_default_pixel_mask_4d(nc=nc, nv=nv, m=m, n=n)
    
    # Relative intensity on one image slice
    I_slc: np.ndarray = np.zeros((m, n))
    # PixelMask on one image slice
    mask_slc: PixelMask = make_default_pixel_mask_2d(m=m, n=n)

    # Iterate over concentrations and voltages
    for ic in range(nc):
        for iv in range(nv):
            # The numerator of the image intensity ratio - the echem image intensity
            num: np.ndarray = I[ic, iv]
            # Pixel mask for the numerator
            mask_num: np.ndarray = PixelMask(
                ok=mask_I.ok[ic, iv], lo=mask_I.lo[ic, iv], hi=mask_I.hi[ic, iv])
            # The denominator of the image intensity ratio is the bright field intensity
            # Calculate the relative image intensity and its PixelMask on this image slice
            I_slc, mask_slc = calc_masked_ratio(num=num, den=B, mask_num=mask_num, mask_den=mask_B)
            # Copy this slice back to the 4d arrays
            I_rel[ic, iv] = I_slc
            mask_I_rel.ok[ic, iv] = mask_slc.ok
            mask_I_rel.lo[ic, iv] = mask_slc.lo
            mask_I_rel.hi[ic, iv] = mask_slc.hi

    return I_rel, mask_I_rel

# **************************************************************************************************
def calc_corr_B_I(B: np.ndarray, I: np.ndarray, mask_B: np.ndarray, mask_I: np.ndarray) \
    -> np.ndarray:
    """
    Calculate the correlation between the bright field and echem intensity on overlapping good pixels
    INPUTS:
        B:              Array of bright field intensities
        I:              Array of image intensities
        mask_B:         Mask for bright field intensity
        mask_I:         Mask for image intensity
    RETURNS:
        corr:   Array of correlations between B and I. shape (nc, nv)
    """
    # Number of concentrations
    nc: int = I.shape[0]
    # Number of voltages
    nv: int = I.shape[1]
    # Correlation between bright field and echem - good pixels only
    corr: np.ndarray = np.zeros((nc, nv))
    # Iterate through concentrations and voltages
    for ic in range (nc):
        for iv in range(nv):
            # Slice of image intensity data
            I_slc: np.ndarray = I[ic, iv]
            # Overlapping mask for B and I
            ok_BI = mask_B.ok & mask_I.ok[ic, iv]
            # Correlation of B and I on this slice - good pixels only
            corr[ic, iv], _ = scipy.stats.pearsonr(B[ok_BI], I_slc[ok_BI])

    return corr
