import cv2
import numpy as np
from pathlib import Path

from image_utils import align_one_tif

# **************************************************************************************************
def main():
    """
    Align the TIF images of the elecrochemical experiment to the bright field.
    Prcocesses clipped PNG versions of the images because the raw TIF data has
    too small a dynamic range for the feature extractor to find the correspondence points.
    """

    # Read reference image
    fname_ref = 'calcs/png_clipped/ex_situ.png'
    # print('Reading reference image : ', fname_ref)
    img_ref = cv2.imread(fname_ref, cv2.IMREAD_GRAYSCALE)

    # Create output directory if necessary
    outdir: Path = Path('calcs/png_aligned')
    outdir.mkdir(parents=True, exist_ok=True)

    # Number of concentrations
    nc: int = 4
    # Number of voltages
    nv: int = 7

    # Distances implied by homography
    rms_vec: np.ndarray = np.zeros((nc, nv))

    # Index of the concentration
    ic: int
    # Index of the voltage
    iv: int

    # Iterate over concentrations and voltages to align each image
    for ic in range(nc):
        for iv in range(nv):
            rms_vec[ic, iv] = align_one_tif(ic=ic, iv=iv, img_ref=img_ref, verbose=False)

    # Calculate summary statistics for the RMS distance
    rms_mean: float = np.mean(rms_vec)
    rms_std: float = np.std(rms_vec)
    rms_min: float = np.min(rms_vec)
    rms_max: float = np.max(rms_vec)

    # Print summary statistics for the RMS distance
    print('Summary statistics for RMS distance implied by homography matrices:')
    print(f'mean: {rms_mean:8.3f}')
    print(f'std : {rms_std:8.3f}')
    print(f'min : {rms_min:8.3f}')
    print(f'max : {rms_max:8.3f}')

    # Print sorted list of RMS distance
    rms_sorted = list(np.sort(rms_vec.flatten()))
    print('\nSorted RMS distance:')
    for x in rms_sorted:
        print(f'{x:6.3f}', end=', ')
    print('')

# **************************************************************************************************
if __name__ == '__main__':
    main()
