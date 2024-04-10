import numpy as np
import scipy.stats
from pathlib import Path

from utils import Quantiles, EchemData, ImageData, \
    report_summary_all, report_summary_mask, make_alignment_grid, print_stars
from data import load_echem_data, load_image_data, load_fibers_tc, locate_fibers
from plot import plot_fibers, plot_intensity_all, plot_hist_all, plot_alignment, \
    plot_ex_situ, plot_I_vs_S, plot_I_reg
from model import I_reg, ImageModel, make_image_model


# **************************************************************************************************
# Should we report summary statistics?
report_summary: bool = True

# Should we load the model parameters?
load_model: bool = False

# Should we estimate the model parameters?
estimate_model: bool = True

# Should we save the model to numpy?
save_model: bool = True

# Should we save to matlab?
save_matlab: bool = True

# Should we rebuild the plots? Tune this manually to save time
make_plots_data: bool = True
make_plots_model: bool = True

# **************************************************************************************************
def compare_fibers(B_raw: np.ndarray):
    """Compare fiber classification between TC and MSE"""

    # Identify fibers using Gaussian mixture model applied to raw data
    is_fiber_mse: np.ndarray = locate_fibers(B=B_raw, fluid_prob_min=0.95, verbose=False)

    # Create output directory if necessary
    outdir: Path = Path('plots', 'classification')
    outdir.mkdir(parents=True, exist_ok=True)

    # Save image for fibers (MSE)
    title = 'Fiber Classification (MSE)'
    fname = (outdir / 'is_fiber_mse.png').as_posix()
    plot_fibers(is_fiber=is_fiber_mse, title=title, fname=fname)

    # Load mask defined by Tom Cochard
    is_fiber_tc: np.ndarray = load_fibers_tc()
    # Save image for fibers (TC)
    title = 'Fiber Classification (TC)'
    fname = (outdir / 'is_fiber_tc.png').as_posix()
    plot_fibers(is_fiber=is_fiber_tc, title=title, fname=fname)
    
    # Correlation between MSE and TC masks
    corr_fiber: np.float64 
    corr_fiber, _ = scipy.stats.pearsonr(is_fiber_mse.flatten(), is_fiber_tc.flatten())

    # Report fraction of fibers and compare TC and MSE results
    print(f'\nFraction of fibers (TC)  = {np.mean(is_fiber_tc):8.6f}')
    print(f'Fraction of fibers (MSE) = {np.mean(is_fiber_mse):8.6f}')
    print(f'Correlation between MSE and TC fiber classifications: {corr_fiber:8.6f}')

# **************************************************************************************************
def report_grid_alignment(ec: EchemData, im: ImageData, ic: int, iv: int):
    """Calculate, report and plot alignment grid - correlation of I vs. B with shifts applied"""

    # Calculate correlation grid for the selected indices
    grid: np.ndarray = make_alignment_grid(B=im.B, I=im.I[ic,iv], sz=12)

    # The concentration
    conc: int = int(ec.T[ic])
    # The voltage
    pot: int = int(ec.V[iv])

    # Save plot of correlation grid
    title = f'Alignment: {conc:d} mM, 225 mV'
    fname = f'alignment_c{conc:d}_v{pot:d}.png'
    plot_alignment(grid=grid, title=title, fname=fname)

    # Report maximum alignment
    di: int
    dj: int
    (di, dj) = np.unravel_index(np.argmax(grid), grid.shape)
    corr_max: float = np.max(grid)
    print(f'Maximum correlation at (di, dj) = ({di:d}, {dj:d}). corr_max = {corr_max:0.6f}')

# **************************************************************************************************
def report_regression(ec: EchemData, im: ImageData, make_plot: bool):
    """Run regression and report the results"""

    # Regression of I vs. concentration
    coef, I_vec, I_pred, se = I_reg(ec=ec, im=im)
    # Calculate the R sqaured for this regression
    r2: float = np.var(I_pred) / np.var(I_vec)

    # Report the coefficients and R squared
    print('Regression: I_bar vs. concentration')
    print(f'alpha   = {coef[0]:9.3f}')
    print(f'beta    = {coef[1]:9.3f}')
    print(f'gamma   = {coef[2]:9.3f}')
    print(f'delta   = {coef[3]:9.3f}')
    print(f'R2      = {r2:9.6f}')
    print(f'Std Err = {se:9.6f}')

    # Build the regression plot of I vs. concentration if requested
    if make_plot:
        plot_I_reg(I_vec=I_vec, I_pred=I_pred)

# **************************************************************************************************
def report_naive_regression(ec: EchemData, im: ImageData, make_plot: bool):
    """Run naive regression of  and report the results"""
    # Initialize array of mean image intensity
    I_bar: np.ndarray = np.zeros((im.nc, im.nv))
    # Initialize array of reduced concentration
    X: np.ndarray = np.zeros((im.nc, im.nv))
    # Caculate mean intensity over the image slices
    for ic in range(im.nc):
        # Total concentration
        t: float = ec.T[ic]
        for iv in range(im.nv):
            # The slice of image data; all pixels, not just valid ones
            I_slc: np.ndarray = im.I[ic, iv]
            # Mask of valid pixels for this image slice
            ok_slc: np.ndarray = im.mask_I.ok[ic, iv]
            # Mean of the valid data
            I_bar[ic, iv] = np.mean(I_slc[ok_slc])
            # Reduced concentration
            X[ic, iv] = ec.S_bar[ic, iv] * t

    # Naive regression of I vs. S_bar
    m: float
    b: float
    m, b = np.polyfit(X.flatten(), I_bar.flatten(), 1)
    # The predicted values
    I_pred: np.ndarray = m * X + b
    # Calculate the R squared for this regression
    r2: float = np.var(I_pred) / np.var(I_bar)
    # Calculate the standard error
    I_se: float = np.sqrt(np.mean(np.square(I_bar - I_pred)))

    # Report the coefficients and R squared
    print('\nNaive Regression: I_bar vs. T * SOC')
    print(f'beta    = {m:9.3f}')
    print(f'delta   = {b:9.3f}')
    print(f'R2      = {r2:9.6f}')
    print(f'Std Err = {I_se:9.6f}')

    # Build the regression plot of I vs. SOC if requested
    if make_plot:
        plot_I_vs_S(im=im, ec=ec)

# **************************************************************************************************
def main():
    """Entry point for console program"""

    # Load all the electrochemical data
    ec: EchemData = load_echem_data()

    # Set quantiles for excluding outliers in the manually processed bright field image
    # quantiles = Quantiles(lo=0.5000, hi=0.9975)
    quantiles = Quantiles(lo=0.5500, hi=0.9950)

    # Strategy for populating mask on 4d data
    upsample_2d: bool = False

    # Load all the image data with manual processing
    im: ImageData = load_image_data(quantiles=quantiles, upsample_2d=upsample_2d)

    # Quantiles for summarizing processed data
    quantiles_summ = Quantiles(lo=0.20, hi=.80)

    # Report summary statistics if requested
    if report_summary:
        # Report summary statistics for raw ex situ data
        name: str ='ex situ intensity (raw, all pixels)'
        report_summary_all(data=im.B_raw, name=name, quantiles=quantiles, show_quantiles=True)

        # Summary for processed ex situ data
        name = 'ex situ intensity (processed, good pixels)'
        report_summary_mask(data=im.B, mask=im.mask_B, 
                            quantiles=quantiles_summ, name=name, show_quantiles=True)

        # Compare fiber classification
        compare_fibers(B_raw = im.B_raw)

        # Report summary statistics for electrochemical image intensity - absolute
        name = 'electrochemical image intensity (absolute, good pixels)'
        report_summary_mask(data=im.I, mask=im.mask_I, name=name, 
                            quantiles=quantiles_summ, show_quantiles=True)

        # Report summary statistics for electrochemical image intensity - relative
        name = 'electrochemical image intensity (relative, good pixels)'
        report_summary_mask(data=im.I_rel, mask=im.mask_I_rel, name=name, 
                            quantiles=quantiles_summ, show_quantiles=True)

        # Report statistics for correlations of B vs. I
        name = 'correlation B vs. I'
        report_summary_all(data=im.corr_BI, name=name, 
                        quantiles=quantiles_summ, show_quantiles=False)

    # Build plots of the data if requested
    if make_plots_data:
        # Create output directories if necessary
        sub_dirs: list[str] = ['intensity_abs', 'intensity_rel', 'scatter', 'soc', 'soc_hist']
        for sub_dir in sub_dirs:
            outdir: Path = Path('plots', sub_dir)
            outdir.mkdir(parents=True, exist_ok=True)

        # Plot all the intensities: bright, absolute, relative
        plot_intensity_all(ec=ec, im=im)
        # Plot all the histograms: bright, absolute, relative
        plot_hist_all(im=im)
    # Always rebuild the ex-situ image
    else:
        plot_ex_situ(im=im)

    # Report regression vs. concentration: both good way and naive way
    print_stars()
    report_regression(ec=ec, im=im, make_plot=True)
    # report_naive_regression(ec=ec, im=im, make_plot=True)

    # Initialize an ImageModel object
    m: ImageModel = make_image_model(im=im, ec=ec)

    # Load the model if requested
    if load_model:
        m.load_npy()
        print_stars()
        m.report_summary()
        print_stars()
        m.report_coeff()

    # Estimate the model if requested
    if estimate_model:
        max_iter: int = 10
        thresh: float = 1.0E-3
        m.estimate(max_iter=max_iter, thresh=thresh, verbose=True)
        print_stars()
        m.report_coeff()
        
    # Save model files to numpy if requested or if model was estimated
    if save_model or estimate_model:
        m.save_npy()

    # Save matlab if requested
    if save_matlab:
        m.save_mat()

    # Run plots from the calibrated model if requested
    if make_plots_model:
        # Build scatter plot of mean intensity - actual vs. predicted
        m.plot_I_bar_scatter()
        # Scatter plot of S_bar - actual vs. predicted
        m.plot_S_bar_scatter()
        # Plot all the SOC images
        m.plot_soc_all()
        # Plot all the SOC histograms
        m.plot_soc_hist_all()

# **************************************************************************************************
if __name__ == '__main__':
    main()
