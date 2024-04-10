import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

from utils import PixelMask, EchemData, ImageData, calc_masked_ratio

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
# Normalization for relative intensity plots
vmin_rel: float = 0.0
vmax_rel: float = 1.25

# **************************************************************************************************
def plot_rgb_array(X: np.ndarray, title: str, fname: str, norm: mpl.colors.Normalize, cmap: mpl.cm):
    """
    Plot a data array using a color map
    INPUTS:
        X:      Numpy array of data of shape (m, n, 3)
                Last axis has (R, G, B) in range [0.0, 1.0]
        title:  Title of the plot
        fname:  Name of file to save a color PNG image.
        norm:   Normalization function that was applied to the data
        cmap:   Color map used to color the data
    """
    # Build the plot axes
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot the 2D data - it's already colored manually
    ax.imshow(X)
    # Add a colorbar
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    # Save the PNG file
    path = Path(fname)
    fig.savefig(path, **mpl_kwargs)
    plt.close(fig)

# **************************************************************************************************
def plot_image(data: np.ndarray, title: str, fname: str, norm: mpl.colors.Normalize, cmap: mpl.cm):
    """
    Plot a data array using a color map
    INPUTS:
        data:   Numpy array of data
        title:  Title of the plot
        fname:  Name of file to save a color PNG image.
        norm:   Normalization function to be applied to the data
        cmap:   Color map used to color the data
    """
    # Apply color map to normalized data
    X = cmap(norm(data))
    # Keep only the three color channels (RGB); discard the transparency channel (A)
    X = X[:,:,0:3]

    # Delegate to plot_rgb_array
    plot_rgb_array(X=X, title=title, fname=fname, norm=norm, cmap=cmap)

# **************************************************************************************************
def plot_image_mask(data: np.ndarray, mask: PixelMask, title: str, fname: str, 
                    norm: mpl.colors.Normalize, cmap: mpl.cm):
    """
    Save a data array with a mask using a color map
    INPUTS:
        data:   Numpy array of data
        mask:   PixelMask object with masks for good, low and high pixels
        title:  Title of the plot
        fname:  Name of file to save a color PNG image.
        norm:   Normalization function to be applied to the data
        cmap:   Color map used to color the data
    """
    # Apply color map to normalized data
    X = cmap(norm(data))
    # Keep only the three color channels (RGB); discard the transparency channel (A)
    X = X[:,:,0:3]

    # Get shape of masks
    m: int = mask.ok.shape[0]
    n: int = mask.ok.shape[1]
    X_shape = (m, n, 3)
    # Upsample the masks for low and high pixels to 3D
    mask_lo_3d = np.broadcast_to(mask.lo.reshape((m, n, 1)), X_shape)
    mask_hi_3d = np.broadcast_to(mask.hi.reshape((m, n, 1)), X_shape)
    
    # Apply the mask for dim pixels; color these black
    X[mask_lo_3d] = 0.0
    # Apply the mask for bright pixels; color these white
    X[mask_hi_3d] = 1.0

    # Delegate to plot_rgb_array
    plot_rgb_array(X=X, title=title, fname=fname, norm=norm, cmap=cmap)

# **************************************************************************************************
def plot_fibers(is_fiber: np.ndarray, title: str, fname: str):
    """Plot fiber classification"""
    # Take complement so fibers show in black
    is_fluid: np.ndarray = (~is_fiber).astype(np.float64)
    
    # Normalization function
    norm: mpl.colors.Normalize = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    # Colormap is gray scale
    cmap: mpl.cm = plt.cm.Greys
    
    # Delegate to plot_image
    plot_image(data=is_fluid, title=title, fname=fname, norm=norm, cmap=cmap)    

# **************************************************************************************************
def plot_ex_situ(im: ImageData):
    """Plot image of the ex situ solution"""
    # Title and file name
    title: str = 'Ex Situ Image'
    fname: str = 'plots/ex_situ.png'
    # Normalization function
    norm: mpl.colors.Normalize = mpl.colors.Normalize(vmin=vmin_abs, vmax=vmax_abs)
    # Colormap is cool - sequential and different from both black and white 
    cmap: mpl.cm = plt.cm.cool

    # Delegate to plot_image_mask
    plot_image_mask(data=im.B, mask=im.mask_B, title=title, fname=fname, norm=norm, cmap=cmap)

# **************************************************************************************************
def plot_echem_abs_one(I: np.ndarray, mask_I: PixelMask, conc: int, pot: int):
    """
    Plot the absolute brightness of an electrochemical experiment
    INPUTS:
        I:          Array of image intensity for the reaction condition
        mask_I:     Pixel mask associated with image I
        conc:       Concentration in millimolar; for chart title and file name
        pot:        Potential in millivolts; for chart title and file name    
    """
    # Title and file name
    title = f'Brightness - {conc:d} mM, {pot:d} mV'
    fname = f'plots/intensity_abs/I_abs_c{conc:d}_v{pot:03d}.png'
    # Normalization function
    norm: mpl.colors.Normalize = mpl.colors.Normalize(vmin=vmin_abs, vmax=vmax_abs)
    # Colormap
    cmap: mpl.cm = plt.cm.cool

    # Delegate to plot_image_mask
    plot_image_mask(data=I, mask=mask_I, title=title, fname=fname, norm=norm, cmap=cmap)

# **************************************************************************************************
def plot_echem_rel_one(I: np.ndarray, B: np.ndarray, 
                       mask_I: PixelMask, mask_B: PixelMask, conc: int, pot: int):
    """
    Plot the relative brightness (over ex situ) of one electrochemical experiment
    INPUTS:
        I:          Array of image intensity for the reaction condition
        B:          Array of image intensity for the reference ex situ pixel (B for bright)
        mask_I:     Pixel mask associated with elecrochemical image I
        mask_B:     Pixel mask associated with ex situ image B
        conc:       Concentration in millimolar; for chart title and file name
        pot:        Potential in millivolts; for chart title and file name
    """
    # The relative intensity
    I_rel: np.ndarray
    # The mask for the relative intensity
    mask_I_rel: PixelMask
    # Calculate the relative intensity and its mask with calc_masked_ratio
    I_rel, mask_I_rel = calc_masked_ratio(num=I, den=B, mask_num=mask_I, mask_den=mask_B)

    # Title and file name
    title = f'Relative Brightness - {conc:d} mM, {pot:d} mV'
    fname = f'plots/intensity_rel/I_rel_c{conc:d}_v{pot:03d}.png'
    # Normalization function
    norm: mpl.colors.Normalize = mpl.colors.Normalize(vmin=vmin_rel, vmax=vmax_rel)
    # Colormap
    cmap: mpl.cm = plt.cm.cool

    # Delegate to plot_image_mask
    plot_image_mask(data=I_rel, mask=mask_I_rel, title=title, fname=fname, norm=norm, cmap=cmap)

# **************************************************************************************************
def plot_intensity_all(ec: EchemData, im: ImageData):
    """Plot the absolute and relative brightness of all the electrochemical experiments"""

    # Generate the ex situ plot
    plot_ex_situ(im=im)

    # Alias the bright field image
    B: np.ndarray = im.B

    # Alias the mask for the ex situ image
    mask_B: PixelMask = im.mask_B

    # Status message
    print('plot_intensity_all: plotting absolute and relative brightness...')

    # Iterate over concentrations
    ic: int
    for ic in range(im.nc):
        # Iterate over voltages
        for iv in range(im.nv):
            # The image slice
            I: np.ndarray = im.I[ic, iv, :, :]
            # The mask slices
            ok: np.ndarray = im.mask_I.ok[ic, iv, :, :]
            lo: np.ndarray = im.mask_I.lo[ic, iv, :, :]
            hi: np.ndarray = im.mask_I.hi[ic, iv, :, :]
            # The mask for the selected image
            mask_I: PixelMask = PixelMask(ok=ok, lo=lo, hi=hi)
            # The concentration as an integer for printing
            conc: int = int(ec.T[ic])
            # The voltage as an integer for printing
            pot: int = int(ec.V[iv])
            # Delegate to plot_echem_abs_one
            plot_echem_abs_one(I=I, mask_I=mask_I, conc=conc, pot=pot)
            # Delegate to plot_echem_rel_one
            plot_echem_rel_one(I=I, B=B, mask_I=mask_I, mask_B=mask_B, conc=conc, pot=pot)
            # Close the plots once done with each reaction
            plt.close('all')
            # Status
            print('.', end='', flush=True)
    print('')

# **************************************************************************************************
def plot_soc(S: np.ndarray, mask: PixelMask, conc: int, pot: int):
    """
    Plot the estimated state of charge for a reaction
    INPUTS:
        S:      Array with estimated state of charge; shape (m, n)
        mask:   PixelMask object for S
        conc:   Total concentration of AQDS
        pot:    Applied potential in millivolts
    Returns:
        None. Saves a plot to the specified location
    """
    # Title and filename
    title: str = f'Estimated State of Charge: {conc:d} mM, {pot:d} mV'
    fname: str = f'soc/soc_c{conc:d}_v{pot:03d}.png'
    # Normalization function; always plot SOC in the range [0.0, 1.0]
    norm: mpl.colors.Normalize = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
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
    path = Path('plots', fname)
    fig.savefig(path, **mpl_kwargs)
    plt.close(fig)

# **************************************************************************************************
def plot_hist(data: np.ndarray, ok: np.ndarray, bins: int, title: str, xlabel: str, fname: str,
            xmin: float = None, xmax: float = None):
    """Plot histogram data on valid pixels"""

    # Extract array of intensities for valid pixels
    x = data[ok]

    # Build the plot axes
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Pixel Count')
    ax.grid(True)

    # Build the histogram
    ax.hist(x=x, bins=bins, density=False, histtype='step', cumulative=False, color='blue')

    # Set the range of the x-axis if specified
    if (xmin is not None) and (xmax is not None):
        ax.set_xlim(xmin=xmin, xmax=xmax)

    # Save histogram to a PNG file
    path = Path('plots', fname)
    fig.savefig(path, **mpl_kwargs)
    plt.close(fig)

# **************************************************************************************************
def plot_hist_ex_situ(im: ImageData):
    """Plot histogram of ex situ image intensity"""

    # Status message
    # print('plot_hist_ex_situ: plotting histograms of pixel intensity...')

    # Number of bins
    bins: int = 64

    # Chart title and axis label
    title = 'Ex Situ Intensity Histogram'
    xlabel = 'Image Intensity'
    fname = 'ex_situ_hist.png'

    # Delegate to plot_hist
    plot_hist(data=im.B, ok=im.mask_B.ok, bins=bins, title=title, xlabel=xlabel, fname=fname)

# **************************************************************************************************
def plot_hist_abs(im: ImageData):
    """Plot histogram of absolute elecrochemical image intensity"""

    # Status message
    # print('plot_hist_abs: plotting histogram of pixel intensity...')

    # Number of bins
    bins: int = 128

    # Chart title and axis label
    title = 'Absolute Image Intensity Histogram (I) - All 28 Reactions'
    xlabel = 'Image Intensity'
    fname = 'echem_hist.png'

    # Delegate to plot_hist
    plot_hist(data=im.I, bins=bins, ok=im.mask_I.ok, title=title, xlabel=xlabel, fname=fname)

# **************************************************************************************************
def plot_hist_rel(im: ImageData):
    """Plot histogram of relative electrochemical image intensity"""

    # Chart title and axis label
    title = 'Relative Intensity Histogram (I / B) - All 28 Reactions'
    xlabel = 'Relative Image Intensity'
    fname = 'echem_rel_hist.png'

    # Number of bins
    bins: int = 256

    # Set x limits
    xmin: float = 0.00
    xmax: float = 1.50

    # Delegate to plot_hist
    plot_hist(data=im.I_rel, ok=im.mask_I_rel.ok, bins=bins, title=title, xlabel=xlabel, fname=fname,
              xmin=xmin, xmax=xmax)

# **************************************************************************************************
def plot_hist_soc_one(S: np.ndarray, ok: np.ndarray, conc: int, pot: int):
    """
    Plot histogram of the estimated state of charge for one iomage
    INPUTS:
        S:      Array with estimated state of charge; shape (mn,)
        ok:     Array of flags indicating which pixels are valid; shape (mn,)
        conc:   Total concentration of AQDS
        pot:    Applied potential in millivolts
    RETURNS:
        None. Saves a plot to the specified location
    """

    # Chart title and axis label
    title = f'State of Charge Histogram - {conc:d} mM, {pot:d} mV'
    xlabel = 'State of Charge'
    fname = f'soc_hist/soc_hist_c{conc:d}_v{pot:03d}.png'

    # Number of bins
    bins: int = 256

    # Set x limits
    xmin: float = 0.0
    xmax: float = 1.0

    # Delegate to plot_hist
    plot_hist(data=S, ok=ok, bins=bins, title=title, xlabel=xlabel, fname=fname, xmin=xmin, xmax=xmax)
    
# **************************************************************************************************
def plot_hist_all(im: ImageData):
    """Plot histograms of bright field; absolute; and relative image intensities"""

    # Status message
    print('plot_hist_all: plotting histograms of pixel intensity...')

    # Delegate to the three implementation functions for each type of histogram
    plot_hist_ex_situ(im=im)
    plot_hist_abs(im=im)
    # plot_hist_rel(im=im)

# **************************************************************************************************
def plot_scatter(X: np.ndarray, Y: np.ndarray, 
                 title: str, xlabel: str, ylabel: str, fname: str, s: float):
    """
    Make a generic scatter plot
    INPUTS:
        X:      Array to plot on the x axis
        Y:      Array to plot on the y axis
        title:  Title of the plot
        xlabel: Label of the x axis
        ylabel: Label of the y axis
        fname:  File name for the output file of the plot
        s:      Weight for dots
    RETURNS:
        None. Saves a plot to the specified location
    """
    # Flatten arrays
    x: np.ndarray = X.flatten()
    y: np.ndarray = Y.flatten()

    # Calculate the slope and intercept of the regression line
    m: float
    b: float
    m, b = np.polyfit(x, y, 1)
    # Calculate the R squared value
    r2: float = np.var(m * x + b) / np.var(y)

    # Calculate the points on the regression line
    x_min: float = np.min(x)
    x_max: float = np.max(x)
    x_reg: np.ndarray = np.linspace(x_min, x_max, 2)
    y_reg: np.ndarray = m * x_reg + b

    # Build the plot axes
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

    # Build the scatter plot
    ax.scatter(x=x, y=y, s=s, color='blue', label='data')
    # Plot the regression line
    label = f'fit (R2={r2:0.6f})'
    ax.plot(x_reg, y_reg, color='red', label=label)
    # Add legend
    ax.legend()

    # Save histogram to a PNG file
    path = Path('plots', fname)
    fig.savefig(path, **mpl_kwargs)
    plt.close(fig)

# **************************************************************************************************
def plot_scatter_mask(X: np.ndarray, Y: np.ndarray, X_ok: np.ndarray, Y_ok: np.ndarray, 
                 title: str, xlabel: str, ylabel: str, fname: str, s: float):
    """
    Make a generic scatter plot on data with masks
    INPUTS:
        X:      Array to plot on the x axis
        Y:      Array to plot on the y axis
        X_ok:   Mask where array X is valid
        Y_ok:   Mask where array Y is valid
        title:  Title of the plot
        xlabel: Label of the x axis
        ylabel: Label of the y axis
        fname:  File name for the output file of the plot
        s:      Weight for dots
    RETURNS:
        None. Saves a plot to the specified location
    """

    # Get mask where both series are valid
    ok: np.ndarray = X_ok & Y_ok

    # Extract arrays on valid region and flatten them
    X_ok: np.ndarray = X[ok].flatten()
    Y_ok: np.ndarray = Y[ok].flatten()

    # Delegate to plot_scatter
    plot_scatter(X=X_ok, Y=Y_ok, title=title, xlabel=xlabel, ylabel=ylabel, fname=fname, s=s)

# **************************************************************************************************
def plot_I_vs_B_one(im: ImageData, ec: EchemData, ic: int, iv: int):
    """
    Build scatter plot of image intensity vs. ex situ for selected concentration and voltage.
    INPUTS:
        im:     ImageData object
        ec:     EchemData obejct
        ic:     Index of selected concentration, e.g. 1 for 20 mM
        iv:     Index of select voltage, e.g. 6 for 225 mV
    RETURNS:
        None. Saves a plot to the specified location
    """
    # Arrays for scatter plot
    X: np.ndarray = im.B
    Y: np.ndarray = im.I[ic, iv]

    # Masks for scatter plot
    X_ok: np.ndarray = im.mask_B.ok
    Y_ok: np.ndarray = im.mask_I.ok[ic, iv]

    # Chart titles / labels
    title: str = 'Image Intensity vs. Ex Situ Field Intensity'
    xlabel: str = 'Ex Situ Image Intensity'
    ylabel: str = 'Electrochemical Image Intensity'
    # The selected concentration
    conc: int = int(ec.T[ic])
    # The selected voltage
    pot: int = int(ec.V[iv])
    # Filename for plot
    fname = f'scatter/scatter_c{conc:d}_v{pot:d}.png'
    # Weight
    s: float = 1.0

    # Delegate to plot_scatter
    plot_scatter_mask(X=X, Y=Y, X_ok=X_ok, Y_ok=Y_ok, title=title, xlabel=xlabel, ylabel=ylabel, fname=fname, s=s)

# **************************************************************************************************
def plot_I_vs_B_all(im: ImageData, ec: EchemData):
    """
    Build scatter plots of image intensity vs. ex situ intensity for all reaction conditions.
    INPUTS:
        im:     ImageData object
        ec:     EchemData obejct
    RETURNS:
        None. Saves a plot to the specified location
    """

    # Iterate over concentrations
    for ic in range(ec.nc):
        # Iterate over voltages
        for iv in range(ec.nv):
            # Delegate to plot_I_vs_B_one
            plot_I_vs_B_one(im=im, ec=ec, ic=ic, iv=iv) 
            # Close the plots once done with each reaction
            plt.close('all')

# **************************************************************************************************
def plot_I_vs_S(im: ImageData, ec: EchemData):
    """
    Build a scatter plot of mean image intensity I_bar vs. mean SOC S_bar over the 28
    electrochemical experiments.
    INPUTS:
        im:     ImageData object
        ec:     EchemData object
    Returns:
        None. Saves a plot to the specified location
    """
    # The mean state of charge for each image slice according to the simple model
    # S_bar: np.ndarray = ec.S_bar
    # The state of charge times the total concentration
    X: np.ndarray = np.zeros((im.nc, im.nv))
    # Initialize array of mean image intensity
    I_bar: np.ndarray = np.zeros((im.nc, im.nv))
    # Caculate mean intensity over the image slices
    for ic in range(im.nc):
        t: float = ec.T[ic]
        for iv in range(im.nv):
            # SOC times T in this image slice
            X[ic, iv] = t * ec.S_bar[ic, iv]
            # The slice of image data; all pixels, not just valid ones
            I_slc: np.ndarray = im.I[ic, iv]
            # Mask of valid pixels for this image slice
            ok_slc: np.ndarray = im.mask_I.ok[ic, iv]
            # Mean of the valid data
            I_bar[ic, iv] = np.mean(I_slc[ok_slc])

    # Plot I_bar vs. S_bar; delegate to plot_scatter
    title: str = 'Mean Intensity vs. (Mean SOC x Total Conc.)'
    xlabel: str = 'Mean State of Charge x Concentration'
    ylabel: str = 'Mean Image Intensity'
    fname: str = 'I_vs_S.png'
    s: float = 20.0
    plot_scatter(X=X, Y=I_bar, title=title, xlabel=xlabel, ylabel=ylabel, fname=fname, s=s)

# **************************************************************************************************
def plot_I_reg(I_vec: np.ndarray, I_pred: np.ndarray):
    """
    Build a scatter plot of mean image intensity I_bar vs. mean regression fit based
    on 28x4 matrix built from concentration at mean SOC plus a constant.
    INPUTS:
        I_vec:  Actual image intensity
        I_pred: Predicted image intensity from regression
    Returns:
        None. Saves a plot to the specified location
    """

    # Plot I_bar vs. S_bar; delegate to plot_scatter
    title: str = 'Mean Intensity vs. Regression Model'
    xlabel: str = 'Mean Image Intensity - Predicted'
    ylabel: str = 'Mean Image Intensity - Actual'
    fname: str = 'I_reg.png'
    s: float = 20.0
    plot_scatter(X=I_vec, Y=I_pred, title=title, xlabel=xlabel, ylabel=ylabel, fname=fname, s=s)

# **************************************************************************************************
def plot_I_model(I_vec: np.ndarray, I_pred: np.ndarray):
    """
    Build a scatter plot of mean image intensity I_bar vs. the mean fitted intensity
    in the calibrated image model.
    INPUTS:
        I_vec:  Actual image intensity
        I_pred: Predicted image intensity from image model
    Returns:
        None. Saves a plot to the specified location
    """

    # Plot I_bar vs. S_bar; delegate to plot_scatter    
    title: str = 'Mean Intensity vs. Calibrated Model'
    xlabel: str = 'Mean Image Intensity - Predicted'
    ylabel: str = 'Mean Image Intensity - Actual'
    fname: str = 'I_model.png'
    s: float = 20.0
    plot_scatter(X=I_vec, Y=I_pred, title=title, xlabel=xlabel, ylabel=ylabel, fname=fname, s=s)

# **************************************************************************************************
def plot_S_bar_scatter(S_bar_data: np.ndarray, S_bar_pred: np.ndarray):
    """
    Build a scatter plot of mean SOC S_bar calculated with the simple model from the mean
    utilization, vs. S_bar predicted by the SOC on good pixels
    INPUTS:
        S_bar_data:     Actual mean SOC (with simple model)
        S_bar_pred:     Predicted image intensity from regression
    Returns:
        None. Saves a plot to the specified location
    """

    # Plot I_bar vs. S_bar; delegate to plot_scatter
    title: str = 'Mean State of Charge: Data vs. Predicted'
    xlabel: str = 'Mean SOC - Predicted'
    ylabel: str = 'Mean SOC - from Utilization'
    fname: str = 'S_data_vs_pred.png'
    s: float = 20.0
    plot_scatter(X=S_bar_data, Y=S_bar_pred, title=title, xlabel=xlabel, ylabel=ylabel, fname=fname, s=s)

# **************************************************************************************************
def plot_alignment(grid: np.ndarray, title: str, fname: str):
    """Plot an alignment grid"""

    # Default normalization function
    norm = plt.Normalize()
    # Apply color map to normalized data
    cmap: mpl.cm = plt.cm.cool
    X = cmap(norm(grid))
    # Keep only the three color channels (RGB); discard the transparency channel (A)
    X = X[:,:,0:3]

    # Build the plot axes
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(title)
    ax.set_xlabel('Shift in i')
    ax.set_ylabel('Shift in j')
    ax.grid(True)

    # Plot color representation of alignment
    ax.imshow(X)

    # Save histogram to a PNG file
    path = Path('plots', fname)
    fig.savefig(path)
    plt.close(fig)
