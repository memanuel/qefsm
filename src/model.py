import numpy as np
import scipy.io
from scipy.interpolate import PchipInterpolator
from scipy.special import logit, expit
from pathlib import Path

from utils import PixelMask, EchemData, ImageData, print_stars
from equilibrium import s2conc_abs, y2conc, make_splines_p2y, make_splines_y2p
from plot import plot_soc, plot_hist_soc_one, plot_S_bar_scatter, plot_I_model

# **************************************************************************************************
def I_reg(im: ImageData, ec: EchemData):
    """
    Fit linear regression model of I_bar vs. concentrations.
    This is used for the initial estimation of the brightness coefficients.
    INPUTS:
        im:     ImageData object
        ec:     EchemData object
    RETURNS:
        coef:   Array of coefficients [alpha, beta, gamma, delta]
                for [AQDS, H2-AQDS, Dimer, Constant]
        I_vec:  Observed mean image intensity as a flattened vector
        I_pred: Predicted mean image intensity from the regression model
        se:     Standard error in the predicted mean image intensity
    """

    # Calculate array of absolute concentrations of all three species
    conc_abs: np.ndarray = np.zeros((ec.nc, ec.nv, 3))
    for ic in range(ec.nc):
        # The total concentration
        t: float = ec.T[ic]        
        # The relative concentration of each species in this slice
        conc_abs[ic, :, :] = s2conc_abs(s=ec.S_bar[ic, :], t=t)

    # Initialize array of mean image intensity
    I_bar: np.ndarray = np.zeros((im.nc, im.nv))
    # Caculate mean intensity over the image slices
    for ic in range(im.nc):
        for iv in range(im.nv):
            # The slice of image data for one reaction; all pixels, not just valid ones
            I_r: np.ndarray = im.I[ic, iv]
            # Mask of valid pixels for this image of one reaction
            ok_r: np.ndarray = im.mask_I.ok[ic, iv]
            # Mean of the valid data on this reaction
            I_bar[ic, iv] = np.mean(I_r[ok_r])

    # Number of reaction conditions
    nr: int = ec.nc * ec.nv

    # Reshape I_bar to a flattened array of shape (nr,)
    I_vec: np.ndarray = I_bar.reshape((nr,))

    # Reshape conc_abs to (n, 4); columns are [x, y, z, 1]
    conc_vec: np.ndarray = np.zeros((nr, 4))
    conc_vec[:, 0:3] = conc_abs.reshape((nr, 3))
    conc_vec[:, 3] = 1.0

    # Solve least squares system
    rcond: float = 1.0E-9
    coef: np.ndarray = np.linalg.lstsq(conc_vec, I_vec, rcond=rcond)[0]

    # Predicted mean intensity
    I_pred: np.ndarray = np.dot(conc_vec, coef)

    # Calculate the standard error; this is the root mean square residual
    se: np.ndarray = np.sqrt(np.mean(np.square(I_vec - I_pred)))

    return coef, I_vec, I_pred, se

# **************************************************************************************************
class ImageModel:
    """Model image intensity on the electrochemical"""

    def __init__(self, nc: int, nv: int, m: int, n:int):
        """Initialize model with empty arrays"""
        # Number of concentrations
        self.nc: int = nc
        # Number of voltages
        self.nv: int = nv
        # Number of rows
        self.m: int = m
        # Number of columns
        self.n: int = n

        # Number of reaction conditions
        self.nr: int = self.nc * self.nv
        # Number of pixels in one image
        self.mn: int = self.m * self.n
        # Total pixels in the data set
        self.N_pix_tot: int = self.nr * self.mn
        # Total valid pixels in the data set
        self.N_pix_ok: int = 0

        # Shape of combined image data after it's flattened; shared by I, Y
        self.shape_imgs = (self.nr, self.mn)
        # Shape of concentrations after it's flattened; last axis is [X, Y, Z]
        self.shape_conc = (self.nr, self.mn, 3)
        # The shape of one image for plotting
        self.shape_plot = (self.m, self.n)
        # Shape of reaction conditions
        self.shape_react = (self.nc, self.nv)

        # Mask of good pixels on B as an array of shape (mn)
        self.B_is_ok: np.ndarray = np.zeros(self.mn, dtype=bool)
        self.B_is_lo: np.ndarray = np.zeros(self.mn, dtype=bool)
        self.B_is_hi: np.ndarray = np.zeros(self.mn, dtype=bool)        

        # Mask of good pixels as an array of shape (nr, mn); applies to I, Y, S
        self.ok: np.ndarray = np.zeros(self.shape_imgs, dtype=bool)

        # The experimental ex-situ image (bright)
        self.B_data: np.ndarray = np.zeros(self.mn)
        # The experimental image intensity as an array of shape (nr, mn)
        self.I_data: np.ndarray = np.zeros(self.shape_imgs)
        # The total concentration as an array of shape (nr)
        self.T: np.ndarray = np.zeros(self.nr)
        # The applied potential as an array of shape (nr)
        self.V: np.ndarray = np.zeros(self.nr)
        # The mean state of charge as an array of shape (nr)
        self.S_bar_data: np.ndarray = np.zeros(self.nr)

        # The mean intensity in the ex situ image on valid pixels
        self.B_bar: float = 0.0
        # The value of the optical factor implied by the ex-situ image, i.e. F0_ij = B_ij / B_bar
        self.F0: np.ndarray = np.zeros(self.mn)
        # The optical factor; shape (mn); a model parameter.  Initial guess is F0
        self.F: np.ndarray = np.zeros(self.mn)

        # The mean image intensity as an array of shape (nr); average taken over good pixels
        self.I_bar: np.ndarray = np.zeros(self.nr)

        # Brightness coefficients for three species in mM; bcoeff = [alpha, beta, gamma]
        self.bcoeff: np.ndarray = np.zeros(3)
        # Brightness constant term, i.e. bconst = delta
        self.bconst: float = 0.0

        # The relative concentration of reduced species; array of shape (nr, mn)
        self.Y: np.ndarray = np.zeros(self.shape_imgs)
        # The predicted relative concentrations; array of shape (nr, mn, 3)
        self.conc = np.zeros(self.shape_conc)
        # The predicted light production (before the optical factor is applied)
        self.P: np.ndarray = np.zeros(self.shape_imgs)
        # The predicted image intensity including the optical factor
        self.I_pred: np.ndarray = np.zeros(self.shape_imgs)
        # The predicted SOC on each image pixel
        self.S: np.ndarray = np.zeros(self.shape_imgs)
        # The predicted mean SOC
        self.S_bar_pred: np.ndarray = np.zeros(self.nr)

        # The standard error and R2 on F, pixelwise with the same shape as F
        self.F_se: np.ndarray = np.zeros(self.mn)
        self.F_r2: np.ndarray = np.zeros(self.mn)

        # The standard error and R2 on I; scalar
        self.I_se: float = 0.0
        self.I_r2: float = 0.0

        # The combined error bar on the reduced species fraction
        self.Y_err: np.ndarray = np.zeros(self.shape_imgs)
        # The error bar on the reduced species fraction - due to uncertainty in I
        self.Y_err_I: np.ndarray = np.zeros(self.shape_imgs)
        # The error bar on the reduced species fraction - due to uncertainty in F
        self.Y_err_F: np.ndarray = np.zeros(self.shape_imgs)

        # The combined error bar on the predicted SOC
        self.S_err: np.ndarray = np.zeros(self.shape_imgs)
        # The error bar on the predicted SOC - due to uncertainty in I
        self.S_err_I: np.ndarray = np.zeros(self.shape_imgs)
        # The error bar on the predicted SOC - due to uncertainty in F
        self.S_err_F: np.ndarray = np.zeros(self.shape_imgs)

        # The previous estimate of some key arrays - used for convergence testing
        # The previous estimate of reduced species fraction
        self.Y_prev: np.ndarray = np.zeros(self.shape_imgs)
        # The previous estimate of the optical factor
        self.F_prev: np.ndarray = np.zeros(self.mn)
        # The previous estimate of the predicted image intensity
        self.I_prev: np.ndarray = np.zeros(self.shape_imgs)
        # Previous values of the brightness coefficients and constant
        self.bcoeff_prev: np.ndarray = np.zeros(3)
        self.bconst_prev: float = 0.0

        # RMS change in reduced species fraction between iterations
        self.Y_chng: float = 0.0
        # RMS change in optical factor between iterations
        self.F_chng: float = 0.0
        # RMS change in predicted image intensity between iterations, relative to the current prediction
        self.I_chng: float = 0.0
        # RMS relative change in brightness coefficients between iterations
        self.bc_chng: float = 0.0

        # The directory for numpy files
        self.dir_npy: Path = Path('calcs/model_npy')

        # Condition number for solving linear equations
        self.rcond: float = 1.0E-9
        # Number of iterations to adjust concentration
        self.ci_max: int = 4

    def load_data(self, im: ImageData, ec: EchemData):
        """Load experimental data"""
        # Mask of good pixels on B as an array of shape (mn)
        self.B_is_ok = im.mask_B.ok.flatten()
        self.B_is_lo = im.mask_B.lo.flatten()
        self.B_is_hi = im.mask_B.hi.flatten()

        # Mask of good pixels as an array of shape (nr, mn)
        self.ok = im.mask_I.ok.reshape(self.shape_imgs)
        self.N_pix_ok = np.sum(self.ok)

        # The experimental ex-situ image (bright)
        self.B_data = im.B.flatten()
        # The experimental image intensity as an array of shape (nr, mn)
        self.I_data = im.I.reshape(self.shape_imgs)

        # The total concentration as an array of shape (nr)
        self.T = np.broadcast_to(ec.T.reshape((self.nc, 1)), self.shape_react).flatten()
        # The applied potential as an array of shape (nr)
        self.V = np.broadcast_to(ec.V.reshape((1, ec.nv)), self.shape_react).flatten()
        # The mean state of charge as an array of shape (nr)
        self.S_bar_data = ec.S_bar.flatten()

    def set_initial_params(self):
        """Set initial values for model parameters"""
        # The mean intensity in the ex situ image on valid pixels
        self.B_bar = np.mean(self.B_data[self.B_is_ok])
        # The minimum optical factor
        F_min: float = np.min(self.B_data[self.B_is_ok]) / self.B_bar
        # The maximum optical factor
        F_max: float = np.max(self.B_data[self.B_is_ok]) / self.B_bar
        # Calculate the optical factor at each pixel
        self.F0 = np.clip(self.B_data / self.B_bar, F_min, F_max)
        self.F = self.F0.copy()

        # Calculate the mean image intensity as an array of shape (nr); average taken over good pixels
        for r in range(self.nr):
            # The adjusted image intensity for this reaction
            I_r: np.ndarray = self.I_data[r]
            # The mask of good pixels for this reaction
            ok_r: np.ndarray = self.ok[r]
            # The mean image intensity for this reaction
            self.I_bar[r] = np.mean(I_r[ok_r])

        # Predict brightness with a linear model from the absolute concentrations
        # [X, Y, Z] = [AQDS, H2-AQDS, Dimer]
        # I_pred = alpha*X + beta*Y + gamma*Z + delta

        # Calculate array of absolute concentrations of all three species consistent with mean SOC
        conc_abs: np.ndarray = np.zeros((self.nr, 4))
        for r in range(self.nr):
            # The total concentration
            t: float = self.T[r]
            # The absolute concentration of each species in this slice
            conc_abs[r, 0:3] = s2conc_abs(s=self.S_bar_data[r], t=t)
            # The constant term
            conc_abs[r, 3] = 1.0

        # Solve least squares system
        reg_coef : np.ndarray = np.linalg.lstsq(conc_abs, self.I_bar, rcond=self.rcond)[0]
        # Get residuals
        resid = self.I_bar - np.dot(conc_abs, reg_coef)
        # Calculate standard error on I
        self.I_se = np.sqrt(np.mean(np.square(resid)))
        # Calculate R2 on this regression
        r2_den: float = np.mean(np.square(self.I_bar))
        self.I_r2 = 1.0 - np.square(self.I_se) / r2_den

        # Brightness coefficients for three species in mM; bcoeff = [alpha, beta, gamma]
        self.bcoeff: np.ndarray = reg_coef[0:3]
        # Brightness constant term, i.e. bconst = delta
        self.bconst: float = reg_coef[3]

        # Build the interpolators with the current brightness coefficients
        self.build_interpolators()

        # Delegate to predict method to populate predictions with initial guess
        self.predict()
        self.predict_bounds()

    def build_interpolators(self) -> None:
        """
        Build monotonic splines to interpolate between relative reduced concentration y 
        and image intensity i using the current brightness coefficients.
        """
        # Stack the regression coefficients into one array of shape (4,)
        reg_coef: np.ndarray = np.append(self.bcoeff, self.bconst)
        # Build splines to interpolate from relative reduced conc. y to image intensity i
        self.func_tbl_y2p: dict[float, PchipInterpolator] = make_splines_y2p(reg_coef=reg_coef)
        # Build splines to interpolate from image intensity i to relative reduced conc. y
        self.func_tbl_p2y: dict[float, PchipInterpolator] = make_splines_p2y(reg_coef=reg_coef)

    def predict_Y(self) -> None:
        """
        Predict the reduced species fraction Y from the image intensity I_pred 
        with the current model parameters.
        """
        # Estimate relative concentration of reduced species at each pixel by interpolating from the adjusted brightness
        for r in range(self.nr):
            # Total concentration for this reaction
            t: float = self.T[r]
            # Interpolator from relative reduced concentration to adjusted intensity I_adj at this concentration
            y2p: PchipInterpolator = self.func_tbl_y2p[t]
            # Get the valid range of intensities that can be interpolated to SOC at this concentration
            P_min: float = y2p(0.0)
            P_max: float = y2p(1.0)
            # Light production for this reaction on the this image slice, clipped to the valid range
            P_r: np.ndarray = np.clip(self.I_data[r] / self.F, P_min, P_max)
            # Interpolator from adjusted intensity I_adj to relative reduced concentration at this concentration
            p2y: PchipInterpolator = self.func_tbl_p2y[t]
            # The interpolated relative reduced concentration
            self.Y[r] = p2y(P_r)

    def predict_PIS(self) -> None:
        """
        Predict the relative concentration, SOC and image intensity at each pixel from the relative 
        reduced species fraction Y with the current model parameters.
        """
        for r in range(self.nr):
            # Total concentration for this reaction
            t: float = self.T[r]
            # The relative concentrations of all three species; [X, Y, Z]
            self.conc[r] = y2conc(y=self.Y[r], t=t)
            # The predicted intensity for this reaction - before optical coefficient
            self.P[r] = (np.dot(self.conc[r], self.bcoeff) * t) + self.bconst
            # The predicted intensity for this reaction
            self.I_pred[r] = self.P[r] * self.F
            # The predicted SOC by pixel is the sum of H2-AQDS (col 1) and dimer (col 2)
            self.S[r] = self.conc[r,:,1] + self.conc[r,:,2]
            # The predicted mean state of charge; average over only good pixels
            ok_r = self.ok[r]
            self.S_bar_pred[r] = np.mean(self.S[r][ok_r])

    def predict(self) -> None:
        """Predict all variables with the current model parameters"""
        self.predict_Y()
        self.predict_PIS()

    def predict_bounds(self) -> None:
        """Predict error bounds on Y and SOC"""
        # The relative error in F
        F_err_rel: np.ndarray = self.F_se / self.F
        # F shifted up
        F_hi: np.ndarray = self.F * (1.0 + F_err_rel)
        # F shifted down
        F_lo: np.ndarray = self.F / (1.0 + F_err_rel)

        for r in range(self.nr):
            # Total concentration for this reaction
            t: float = self.T[r]
            # Interpolator from relative reduced concentration to adjusted intensity I_adj at this concentration
            y2p: PchipInterpolator = self.func_tbl_y2p[t]
            # Get the valid range of intensities that can be interpolated to SOC at this concentration
            P_min: float = y2p(0.0)
            P_max: float = y2p(1.0)
            # Interpolator from adjusted intensity I_adj to relative reduced concentration at this concentration
            p2y: PchipInterpolator = self.func_tbl_p2y[t]

            # The lower bound on the predicted light production using standard error on I
            P_lo_I: np.ndarray = np.clip((self.I_data[r] - self.I_se) / self.F, P_min, P_max)
            # The lower bound on the predicted reduced concentration
            Y_lo_I: np.ndarray = p2y(P_lo_I)
            # Corresponding concentrations at the lower bound of SOC
            conc_lo_I: np.ndarray = y2conc(y=Y_lo_I, t=t)
            # The lower bound on SOC
            S_lo_I: np.ndarray = conc_lo_I[:,1] + conc_lo_I[:,2]
            
            # The upper bound on the predicted light production using standard error on I
            P_hi_I: np.ndarray = np.clip((self.I_data[r] + self.I_se) / self.F, P_min, P_max)
            # The upper bound on the predicted reduced concentration
            Y_hi_I: np.ndarray = p2y(P_hi_I)
            # Corresponding concentrations at the upper bound of SOC
            conc_hi_I: np.ndarray = y2conc(y=Y_hi_I, t=t)
            # The upper bound on SOC
            S_hi_I: np.ndarray = conc_hi_I[:,1] + conc_hi_I[:,2]

            # The lower bound on the predicted light production using standard error on F
            P_lo_F: np.ndarray = np.clip(self.I_data[r] / F_hi, P_min, P_max)
            # The lower bound on the predicted reduced concentration
            Y_lo_F: np.ndarray = p2y(P_lo_F)
            # Corresponding concentrations at the lower bound of SOC
            conc_lo_F: np.ndarray = y2conc(y=Y_lo_F, t=t)
            # The lower bound on SOC
            S_lo_F: np.ndarray = conc_lo_F[:,1] + conc_lo_F[:,2]

            # The upper bound on the predicted light production using standard error on F
            P_hi_F: np.ndarray = np.clip(self.I_data[r] / F_lo, P_min, P_max)
            # The upper bound on the predicted reduced concentration
            Y_hi_F: np.ndarray = p2y(P_hi_F)
            # Corresponding concentrations at the upper bound of SOC
            conc_hi_F: np.ndarray = y2conc(y=Y_hi_F, t=t)
            # The upper bound on SOC
            S_hi_F: np.ndarray = conc_hi_F[:,1] + conc_hi_F[:,2]

            # Error in Y from each source (I and F)
            self.Y_err_I[r] = (Y_hi_I - Y_lo_I) / 2.0
            self.Y_err_F[r] = (Y_hi_F - Y_lo_F) / 2.0

            # Error in S from each source (I and F)
            self.S_err_I[r] = (S_hi_I - S_lo_I) / 2.0
            self.S_err_F[r] = (S_hi_F - S_lo_F) / 2.0

        # Assume independent sources of error and take the Pythagorean sum of errors
        # due to uncertainty in I and F

        # The combined error bar on the reduced species fraction
        self.Y_err = np.hypot(self.Y_err_I, self.Y_err_F)
        # The combined error bar on the state of charge
        self.S_err = np.hypot(self.S_err_I, self.S_err_F)

    def calc_err_I_abs(self) -> np.float64:
        """Calculate the RMS error on the recovered image I; absolute scale in nits"""
        # Calculate an array of absolute errors
        err_vec: np.ndarray = self.I_data - self.I_pred
        return np.sqrt(np.mean(np.square(err_vec[self.ok])))

    def calc_err_I_rel(self) -> np.float64:
        """Calculate the RMS error on the recovered image I; dimensionless scale"""
        # Calculate an array of relative errors; numerator is difference in pixel intensity
        # denominator is mean intensity for this image
        err_vec: np.ndarray = (self.I_data - self.I_pred) / self.I_bar.reshape((self.nr, 1))
        return np.sqrt(np.mean(np.square(err_vec[self.ok])))

    def calc_err_S_bar(self) -> np.float64:
        """Calculate the RMS error on the recovered mean SOC S_bar"""
        return np.sqrt(np.mean(np.square(self.S_bar_pred - self.S_bar_data)))

    def calc_avg_se_S(self) -> np.array:
        """Calculate the standard error on the predicted SOC, averaged over good pixels"""
        return np.sqrt(np.mean(np.square(self.S_err[self.ok]), axis=0))

    def calc_avg_se_S_I(self) -> np.array:
        """Calculate the standard error on the predicted SOC due to I, averaged over good pixels"""
        return np.sqrt(np.mean(np.square(self.S_err_I[self.ok]), axis=0))

    def calc_avg_se_S_F(self) -> np.array:
        """Calculate the standard error on the predicted SOC due to F, averaged over good pixels"""
        return np.sqrt(np.mean(np.square(self.S_err_F[self.ok]), axis=0))

    def calc_avg_se_F(self) -> np.array:
        """Calculate the standard error on the optical factor, averaged over good pixels"""
        # return np.mean(self.F_se[self.B_is_ok])
        return np.sqrt(np.mean(np.square(self.F_se[self.B_is_ok])))

    def calc_bcoeff(self, eta: float) -> np.ndarray:
        """Calculate updated brightness coefficients using linear regression run pixelwise"""
        # Build array of covariates including 1s of shape (N, 4)
        X: np.ndarray = np.zeros((self.N_pix_tot, 4))
        # Build an array of responses of shape (N,)
        Y: np.ndarray = np.zeros((self.N_pix_tot))

        # Slice of covariates for one reaction
        X_r: np.ndarray = np.zeros((self.mn, 4))
        # Slice of responses for one reaction
        Y_r: np.ndarray = np.zeros(self.mn)
        # Optical factors reshaped to (N, 1)
        F: np.ndarray = self.F.reshape((self.mn, 1))

        # Populate both X and Y for slices of rows corresponding to each reaction
        for r in range(self.nr):
            # Total concentration for this reaction
            t = self.T[r].reshape((1,1))
            # Covariates for this reaction, INCLUDING the optical factor
            X_r = self.conc[r] * t * F
            # Responses for this reaction - the measured intensity (which also includes F)
            Y_r: np.ndarray = self.I_data[r]
            # Range of rows in the main covariate and response arrays (combining all reactions)
            i0: int = r * self.mn
            i1: int = i0 + self.mn
            # Mask of valid pixels for this reaction
            ok_r: np.ndarray = self.ok[r]
            # Copy the reaction slices into the full arrays, masking out bad pixels
            X[i0:i1, 0:3] = X_r * ok_r.reshape((self.mn, 1))
            # The last column of X is 1s when regressing against P; here we regress against I
            # so the covariates (both conc_abs and 1) are scaled by F
            X[i0:i1, 3] = self.F * ok_r
            Y[i0:i1] = Y_r * ok_r

        # Solve least squares system
        reg_coef: np.ndarray = np.linalg.lstsq(X, Y, rcond=self.rcond)[0]

        # Do we need to apply an update?
        apply_update: bool = (eta > 0.0)
        # Save brightness coefficients and constant term with updated values
        if apply_update:
            self.bcoeff = eta * reg_coef[0:3] + (1.0 - eta) * self.bcoeff
            self.bconst = eta * reg_coef[3] + (1.0 - eta) * self.bconst
            # Always rebuild the interpolators when the brightness coefficients are updated
            self.build_interpolators()

        # Predicted image intensity from the linear fit (NOT necessarily current model if eta < 1)
        Y_pred: np.ndarray = np.dot(X, reg_coef)
        # Residuals on the predicted image intensity, including Y
        Y_resid: np.ndarray = Y - Y_pred
        
        # Standard error of the linear fit to image intensity, averaged over good pixels
        self.I_se = np.sqrt(np.sum(np.square(Y_resid)) / self.N_pix_ok)
        # R2 of the linear fit to image intensity
        ok_flat: np.ndarray = self.ok.flatten()
        r2_den: float = np.mean(np.square(Y[ok_flat]))
        self.I_r2 = 1.0 - np.square(self.I_se) / r2_den

        # Standard error on I_rel; use mean intensity for each image
        # resid_rel: np.ndarray = Y_resid / self.I_bar.reshape((self.nr, 1))
        # self.I_rel_se = np.sqrt(np.sum(np.square(resid_rel) / self.N_pix_ok))

    def calc_F(self, eta: float) -> None:
        """Calculate an updated optical factor F using linear regression run pixelwise"""
        # Here the covariate X is the light production P and the response Y is the image intensity I
        # The data for each regression at a single pixel includes 28 reactions and the ex situ image
        # Each data point is of the form (P, I) where P is a light production and I is an intensity
        # For the reaction data, P is an estimate that change; for the ex situ, it's a constant
        # equal to the average brightness.

        # Weighting factor applied to the ex situ image
        # w_ex: float = float(self.nr)
        w_ex: float = np.sqrt(float(self.nr))
        # Calculate the covariance E[XY] summed along the reaction conditions; shape (mn,)
        sum_xy: np.ndarray = np.sum(self.P * self.I_data, axis=0)
        # Calculate the augmented sum of XY including the ex-situ image
        sum_xy_aug: np.ndarray = sum_xy + w_ex * self.B_bar * self.B_data
        # Calculate the variance E[XX] summed along the reaction conditions; shape (mn,)
        sum_xx: np.ndarray = np.sum(np.square(self.P), axis=0)
        # Calculate the augmented variance of XX including the ex-situ image
        sum_xx_aug: np.ndarray = sum_xx + w_ex * np.square(self.B_bar)
        # Effective number of observations for each pixel
        # n_obs: float = self.nr + w_ex
        # The revised estimate of the optical factor
        F: np.ndarray = sum_xy_aug / sum_xx_aug
        # The residual image intensity in this regression (not to be confused with the brightness coefficient residuals)
        resid_I = self.I_data - self.P * F
        # The standard error on I pixelwise, in THIS regression (again, this is NOT self.I_se!)
        se_I = np.sqrt(np.mean(np.square(resid_I), axis=0))
        # The standard error on F pixelwise; update this on the model object
        self.F_se = se_I / np.sqrt(sum_xx / self.nr)

        # Do we need to apply an update?
        apply_update: bool = (eta > 0.0)

        # Save the updated optical factor if necessary.
        # Impose the constraint that the mean of F is 1.0 on good pixels
        # Do this update AFTER calculating the standard error on F to avoid conflating
        # the regression estimation error with this adjustment
        if apply_update:
            # The weighted average of the old and new estimates
            F = eta * F + (1.0 - eta) * self.F
            # Adjust F to have mean 1.0 on good pixels
            self.F = F / np.mean(F[self.B_is_ok])

    def shift_conc_logit(self, eta: float) -> None:
        """Shift the concentrations to match the mean SOC in logit space"""
        # Transform Y for [0, 1] to [-inf, inf] via the logit transform
        logit_Y: np.ndarray = logit(self.Y)
        # Small shift to logit_Y for numerical derivatives
        dlY: float = 1.0E-6
        # Calculate shifted values of Y via logit
        Y_dn: np.ndarray = expit(logit_Y - dlY)
        Y_up: np.ndarray = expit(logit_Y + dlY)

        # Calculate shifted values of SOC using logistic transform
        for r in range(self.nr):
            # Concentration for this reaction
            t: float = self.T[r]
            # The relative concentrations and state of charge with Y shifted down
            conc_dn: np.ndarray = y2conc(y=Y_dn[r], t=t)
            S_dn = conc_dn[:,1] + conc_dn[:,2]
            # The relative concentrations and state of charge with Y shifted up
            conc_up: np.ndarray = y2conc(y=Y_up[r], t=t)
            S_up = conc_up[:,1] + conc_up[:,2]
            # Numerical derivative of SOC with respect to logit Y
            dS_dlY: np.ndarray = (S_up - S_dn) / (2.0 * dlY)
            # Numerical derivative of mean SOC w.r.t. logit Y, averaged on good pixels
            dS_bar_dlY: np.ndarray = np.mean(dS_dlY[self.ok[r]])
            # Adjustment to logit Y to match mean SOC
            shift_lY: np.ndarray = (self.S_bar_data[r] - self.S_bar_pred[r]) / dS_bar_dlY
            # Revised values of Y on this reaction to match the observed mean SOC
            self.Y[r] = expit(logit_Y[r] + shift_lY * eta)
            
        # Delegate to predict_PIS() to make other variables consistent with shifted Y
        self.predict_PIS()

    def shift_conc_linear(self, eta: float) -> None:
        """Shift the concentrations to match the mean SOC in linear space"""
        # Small shift to Y for numerical derivatives
        dY: float = 1.0E-6
        # Allowed range of Y
        Y_min: float = 0.0
        Y_max: float = 1.0
        # Calculate shifted values of Y for numerical derivatives
        Y_dn: np.ndarray = np.clip(self.Y - dY, Y_min, Y_max)
        Y_up: np.ndarray = np.clip(self.Y + dY, Y_min, Y_max)

        # Calculate shifted values of SOC using logistic transform
        for r in range(self.nr):
            # Concentration for this reaction
            t: float = self.T[r]
            # The relative concentrations and state of charge with Y shifted down
            conc_dn: np.ndarray = y2conc(y=Y_dn[r], t=t)
            S_dn = conc_dn[:,1] + conc_dn[:,2]
            # The relative concentrations and state of charge with Y shifted up
            conc_up: np.ndarray = y2conc(y=Y_up[r], t=t)
            S_up = conc_up[:,1] + conc_up[:,2]
            # Numerical derivative of SOC with respect to Y
            dS_dY: np.ndarray = (S_up - S_dn) / (2.0 * dY)
            # Numerical derivative of mean SOC w.r.t. Y, averaged on good pixels
            dS_bar_dY: np.ndarray = np.mean(dS_dY[self.ok[r]])
            # Adjustment to Y to match mean SOC
            shift_Y: np.ndarray = (self.S_bar_data[r] - self.S_bar_pred[r]) / dS_bar_dY
            # Revised values of Y on this reaction to match the observed mean SOC
            self.Y[r] = np.clip(self.Y[r] + shift_Y * eta, Y_min, Y_max)
            
        # Delegate to predict_PIS() to make other variables consistent with shifted Y
        self.predict_PIS()

    def shift_conc(self, eta: float):
        """Shift the concentrations to match the mean SOC"""
        # Delegate to shift_conc_logit
        self.shift_conc_logit(eta=eta)

    def estimate_one_step(self, verbose: bool) -> None:
        """Perform one iteration of the estimation update algorithm"""
        # Save previous estimates of Y, F and I
        self.Y_prev = self.Y.copy()
        self.F_prev = self.F.copy()
        self.I_prev = self.I_pred.copy()
        # Save previous regression coefficients
        self.bcoeff_prev = self.bcoeff.copy()
        self.bconst_prev = self.bconst.copy()

        # Set learning rates
        eta_F: float = 0.50
        eta_bcoeff: float = 1.00
        eta_conc: float = 1.00

        # Update optical factor pixelwise
        self.calc_F(eta=eta_F)
        self.predict()
        if verbose:
            print('Shifted optical factors:')
            self.report_summary()

        # Shift concentrations to match S_bar
        for ci in range(self.ci_max):
            self.shift_conc(eta=eta_conc)
        if verbose:
            print('Shifted concentrations:')
            self.report_summary()

        # Update brightness coefficient pixelwise
        self.calc_bcoeff(eta=eta_bcoeff)
        self.predict()
        if verbose:
            print('Shifted brightness coefficients:')
            self.report_summary()

        # Shift concentrations to match S_bar
        for ci in range(self.ci_max):
            self.shift_conc(eta=eta_conc)
        if verbose:
            print('Shifted concentrations:')
            self.report_summary()

        # Calculate the RMS change Y and F over this iteration
        self.Y_chng = np.sqrt(np.mean(np.square(self.Y[self.ok] - self.Y_prev[self.ok])))
        self.F_chng = np.sqrt(np.mean(np.square(self.F[self.B_is_ok] - self.F_prev[self.B_is_ok])))
        # Calculate the RMS change in predicted I relative to the average brightness on each reaction
        delta_I_rel: np.ndarray = (self.I_pred[self.ok] - self.I_prev[self.ok]) / self.I_bar.reshape((self.nr, 1))
        self.I_chng = np.sqrt(np.mean(np.square(delta_I_rel)))
        # Relative change in brightness coefficients over this iteration
        chng_rel: np.ndarray = np.zeros(4)
        chng_rel[0:3] = (self.bcoeff - self.bcoeff_prev) / np.max(np.abs(self.bcoeff_prev))
        chng_rel[3] = (self.bconst - self.bconst_prev) / self.bconst_prev
        self.bc_chng = np.sqrt(np.mean(np.square(chng_rel)))

    def estimate(self, max_iter: int, thresh: float, verbose: bool):
        """
        Estimate the whole model by running the update algorithm for multiple iterations.
        INPUTS:
            max_iter:   maximum number of iterations
            thresh:     convergence threshold
            verbose:    control verbosit of output to console
        """
        # Enforce recovery of average state of charge
        for ci in range(self.ci_max):
            self.shift_conc(eta=1.0)
        # Error from initial guess with corrected SOC
        if verbose:
            print_stars()
            print('Model summary: initial correction to match mean SOC')
            self.report_summary()

        # Subfunction to report changes in parameters
        def print_chng():
            print(f'\nRMS change in model parameters:')
            print(f'Y    : {self.Y_chng:5.2e}')
            print(f'F    : {self.F_chng:5.2e}')
            print(f'I_rel: {self.I_chng:5.2e}')
            print(f'coeff: {self.bc_chng:5.2e}')            

        # Run the model for multiple iterations
        for i in range(max_iter):
            print_stars()
            print('Model summary: iteration', i+1)
            self.estimate_one_step(verbose=False)
            self.report_summary()

            # Terminate early if small enough change in parameters
            early_term: bool = (self.Y_chng < thresh) and (self.F_chng < thresh) and \
                (self.I_chng < thresh) and (self.bc_chng < thresh)
            if verbose or early_term:
                print_chng()
            if early_term:
                print('Terminating early due to small change in parameters.')
                break

    def get_conc(self, r: int) -> np.float64:
        """Get the concentration for reaction condition r"""
        return self.T[r]

    def get_conc_int(self, r: int) -> int:
        """Get the concentration for reaction condition r"""
        return int(self.get_conc(r))

    def get_pot(self, r: int) -> np.float64:
        """Get the applied potential for reaction condition r"""
        return self.V[r]

    def get_pot_int(self, r: int) -> int:
        """Get the applied potential for reaction condition r"""
        return int(self.get_pot(r))

    def get_ok(self, r: int):
        """Get mask where pixels are OK for reaction condition r as flattened array"""
        return self.ok[r] & self.B_is_ok

    def get_ok_2d(self, r: int):
        """Get mask where pixels are OK for reaction condition r in shape of the image"""
        return np.reshape(self.get_ok(r), self.shape_plot) 

    def get_mask_2d(self, r: int):
        """Get PixelMask object for reaction condition r"""
        ok: np.ndarray = self.get_ok_2d(r)
        lo: np.ndarray = self.B_is_lo.reshape(self.shape_plot)
        hi: np.ndarray = self.B_is_hi.reshape(self.shape_plot)
        return PixelMask(ok=ok, lo=lo, hi=hi)        

    def get_S(self, r: int)  -> np.ndarray:
        """Get SOC for reaction condition r in shape of a flat array"""
        return self.S[r]

    def get_S_2d(self, r: int)  -> np.ndarray:
        """Get SOC for reaction condition r in shape of the image"""
        return np.reshape(self.S[r], self.shape_plot)

    def plot_soc_one(self, r: int):
        """Generate one SOC plot"""
        S: np.ndarray = self.get_S_2d(r)
        mask: PixelMask = self.get_mask_2d(r)
        conc: int = self.get_conc_int(r)
        pot: int  = self.get_pot_int(r)
        plot_soc(S=S, mask=mask, conc=conc, pot=pot)

    def plot_soc_all(self):
        """Generate all the SOC plots"""
        for r in range(self.nr):
            self.plot_soc_one(r)

    def plot_soc_hist_one(self, r: int):
        """Generate SOC histogram for one reaction"""
        S: np.ndarray = self.get_S(r)
        ok: np.ndarray = self.get_ok(r)
        conc: int = self.get_conc_int(r)
        pot: int  = self.get_pot_int(r)
        plot_hist_soc_one(S=S, ok=ok, conc=conc, pot=pot)

    def plot_soc_hist_all(self):
        """Generate SOC histogram for one reaction"""
        for r in range(self.nr):
            self.plot_soc_hist_one(r)

    def plot_S_bar_scatter(self):
        """Generate scatter plot of mean SOC - data vs. predicted"""
        plot_S_bar_scatter(S_bar_data=self.S_bar_data, S_bar_pred=self.S_bar_pred)

    def plot_I_bar_scatter(self):
        """Generate scatter plot of mean intensity vs. predicted mean intensity"""
        # Mean image intensity for data
        I_vec: np.ndarray = np.zeros(self.nr)
        # Mean image intensity for model
        I_pred: np.ndarray = np.zeros(self.nr)
        # Loop over reaction conditions
        for r in range(self.nr):
            ok_r = self.ok[r]
            I_vec[r] = np.mean(self.I_data[r][ok_r])
            I_pred[r] = np.mean(self.I_pred[r][ok_r])
        # Build the scatter plot
        plot_I_model(I_vec=I_vec, I_pred=I_pred)

    def save_npy(self):
        """Save model output to .npy files"""
        # Alias output direcotory
        dir: Path = self.dir_npy
        # Make directory if missing
        dir.mkdir(parents=True, exist_ok=True)

        # Save the arrays to .npy files
        np.save(dir / 'T.npy',          self.T)
        np.save(dir / 'V.npy',          self.V)
        np.save(dir / 'S_bar_data.npy', self.S_bar_data)
        np.save(dir / 'B_data.npy',     self.B_data)
        np.save(dir / 'I_data.npy',     self.I_data)
        np.save(dir / 'B_is_ok.npy',    self.B_is_ok)
        np.save(dir / 'B_is_lo.npy',    self.B_is_lo)
        np.save(dir / 'B_is_lo.npy',    self.B_is_hi)
        np.save(dir / 'ok.npy',         self.ok)
        np.save(dir / 'bcoeff.npy',     self.bcoeff)
        np.save(dir / 'bconst.npy',     self.bconst)
        np.save(dir / 'conc.npy',       self.conc)
        np.save(dir / 'Y.npy',          self.Y)
        np.save(dir / 'S.npy',          self.S)
        np.save(dir / 'P.npy',          self.P)
        np.save(dir / 'F.npy',          self.F)
        np.save(dir / 'I_pred.npy',     self.I_pred)
        np.save(dir / 'S_bar_pred.npy', self.S_bar_pred)

    def load_npy(self):
        """Load tunable parameters from .npy files found in input directory"""
        dir: Path = self.dir_npy
        print(f'Loading model data in .npy files in directory {dir}.')
        self.F = np.load(dir / "F.npy")
        self.Y = np.load(dir / "Y.npy")
        self.bcoeff = np.load(dir / "bcoeff.npy")
        self.bconst = np.load(dir / "bconst.npy")
        # Predict SOC and image intensity from current parameters
        self.predict_PIS()
        # Calculate standard errors on I and F
        self.calc_bcoeff(eta=0.0)
        self.calc_F(eta=0.0)
        # Calculate the error bounds
        self.predict_bounds()

    def save_mat(self):
        """Save model output to MatLab .mat files"""

        # Shape of images when saving to disk
        shape_plot = (self.m, self.n)
        shape_imgs = (self.nc, self.nv, self.m, self.n)
        shape_conc = (self.nc, self.nv, self.m, self.n, 3)

        # The output directory
        out_dir: Path = Path('calcs')
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create a dictionary of arrays to save
        mdict: dict[str, np.ndarray] = {
            'T':            self.T.reshape(self.shape_react),
            'V':            self.V.reshape(self.shape_react),
            'S_bar_data':   self.S_bar_data.reshape(self.shape_react),
            'B_data':       self.B_data.reshape(shape_plot),
            'I_data':       self.I_data.reshape(shape_imgs),
            'B_is_ok':      self.B_is_ok.reshape(shape_plot),
            'B_is_lo':      self.B_is_lo.reshape(shape_plot),
            'B_is_hi':      self.B_is_hi.reshape(shape_plot),
            'ok':           self.ok.reshape(shape_imgs),
            'conc':         self.conc.reshape(shape_conc),
            'S':            self.S.reshape(shape_imgs),
            'P':            self.P.reshape(shape_imgs),
            'F':            self.F.reshape(shape_plot),
            'I_pred':       self.I_pred.reshape(shape_imgs),
            'S_bar_pred':   self.S_bar_pred.reshape(self.shape_react),
        }

        # Save the dictionary to a .mat file
        file_name: str = Path('calcs', 'model.mat')
        scipy.io.savemat(file_name=file_name, mdict=mdict, format='5', 
                         do_compression=False, oned_as='column')

    def report_summary(self):
        """Report summary of error estimates"""

        # Error from initial guess
        print('RMS error for recovered image intensity and mean SOC:')
        print(f'I_abs : {self.calc_err_I_abs():8.6f}')
        print(f'I_rel : {self.calc_err_I_rel():8.6f}')
        print(f'S_bar : {self.calc_err_S_bar():8.6f}')

        # Standard errors for image intensity and optical factor
        # print('\nRMS standard error on regression parameters:')
        # print(f'I_abs: {self.I_se:8.6f}')
        # print(f'I_rel: {self.I_rel_se:8.6f}')
        # print(f'F    : {self.calc_avg_se_F():8.6f}')

        # Adjustment in optical factor
        F_adj_abs: np.ndarray = self.F - self.F0
        F_adj_rel: np.ndarra = F_adj_abs / self.F0
        rms_abs: float = np.sqrt(np.mean(np.square(F_adj_abs[self.B_is_ok])))
        rms_rel: float = np.sqrt(np.mean(np.square(F_adj_rel[self.B_is_ok])))
        print(f'RMS Adjustment in optical factor: {rms_abs:8.6f} abs / {rms_rel:8.6f} rel')

        # Summary statistics for estimated state of charge
        soc_mean_vec: float = np.mean(self.S[self.ok], axis=0)
        soc_mean: float = np.mean(soc_mean_vec)
        soc_se_rms: float = np.sqrt(np.mean(np.square(self.calc_avg_se_S())))
        soc_se_rms_I: float = np.sqrt(np.mean(np.square(self.calc_avg_se_S_I())))
        soc_se_rms_F: float = np.sqrt(np.mean(np.square(self.calc_avg_se_S_F())))
        soc_se_rel: float = soc_se_rms / soc_mean

        print('\nEstimated SOC summary:')
        print(f'Mean              : {soc_mean:8.6f}')
        print(f'Std Err (Absolute): {soc_se_rms:8.6f}')
        print(f'- Uncertainty in I: {soc_se_rms_I:8.6f}')
        print(f'- Uncertainty in F: {soc_se_rms_F:8.6f}')
        print(f'Std Err (Relative): {soc_se_rel:8.6f}')

    def report_coeff(self):
        """Report brightness coefficients"""
        print('Brightness Coefficients - light production vs. concentration')
        print(f'alpha   = {self.bcoeff[0]:9.3f}')
        print(f'beta    = {self.bcoeff[1]:9.3f}')
        print(f'gamma   = {self.bcoeff[2]:9.3f}')
        print(f'delta   = {self.bconst:9.3f}')
        print(f'R2      = {self.I_r2:9.6f}')
        print(f'Std Err = {self.I_se:9.6f}')

# **************************************************************************************************
def make_image_model(im: ImageData, ec: EchemData):
    """Factory function to build one instance of an ImageModel"""
    # Get the sizes of the image data
    nc: int = ec.nc
    nv: int = ec.nv
    m: int = im.m
    n: int = im.n
    
    # Build the empty ImageModel
    m: ImageModel = ImageModel(nc=nc, nv=nv, m=m, n=n)
    
    # Load the image data
    m.load_data(im=im, ec=ec)
    # Set initial parameter values
    m.set_initial_params()

    return m
