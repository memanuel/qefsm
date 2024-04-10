import numpy as np
from scipy.interpolate import PchipInterpolator

# Module constant: equilibrium constant in mM^(-1) for the relationship
# [Dimer] = K * [AQDS] * [H2-AQDS]
K: float = 0.080

# **************************************************************************************************
def check_conc(conc: np.ndarray, t: float):
    """
    Check that relative concentrations satisfy the two equations
    INPUTS:
        conc:   Relative concentrations; shape (n, 3); dimension 1 holds [x, y, z]
        t:      The total concentration of AQDS; in mM
    RETURNS:
        err:    Array of shape (n, 2); each column has the error from an equation
                column 0: Errors from equation for dimer concentration, z = R * x * y
                column 1: Errors from equation for total AQDS, x + y + 2z = 1.0
    """
    # Unpack concentration array
    x: np.ndarray = conc[:, 0]
    y: np.ndarray = conc[:, 1]
    z: np.ndarray = conc[:, 2]
    
    # Number of data points
    n: int = x.size
    # Initizlize array to store errors
    err: np.ndarray = np.zeros((n, 2))

    # The dimensionless rate constant
    R: float = K * t

    # Expected concentration of z from equilibrium
    z_eq: np.ndarray = R * x * y    
    # Error in z
    err[:, 0] = np.abs(z - z_eq) 

    # Total relative concentration of AQDS; should be 1.0
    t_rel: np.ndarray = x + y + z * 2.0

    # Error in total concentration
    err[:, 1] = np.abs(t_rel - 1.0)

    return err

# **************************************************************************************************
def x2conc(x: np.ndarray, t: float):
    """
    Calculate the concentration of all three species from the relative concentration
    of oxidized species
    INPUTS:
        x:      The relative concentration of oxidized species; in [0, 1]
        t:      The total concentration of AQDS; in mM
    RETURNS:
        conc:   Relative concentrations; shape (n, 3); dimension 1 holds [x, y, z]
    """
    # The dimensionless rate constant
    R: float = K * t
    # The denominator shared by y and z
    q: np.ndarray = 1.0 + 2.0 * R * x
    
    # Shape of the concentrations
    shape_conc: tuple[int, int] = (x.size, 3,)
    # Initialize array for the answer
    conc: np.ndarray = np.zeros(shape_conc)
    # The concentration of oxidized species, x, was an input
    conc[:, 0] = x
    # The concentration of reduced species, y
    conc[:, 1] = (1.0 - x) / q
    # The concentration of the dimer, z
    conc[:, 2] = R * x * (1.0 - x) / q

    return conc

# **************************************************************************************************
def y2conc(y: np.ndarray, t: float):
    """
    Calculate the relative concentration of all three species from the relative concentration
    of reduced species
    INPUTS:
        y:      The relative concentration of reduced species; in [0, 1]
        t:      The total concentration of AQDS; in mM
    RETURNS:
        conc:   Relative concentrations; shape (n, 3); dimension 1 holds [x, y, z]
    """
    # The dimensionless rate constant
    R: float = K * t
    # The denominator shared by y and z
    q: np.ndarray = 1.0 + 2.0 * R * y
    
    # Shape of the concentrations
    shape_conc: tuple[int, int] = (y.size, 3,)
    # Initialize array for the answer
    conc: np.ndarray = np.zeros(shape_conc)
    # The concentration of oxidized species, x
    conc[:, 0] = (1.0 - y) / q
    # The concentration of reduced species, y, was an input
    conc[:, 1] = y
    # The concentration of the dimer, z
    conc[:, 2] = R * y * (1.0 - y) / q

    return conc

# **************************************************************************************************
def conc2s(conc: np.ndarray) -> np.ndarray:
    """
    Calculate the state of charge from a concentration array
    INPUTS:
        conc:   Relative concentrations; shape (n, 3); dimension 1 holds [x, y, z]
    RETURNS:
        s:      Array of state of charge; shape (n,)
    """
    # Unpack concentration array
    y: np.ndarray = conc[:, 1]
    z: np.ndarray = conc[:, 2]
    # Recovered state of charge
    s: np.ndarray = y + z
    return s

# **************************************************************************************************
def build_spline_s2y(t: float) -> PchipInterpolator:
    """
    Build a spline of state of charge s vs. reduced conctentration y
    INPUTS:
        t:      The total concentration of AQDS; in mM
    RETURNS:
        func:   Monotonic interpolation function, where y = func(s)
    """
    # Input concentration of reduced species
    dy: float = 1.0 / 65536.0
    y: np.ndarray = np.arange(0.0, 1.0 + dy, dy)
    # Recovered concentrations starting from y
    conc: np.ndarray = y2conc(y=y, t=t)
    # The state of charge for these concentrations - sum of columns 1 (y) and 2 (z)
    s: np.ndarray = np.sum(conc[:, 1:3], axis=1)
    # Build a monotonic cubic spline; input = s, output = y
    return PchipInterpolator(s, y, extrapolate=False)

# **************************************************************************************************
def build_spline_y2s(t: float) -> PchipInterpolator:
    """
    Build a spline of reduced conctentration y vs. state of charge s
    INPUTS:
        t:      The total concentration of AQDS; in mM
    RETURNS:
        func:   Monotonic interpolation function, where s = func(y)
    """
    # Input concentration of reduced species
    dy: float = 1.0 / 65536.0
    y: np.ndarray = np.arange(0.0, 1.0 + dy, dy)
    # Recovered concentrations starting from y
    conc: np.ndarray = y2conc(y=y, t=t)
    # The state of charge for these concentrations - sum of columns 1 (y) and 2 (z)
    s: np.ndarray = np.sum(conc[:, 1:3], axis=1)
    # Build a monotonic cubic spline; input = y, output = s
    return PchipInterpolator(y, s, extrapolate=False)

# **************************************************************************************************
# Table of interpolators for mapping s to y
func_tbl_s2y: dict[float, PchipInterpolator] = dict()
# Table of interpolators for mapping y to s
func_tbl_y2s: dict[float, PchipInterpolator] = dict()
# Populate both tables of interpolators
for t in [10.0, 20.0, 30.0, 40.0]:
    func_tbl_s2y[t] = build_spline_s2y(t=t)
    func_tbl_y2s[t] = build_spline_y2s(t=t)

# **************************************************************************************************
def s2y(s: np.ndarray, t: float) -> np.ndarray:
    """
    Calculate the concentration of reduced species from the state of charge
    INPUTS:
        s:      The desired state of charge; shape (n,); in range [0, 1]
        t:      The total concentration of AQDS; in mM
    RETURNS:
        y:      Array of relative concentration of reduced species; in range [0, 1]
    """
    # Load relevant interpolator
    func: PchipInterpolator = func_tbl_s2y[t]
    # Apply interpolation function
    y: np.ndarray = func(s)
    return y

# **************************************************************************************************
def y2s(y: np.ndarray, t: float) -> np.ndarray:
    """
    Calculate the concentration of reduced species from the state of charge
    INPUTS:
        y:      The input fraction of reduced species; in range [0, 1]
        t:      The total concentration of AQDS; in mM
    RETURNS:
        s:      The state of charge; in range [0, 1]
    """
    # Load relevant interpolator
    func: PchipInterpolator = func_tbl_y2s[t]
    # Apply interpolation function
    s: np.ndarray = func(y)
    return s

# **************************************************************************************************
def s2conc(s: np.ndarray, t: float) -> np.ndarray:
    """
    Calculate the relative concentration of all three species from the state of charge
    INPUTS:
        s:      The desired state of charge; shape (n,); in range [0, 1]
        t:      The total concentration of AQDS; in mM
    RETURNS:
        conc:   Relative concentrations; shape (n, 3); dimension 1 holds [x, y, z]
    """
    # Calculate the relative concentration of reduced species
    y: np.ndarray = s2y(s=s, t=t)
    # Calculate full relative concentration array
    return y2conc(y=y, t=t)

# **************************************************************************************************
def s2conc_abs(s: np.ndarray, t: float) -> np.ndarray:
    """
    Calculate the absolute concentration of all three species from the state of charge
    INPUTS:
        s:      The desired state of charge; shape (n,); in range [0, 1]
        t:      The total concentration of AQDS; in mM
    RETURNS:
        conc:   Absolute concentrations; shape (n, 3); dimension 1 holds [x, y, z]
    """
    # Calculate the relative concentration of reduced species
    y: np.ndarray = s2y(s=s, t=t)
    # Calculate full relative concentration array
    conc: np.ndarray = y2conc(y=y, t=t)
    return conc * t

# **************************************************************************************************
def build_spline_y2p(reg_coef: np.ndarray, t: float):
    """
    Build a spline to predict y from light production p for a given concentration
    INPUTS:
        reg_coef:   Linear regression coefficients [alpha, beta, gamma, delta]
                    for brightness vs. [X, Y, Z, 1]
        t:          The total concentration
    """
    # Input concentration of reduced species
    dy: float = 1.0 / 65536.0
    y: np.ndarray = np.arange(0.0, 1.0 + dy, dy)
    # Recovered concentrations starting from y
    conc: np.ndarray = y2conc(y=y, t=t)
    # The predicted light production from the concentrations
    p: np.ndarray = np.dot(conc, reg_coef[0:3])*t + reg_coef[3]
    # Build a monotonic cubic spline; input = y, output = p
    return PchipInterpolator(y, p, extrapolate=False)

# **************************************************************************************************
def build_spline_p2y(reg_coef: np.ndarray, t: float):
    """
    Build a spline to predict y from light production for a given concentration
    INPUTS:
        reg_coef:   Linear regression coefficients [alpha, beta, gamma, delta]
                    for brightness vs. [X, Y, Z, 1]
        t:          The total concentration
    """
    # Input concentration of reduced species
    dy: float = 1.0 / 65536.0
    y: np.ndarray = np.arange(0.0, 1.0 + dy, dy)
    # Recovered concentrations starting from y
    conc: np.ndarray = y2conc(y=y, t=t)
    # The predicted light production from the concentrations
    p: np.ndarray = np.dot(conc, reg_coef[0:3])*t + reg_coef[3]
    # Build a monotonic cubic spline; input = i, output = y
    return PchipInterpolator(p, y, extrapolate=False)

# **************************************************************************************************
def make_splines_y2p(reg_coef: np.ndarray) -> dict[float, PchipInterpolator]:
    """
    Construct a table of interpolators to predict reduced relative conc. from image intensity
    INPUTS:
        reg_coef:   Linear regression coefficients [alpha, beta, gamma, delta]
                    for brightness vs. [X, Y, Z, 1]
    """
    # Table of interpolators for mapping y to light production p
    func_tbl_y2p: dict[float, PchipInterpolator] = dict()
    # Populate table of interpolators
    for t in [10.0, 20.0, 30.0, 40.0]:
        func_tbl_y2p[t] = build_spline_y2p(reg_coef=reg_coef, t=t)

    return func_tbl_y2p

# **************************************************************************************************
def make_splines_p2y(reg_coef: np.ndarray) -> dict[float, PchipInterpolator]:
    """
    Construct a table of interpolators to predict light production from reduced relative conc.
    INPUTS:
        reg_coef:   Linear regression coefficients [alpha, beta, gamma, delta]
                    for brightness vs. [X, Y, Z, 1]
    """
    # Table of interpolators for mapping adjusted image intensity to y
    func_tbl_p2y: dict[float, PchipInterpolator] = dict()
    # Populate table of interpolators
    for t in [10.0, 20.0, 30.0, 40.0]:
        func_tbl_p2y[t] = build_spline_p2y(reg_coef=reg_coef, t=t)

    return func_tbl_p2y

# **************************************************************************************************
def test_calc_conc():
    """Test the function calc_conc"""

    # Total concentration
    t: float = 20.0

    # Input concentration of oxidized species
    x_in: np.ndarray = np.linspace(0.0, 1.0, 1001)
    # Recovered concentrations starting from x
    conc_x: np.ndarray = x2conc(x=x_in, t=t)
    # Error in concentrations from x
    err_x: np.ndarray = check_conc(conc=conc_x, t=t)
    # Report maximum error
    print(f'Maximum error in x2conc = [{np.max(err_x[:,0]):5.2e}, {np.max(err_x[:,1]):5.2e}]')

    # Input concentration of reduced species
    y_in: np.ndarray = np.linspace(0.0, 1.0, 1001)
    # Recovered concentrations starting from y
    conc_y: np.ndarray = y2conc(y=y_in, t=t)
    # Error in concentrations from y
    err_y: np.ndarray = check_conc(conc=conc_y, t=t)
    # Report maximum error
    print(f'Maximum error in y2conc = [{np.max(err_y[:,0]):5.2e}, {np.max(err_y[:,1]):5.2e}]')

# **************************************************************************************************
def test_s2y():
    """Test the function s2y"""

    # Total concentration
    t: float = 20.0

    # Input state of charge
    s_in: np.ndarray = np.linspace(0.0, 1.0, 1001)
    # Predicted concentration of reduced species
    y_out: np.ndarray = s2y(s=s_in, t=t)
    # Recovered concentration of all three species
    conc_out: np.ndarray = y2conc(y=y_out, t=t)
    # Recovered state of charge
    s_out: np.ndarray = conc2s(conc=conc_out)
    # Check the recovered state of charge from the concentrations
    err: np.ndarray = np.abs(s_out - s_in)
    # Report maximum error
    print(f'Maximum error in s2y = {np.max(err):5.2e}')

# **************************************************************************************************
def test_y2s():
    """Test the function y2s"""

    # Total concentration
    t: float = 20.0

    # Input concentration of reduced species
    y_in: np.ndarray = np.linspace(0.0, 1.0, 1001)
    # Recovered concentration of all three species consistent with inputs
    conc_in: np.ndarray = y2conc(y=y_in, t=t)
    # Input state of charge
    s_in: np.ndarray = conc2s(conc=conc_in)
    # Predicted state of charge
    s_out: np.ndarray = y2s(y=y_in, t=t)
    # Check the recovered state of charge from the concentrations
    err: np.ndarray = np.abs(s_out - s_in)
    # Report maximum error
    print(f'Maximum error in y2s = {np.max(err):5.2e}')

# **************************************************************************************************
def main():
    """Test some functions"""
    test_calc_conc()
    test_s2y()
    test_y2s()

# **************************************************************************************************
if __name__ == '__main__':
    main()
