from collections import Counter
import numpy as np
import pickle
from sklearn.linear_model import Lasso
import pysindy as ps
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from itertools import chain
import warnings
warnings.filterwarnings('ignore')
# arial font
import matplotlib.font_manager as fm
my_font = fm.FontProperties(fname='arial.ttc')

degrees=10

# Functions needed
def trans(x,N):
    '''
    Convert a decimal number to its binary representation.

    Parameters
    ----------
    x : int or float
        The number to be converted.
    N : int
        The desired length (number of bits) for the binary representation.

    Returns
    -------
    w : numpy.ndarray
        A 2D binary vector (row vector) of length N representing the number x.
    '''

    y=np.copy(x)
    if y == 0: return[0]
    bit = []
    for i in np.arange(N):
        bit.append(y % 2)
        y >>= 1
    return np.atleast_2d(np.asarray(bit[::-1]))

#go from binary representation to decimal number
def trans_rev(x):

    '''
    Convert a binary vector representation to its decimal number.

    Parameters
    ----------
    x : numpy.ndarray or list
        The binary vector (list of 0s and 1s).

    Returns
    -------
    dec : int
        The resulting decimal number.
    '''


    n = len(x)-1
    dec = 0
    for i in np.arange(n+1):
        dec = dec + x[i]*2**(n-i)
    return dec

def f_line(x, a, b):
    '''
    Defines a straight line function: $f(x) = a*x + b$.

    Parameters
    ----------
    x : numpy.ndarray or float
        The independent variable.
    a : float
        The slope of the line.
    b : float
        The y-intercept.

    Returns
    -------
    float or numpy.ndarray
        The calculated value(s) of the function.
    '''
    return a*x + b

def f_poly2(t,a,b,c):
    '''
    Defines a second-degree polynomial function: $f(t) = a*t^2 + b*t + c$.

    Parameters
    ----------
    t : numpy.ndarray or float
        The independent variable.
    a : float
        Coefficient for $t^2$.
    b : float
        Coefficient for $t$.
    c : float
        The constant term.

    Returns
    -------
    float or numpy.ndarray
        The calculated value(s) of the function.
    '''
    return a*pow(t,2) + b*t + c

def f_poly3(t,a,b,c,d):
    '''
    Defines a third-degree polynomial function: $f(t) = a*t^3 + b*t^2 + c*t + d$.

    Parameters
    ----------
    t : numpy.ndarray or float
        The independent variable.
    a : float
        Coefficient for $t^3$.
    b : float
        Coefficient for $t^2$.
    c : float
        Coefficient for $t$.
    d : float
        The constant term.

    Returns
    -------
    float or numpy.ndarray
        The calculated value(s) of the function.
    '''
    return a*pow(t,3) + b*pow(t,2) + c*t + d

def flatten_chain(matrix):
    '''
    Flatten a matrix (list of lists) into a single list using itertools.chain.

    Parameters
    ----------
    matrix : list of list
        A list of lists to be flattened.

    Returns
    -------
    list
        A single, flattened list containing all elements of the input matrix.
    '''
    return list(chain.from_iterable(matrix))

def format_rp_rd(rp,rd):
    '''
    Format the values of $r_p$ (proliferation rate) and $r_d$ (death rate, assumed $r_d = r_p / 2$)
    by rounding them to integers or a specific number of decimal places.

    Parameters
    ----------
    rp : float
        The proliferation rate, $r_p$.
    rd : float
        The death rate, $r_d$.

    Returns
    -------
    rp_formatted : int or float
        The formatted proliferation rate (integer if whole, rounded to 2 decimals otherwise).
    rd_formatted : int or float
        The formatted death rate (integer if whole, rounded to 3 decimals otherwise).
    '''
    if rp==int(rp):
        rp = int(rp)
    else:
        rp = round(rp,2)
    rd = rp/2;
    if rd==int(rd):
        rd = int(rd)
    else:
        rd = round(rd,3)
    return rp,rd

def MSE(a,b):
    '''
    Compute the Mean Squared Error (MSE) between two NumPy arrays.

    Parameters
    ----------
    a : numpy.ndarray
        The first array (e.g., predicted values).
    b : numpy.ndarray
        The second array (e.g., true values).

    Returns
    -------
    float
        The Mean Squared Error. Returns np.nan if array shapes do not match.
    '''
    if a.shape != b.shape:
        return(np.nan)
    return ((a - b)**2).mean()

def simulate_meanfield_model(t, C0, xi):
    '''
    Simulate the mean-field model using `scipy.integrate.odeint`.

    Parameters
    ----------
    t : numpy.ndarray
        Time points at which to solve the ODE.
    C0 : float
        Initial condition for the concentration variable.
    xi : tuple or list
        Parameters for the mean-field model's RHS function, $xi = (xi_0, xi_1)$.

    Returns
    -------
    numpy.ndarray
        The solved concentration values at time points `t`.
    '''
    sol = odeint(meanfield_RHS, C0, t, args=(xi,))
    return sol

def meanfield_RHS(u,t,xi):
    '''
    Right-Hand Side (RHS) function for the mean-field Ordinary Differential Equation (ODE).

    The ODE is $\frac{du}{dt} = xi_0 u(1-u) - xi_1 u$.

    Parameters
    ----------
    u : float
        The state variable (concentration).
    t : float
        The time variable (unused in this specific RHS form).
    xi : tuple or list
        Parameters $xi_0$ and $xi_1$.

    Returns
    -------
    float
        The value of the derivative, $\frac{du}{dt}$.
    '''
    dudt = xi[0]*u*(1-u) - xi[1]*u
    return dudt

#ODE RHS for BDM model
def BDM_RHS(t, x, coefs, deg):
    '''
    Right-Hand Side (RHS) function for learned DE for the BDM  model.
    Designed to work with `scipy.integrate.solve_ivp`.
    This version uses the degree of the polynomial library to construct the features.

    Parameters
    ----------
    t : float
        The time variable.
    x : numpy.ndarray
        The state variable (e.g., concentration). Expects a 1D array, $x=[C]$.
    coefs : numpy.ndarray
        The learned coefficient matrix from SINDy. Expects shape (1, M) where M is the number of features.
    deg : int
        The maximum degree of the polynomial features.

    Returns
    -------
    numpy.ndarray
        The value of the derivative, $\frac{dC}{dt}$.
    '''
    X = np.array([x[0]**p for p in np.arange(1,deg+1)]).T
    return np.matmul(X,coefs[0])

def BDM_RHS_eval(t, x, coefs,deg):
    '''
    Right-Hand Side (RHS) function for BDM model evaluation.
    Similar to BDM_RHS but expects a flattened coefficient array.

    Parameters
    ----------
    t : float
        The time variable.
    x : numpy.ndarray
        The state variable (e.g., concentration). Expects a 1D array, $x=[C]$.
    coefs : numpy.ndarray
        The learned coefficient vector. Expects shape (M,) where M is the number of features.
    deg : int
        The maximum degree of the polynomial features.

    Returns
    -------
    numpy.ndarray
        The value of the derivative, $\frac{dC}{dt}$.
    '''
    X = np.array([x[0]**p for p in np.arange(1,deg+1)]).T
    return np.matmul(X,coefs) # coefs instead of coefs[0]

def BDM_RHS_structure(t, x, coefs, deg):
    '''
    Right-Hand Side (RHS) function for the BDM model, specifically
    using a list of *active* degrees to construct the features.

    Parameters
    ----------
    t : float
        The time variable.
    x : numpy.ndarray
        The state variable (e.g., concentration). Expects a 1D array, $x=[C]$.
    coefs : numpy.ndarray
        The active learned coefficient vector.
    deg : list of int
        A list of degrees (exponents minus one) corresponding to the terms present in the model.
        e.g., [0, 1] corresponds to $C^{0+1} = C^1$ and $C^{1+1} = C^2$.

    Returns
    -------
    numpy.ndarray
        The value of the derivative, $\frac{dC}{dt}$.
    '''
    X = np.array([x[0]**(p+1) for p in deg]).T
    return np.matmul(X,coefs)

# Forward solve the learned ODE sindy models
def BDM_solve_ODE(coeffs,degrees,rp,rd,C0):
    '''
    Forward solve the BDM ODE using the SINDy-learned coefficients.

    Parameters
    ----------
    coeffs : numpy.ndarray
        The learned coefficient vector for the active terms.
    degrees : list of int
        List of degrees (exponents minus one) corresponding to the terms present in the model.
    rp : float
        Proliferation rate $r_p$. Used to define the end time $t_f = 20/(r_p - r_d)$.
    rd : float
        Death rate $r_d$.
    C0 : numpy.ndarray
        Initial condition, $C_0$.

    Returns
    -------
    numpy.ndarray
        The solved concentration values $C(t)$ over time.
    '''
    tf             = 20/(rp-rd)
    t_solve        = np.linspace(0, tf, 100)
    t_solve_span = (t_solve[0], t_solve[-1])
    u0_solve       = C0 #ABM[0], initial condition
    u              = solve_ivp(BDM_RHS_structure, t_solve_span, u0_solve, t_eval=t_solve,
                                 args = (coeffs,degrees)).y.T
    return(u)


# Find the least squares difference between the solved ODE and the data
def BDM_LSQ_wdata(coeffs,degrees,rp,rd,C0,data):
    '''
    Calculate the Least Squares Error (LSE) between the forward-solved BDM ODE
    and the given data. Used for parameter fitting/optimization.

    Parameters
    ----------
    coeffs : numpy.ndarray
        The learned coefficient vector for the active terms.
    degrees : list of int
        List of degrees (exponents minus one) corresponding to the terms present in the model.
    rp : float
        Proliferation rate $r_p$.
    rd : float
        Death rate $r_d$.
    C0 : numpy.ndarray
        Initial condition.
    data : numpy.ndarray
        The true data to compare the ODE solution against.

    Returns
    -------
    float
        The Least Squares Error (LSE). Returns a large value (10^7) if ODE solving fails.
    '''
    u   = BDM_solve_ODE(coeffs,degrees,rp,rd,C0)
    if len(u)<100:
        return(10**7)
    lsq = LSE(u,data)
    return(lsq)

# Least squares error between data and ODE
def LSE(a,b):
    '''
    Compute the Least Squares Error (LSE), which is the sum of squared differences,
    between two NumPy arrays.

    Parameters
    ----------
    a : numpy.ndarray
        The first array (e.g., predicted values).
    b : numpy.ndarray
        The second array (e.g., true values).

    Returns
    -------
    float
        The Least Squares Error $\sum (a_i - b_i)^2$.

    Raises
    ------
    AssertionError
        If the shapes of arrays `a` and `b` do not match.
    '''
    assert a.shape == b.shape
    return np.sum((a - b)**2)

def model_training_CV(optimizer,sindy_library,t_train_list,t_test_list,ABM_train_list,ABM_test_list,ABM_t_train_list,ABM_t_test_list,rp_):
    '''
    Perform cross-validated SINDy model training, forward-solve the resulting ODE,
    and compute the Akaike Information Criterion (AIC) score on the test data.

    Parameters
    ----------
    optimizer : ps.SINDyOptimizer
        An instantiated SINDy optimizer (e.g., `ps.STLSQ`).
    sindy_library : ps.feature_library
        An instantiated feature library (e.g., `ps.PolynomialLibrary`).
    t_train_list : list of numpy.ndarray
        List of training time arrays for each cross-validation split.
    t_test_list : list of numpy.ndarray
        List of testing time arrays for each cross-validation split.
    ABM_train_list : list of numpy.ndarray
        List of training data arrays (states, $C$) for each split.
    ABM_test_list : list of numpy.ndarray
        List of testing data arrays (states, $C$) for each split.
    ABM_t_train_list : list of numpy.ndarray
        List of training derivative arrays ($\\dot{C}$) for each split.
    ABM_t_test_list : list of numpy.ndarray
        List of testing derivative arrays ($\\dot{C}$) for each split.
    rp_ : float
        The proliferation rate $r_p$ used in the forward-solving time span calculation.

    Returns
    -------
    sindy_aic_list : list of float
        List of AIC scores computed on the test data for each split.
    sindy_model_coeffs_list : list of numpy.ndarray
        List of the learned SINDy model coefficient matrices for each split.
    '''
    sindy_aic_list = []
    sindy_model_coeffs_list = []
    for kk in range(len(ABM_train_list)):
      ABM_train = ABM_train_list[kk]
      ABM_test = ABM_test_list[kk]
      ABM_t_train = ABM_t_train_list[kk]
      ABM_t_test = ABM_t_test_list[kk]
      t_train = t_train_list[kk]
      t_test = t_test_list[kk]

      sindy_model = ps.SINDy(feature_library=sindy_library,optimizer=optimizer)
      sindy_model.fit(ABM_train, x_dot=ABM_t_train)
      sindy_model_coeffs = sindy_model.coefficients()

      #forward solve the learned sindy model
      rp=rp_
      rd = rp/2
      tf = 20/(rp-rd)
      t_solve = np.linspace(0, tf, 100)
      t_solve_span = (t_solve[0], t_solve[-1])
      u0_solve = ABM_train[0]
      u_pred_sindy = solve_ivp(
         BDM_RHS, t_solve_span, u0_solve, t_eval=t_solve, args = (sindy_model.coefficients(),degrees)
         ).y.T
      #1d interpolation to predict the test data
      sindy_interp = interp1d(t_solve, u_pred_sindy[:,0],axis=0)
      sindy_pred_test = sindy_interp(t_test)#[:,0]
      #compute the AIC score on the test data
      sindy_sse = np.dot((sindy_pred_test[:,0]-ABM_test[:,0]),(sindy_pred_test[:,0]-ABM_test[:,0]))
      sindy_num_params = np.sum(sindy_model.coefficients()[0]!=0)
      sindy_aic = ABM_test.shape[0]*np.log(sindy_sse/ABM_test.shape[0]) + 2*sindy_num_params

      sort_train = np.argsort(t_train,axis=0)
      ABM_train = ABM_train[sort_train,0]
      ABM_t_train = ABM_t_train[sort_train,0]
      t_train = t_train[sort_train,0]

      sindy_aic_list.append(sindy_aic)
      sindy_model_coeffs_list.append(sindy_model_coeffs)

    return sindy_aic_list, sindy_model_coeffs_list

def find_opt_with_threshold(aic_list,old_opt,max_param_size,coeff_list,CV_nums):
    '''
    Find an optimal model index based on a threshold applied to the absolute
    value of the model coefficients across multiple cross-validation (CV) splits.

    This function iterates through models (indexed by 'old_opt') and checks if
    any coefficient in the model exceeds a maximum threshold size in any CV split.

    Parameters
    ----------
    aic_list : list of float
        List of AIC scores (unused in the current implementation logic).
    old_opt : int
        The starting index (or previous optimal index) to begin the search.
    max_param_size : float
        The maximum allowed absolute value for a model coefficient.
    coeff_list : list of list of numpy.ndarray
        A nested list of coefficient matrices: `coeff_list[model_index][cv_split][0]`.
    CV_nums : int
        The total number of cross-validation splits.

    Returns
    -------
    int
        The index of the first model (starting from `old_opt`) that *does not*
        have any coefficient exceeding `max_param_size` across all CV splits.
    '''
    for kk in range(old_opt,len(aic_list)+1):
      num_past_thresh = 0
      for jj in range(CV_nums):
        sindy_model_coeffs = coeff_list[old_opt][jj][0]
        if np.sum(np.abs(sindy_model_coeffs)>max_param_size)>0:
          num_past_thresh+=1
      if num_past_thresh>0:
        old_opt = kk+1
      else:
        break
    return old_opt

def BDM_u_t(u,coefs,deg):
    '''
    Evaluate the BDM ODE right-hand side, $\frac{du}{dt}$, as a polynomial function of $u$.

    The RHS is $\sum_{k=1}^{deg} \text{coefs}[k-1] \cdot u^k$.

    Parameters
    ----------
    u : float
        The state variable (concentration).
    coefs : numpy.ndarray
        The learned coefficient vector.
    deg : int
        The maximum degree of the polynomial features.

    Returns
    -------
    float
        The value of the derivative, $\frac{du}{dt}$.
    '''
    rhs = 0.0
    for kk in range(deg):
      rhs = rhs + coefs[kk]*u**(kk+1)
    return rhs

def convert_array_to_array_label(array):
    '''
    Convert a list of feature indices (degrees minus one) into a LaTeX-formatted
    string label representing the polynomial terms $C^{k}$.

    Parameters
    ----------
    array : list of int
        List of feature indices, where $k$ is the exponent minus one.
        e.g., `[0, 1, 2]` converts to $C^1, C^2, C^3$.

    Returns
    -------
    str
        The LaTeX-formatted string of the terms, enclosed in dollar signs.
        Example: "$C^1, C^2, C^3$"
    '''
    array_label = "$"
    for i,arr in enumerate(array):
        array_label += f"C^{arr+1}"
        if i < len(array)-1:
            array_label += f", "
        else:
            array_label += f"$"
    return array_label


def ES_sindy_coefficient_values(noise, IC, drp, rp_vect):
    '''
    Provides a library of pre-calculated SINDy coefficient vectors and the
    corresponding active degrees based on experimental conditions (noise level,
    initial condition (IC), spacing between rp values (drp), and a vector of proliferation rates (rp_vect)).

    This function appears to act as a lookup table for expected SINDy models
    under various simulated conditions.

    Parameters
    ----------
    noise : str
        Noise level, either "nonoise" or "lessnoise".
    IC : float
        Initial condition, $C_0$, e.g., 0.05 or 0.25.
    drp : float
        Relative death rate parameter, e.g., 0.01, 0.1, 0.5, or 1.
    rp_vect : numpy.ndarray
        A vector of proliferation rates $r_p$ that scales the coefficients.

    Returns
    -------
    full_coeff_library : list of numpy.ndarray
        A list where each element is a coefficient vector corresponding to a
        term in the model (e.g., coefficient for $C^1$, coefficient for $C^2$, etc.).
    unified_degrees : list of int
        A list of degrees (exponents minus one) corresponding to the terms
        present in the returned coefficient library.
    '''
    if noise == "nonoise":
        if IC == 0.05:
            coeff_C1_library = 0.5*rp_vect
            coeff_C2_library = -1*rp_vect
            full_coeff_library = [coeff_C1_library,
                                  coeff_C2_library]
            unified_degrees = [0,1]
        elif IC == 0.25:
            if drp == 0.01:
                coeff_C1_library =  0.49*rp_vect
                coeff_C2_library = -0.97*rp_vect
                coeff_C3_library = -0.03*rp_vect
                full_coeff_library = [coeff_C1_library,
                                      coeff_C2_library,
                                      coeff_C3_library]
                unified_degrees = [0,1,2]
            elif drp == 0.1:
                coeff_C1_library =  0.5*rp_vect
                coeff_C2_library = -0.98*rp_vect
                coeff_C3_library = -0.02*rp_vect
                full_coeff_library = [coeff_C1_library,
                                      coeff_C2_library,
                                      coeff_C3_library]
                unified_degrees = [0,1,2]
            elif drp == 0.5:
                coeff_C1_library =  0.5*rp_vect
                coeff_C2_library = -0.98*rp_vect
                coeff_C3_library = -0.02*rp_vect
                full_coeff_library = [coeff_C1_library,
                                      coeff_C2_library,
                                      coeff_C3_library]
                unified_degrees = [0,1,2]
            elif drp == 1:
                coeff_C1_library =  0.5*rp_vect
                coeff_C2_library = -1.02*rp_vect
                coeff_C3_library =  0.03*rp_vect
                full_coeff_library = [coeff_C1_library,
                                      coeff_C2_library,
                                      coeff_C3_library]
                unified_degrees = [0,1,2]
    elif noise == "lessnoise":
        if IC == 0.05:
            if drp == .01:
                coeff_C1_library =  0.5*rp_vect
                coeff_C2_library = -1.02*rp_vect
                coeff_C3_library =  0.02*rp_vect
                full_coeff_library = [coeff_C1_library,
                                      coeff_C2_library,
                                      coeff_C3_library]
                unified_degrees = [0,1,2]
            elif drp in [0.1, 0.5, 1]:
                coeff_C1_library = 0.5*rp_vect
                coeff_C2_library = -1.0*rp_vect
                full_coeff_library = [coeff_C1_library,
                                      coeff_C2_library]
                unified_degrees = [0,1]
        elif IC == 0.25:
            if drp == .01:
                coeff_C1_library =   0.5*rp_vect
                coeff_C2_library = -1.01*rp_vect
                coeff_C3_library =  0.01*rp_vect
                full_coeff_library = [coeff_C1_library,
                                      coeff_C2_library,
                                      coeff_C3_library]
                unified_degrees = [0,1,2]
            elif drp == .1:
                coeff_C1_library =   0.5*rp_vect
                coeff_C2_library = -0.98*rp_vect
                coeff_C3_library = -0.02*rp_vect
                full_coeff_library = [coeff_C1_library,
                                      coeff_C2_library,
                                      coeff_C3_library]
                unified_degrees = [0,1,2]
            elif drp == .5:
                coeff_C1_library =  0.51*rp_vect
                coeff_C2_library = -1.04*rp_vect
                coeff_C3_library =  0.05*rp_vect
                full_coeff_library = [coeff_C1_library,
                                      coeff_C2_library,
                                      coeff_C3_library]
                unified_degrees = [0,1,2]
            elif drp == 1:
                coeff_C1_library =  0.51*rp_vect
                coeff_C2_library = -1.06*rp_vect
                coeff_C3_library =  0.07*rp_vect
                full_coeff_library = [coeff_C1_library,
                                      coeff_C2_library,
                                      coeff_C3_library]
                unified_degrees = [0,1,2]
    return full_coeff_library, unified_degrees