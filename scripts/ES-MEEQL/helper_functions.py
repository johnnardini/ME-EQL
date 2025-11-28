import glob, pdb
import numpy as np
from sklearn.linear_model import Lasso
import pysindy as ps
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from pysindy.feature_library import IdentityLibrary
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from collections import Counter
from scipy.optimize import minimize

def format_rp_rd(rp,rd):
    """
    Formats the reaction parameter (rp) and calculates the corresponding
    reaction diameter (rd) as half of rp.

    rp is rounded to 2 decimal places, or converted to an integer if it's
    a whole number. rd is then calculated as rp/2 and is rounded to 3
    decimal places, or converted to an integer if it's a whole number.

    Args:
        rp (float): The reaction parameter (Pp).
        rd (float): The reaction diameter (ignored, recalculated from rp).

    Returns:
        tuple: A tuple containing the formatted (rp, rd).
    """
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

def BDM_RHS(t, C, Pp, coefs, param_degree=1, C_degree = 4):
    """
    Calculates the right-hand side (RHS) of the differential equation (dC/dt)
    using a Basis Data Model (BDM) with polynomial features.

    The features are constructed from powers of the parameter (Pp) and the
    concentration (C), and the RHS is a linear combination of these features
    using the provided coefficients.

    Args:
        t (float or np.ndarray): Current time point(s). (Required by solve_ivp).
        C (float or np.ndarray): Current concentration value(s).
        Pp (float): The reaction parameter (e.g., rp).
        coefs (list of np.ndarray): The coefficients for the linear combination.
                                    Expected to be in the form [xi], where xi is
                                    a 1D array of coefficients.
        param_degree (int or np.ndarray, optional): The maximum degree of the
                                                    parameter Pp in the features,
                                                    or an array of parameter degrees
                                                    if C_degree is also an array. Defaults to 1.
        C_degree (int or np.ndarray, optional): The maximum degree of the
                                                concentration C in the features,
                                                or an array of concentration degrees
                                                if param_degree is also an array. Defaults to 4.

    Returns:
        np.ndarray: The value of dC/dt (the RHS of the ODE).
    """

    assert type(param_degree) in [int,np.ndarray], "param_degree must be int or np.ndarray"
    assert type(C_degree) in [int,np.ndarray], "C_degree must be int or np.ndarray"
    assert type(C_degree) == type(param_degree), "types of param_degree and C_degree must match"

    if type(param_degree) == int:

        p_deg_mesh, C_deg_mesh = np.meshgrid(np.arange(1,param_degree+1),
                                             np.arange(1,C_degree+1),
                                             indexing = "ij")

        features = [(Pp**pdeg)*(C**cdeg) for (pdeg,cdeg) in zip(p_deg_mesh.reshape(-1),
                                                                C_deg_mesh.reshape(-1))] #+ [Pp**2*(C**p) for p in np.arange(1,deg+1)]

    elif type(param_degree) == np.ndarray:

        features = [(Pp**pdeg)*(C**cdeg) for (pdeg,cdeg) in zip(param_degree,C_degree)]

    X = np.array(features).T

    return np.matmul(X,coefs[0])

def forward_solve(t, IC, Pp, coefficients, param_degree=1, C_degree = 4):
    """
    Solves the initial value problem (IVP) defined by BDM_RHS forward in time.

    Uses `scipy.integrate.solve_ivp` to compute the concentration C(t)
    given the initial condition and the differential equation model.

    Args:
        t (np.ndarray): Array of time points at which to evaluate the solution.
        IC (float or np.ndarray): The initial condition C(t[0]).
        Pp (float): The reaction parameter (rp).
        coefficients (list of np.ndarray): Coefficients for the BDM_RHS function.
        param_degree (int or np.ndarray, optional): Parameter degree(s) for BDM_RHS. Defaults to 1.
        C_degree (int or np.ndarray, optional): Concentration degree(s) for BDM_RHS. Defaults to 4.

    Returns:
        np.ndarray: The computed solution C(t) as a 2D array where columns
                    are variables and rows are time points.
    """
    t_solve_span = (t[0], t[-1])

    return solve_ivp(
                        BDM_RHS, t_solve_span, IC, t_eval=t,
                        args = (Pp,coefficients,param_degree, C_degree)
                    ).y.T

def tensor_data_build(files):
    """
    Loads experimental/simulation data from multiple .npy files and
    extracts the concentration, time, and reaction parameters (Pp).

    It is assumed that each file contains a dictionary with keys:
    'variables' (a list/array where index 1 is C and index 0 is t),
    'rp' (reaction parameter Pp), 'rd', and 'rm'.

    Args:
        files (list of str): A list of file paths to load.

    Returns:
        tuple: A tuple (Cds, ts, Pps) where:
            Cds (np.ndarray): 2D array of concentration data (C) from all files.
            ts (list of np.ndarray): List of time arrays (t) for each file.
            Pps (list of float): List of reaction parameter values (Pp) for each file.
    """

    #initialize tensor
    Cds = []
    #time arrays
    ts = []
    #Pps
    Pps = []

    for file in files:

        mat = np.load(file,allow_pickle=True).item()

        C_ = mat['variables'][1]
        Ct_ = mat['variables'][2] # Not used in return, but loaded
        t_ = mat['variables'][0]

        Pp_ = float(mat['rp'][0][0])
        Pd_ = float(mat['rd'][0][0]) # Not used in return, but loaded
        Pm_ = float(mat['rm'][0][0]) # Not used in return, but loaded

        Cds.append(C_[:,0])
        ts.append(t_[:,0])
        Pps.append(Pp_)

    Cds = np.array(Cds)

    return Cds, ts, Pps

def tensor_model_build(Cds,ts,Pps,coefficients,param_degree=1,C_degree=4):
    """
    Computes model solutions (Cms) for a set of initial conditions,
    time arrays, and reaction parameters using the forward_solve function.

    The initial condition (IC) is taken as the first value of the
    concentration data (Cds) for each time series.

    Args:
        Cds (np.ndarray): 2D array of concentration data, used to extract ICs.
        ts (list of np.ndarray): List of time arrays for each solution.
        Pps (list of float): List of reaction parameter values (Pp) for each solution.
        coefficients (list of np.ndarray): Coefficients for the BDM_RHS function.
        param_degree (int or np.ndarray, optional): Parameter degree(s) for BDM_RHS. Defaults to 1.
        C_degree (int or np.ndarray, optional): Concentration degree(s) for BDM_RHS. Defaults to 4.

    Returns:
        np.ndarray or float: A 2D numpy array of model solutions (Cms) if all
                             solutions have the same length as ts[0].
                             Returns a large constant (1e10 * ones array) if solution
                             lengths are inconsistent.
    """

    Cms = []
    for Cd,t,Pp in zip(Cds, ts, Pps):

        IC = Cd[0,None] #IC must be 1d; cannot be a scalar
        Cm = forward_solve(t,IC,Pp,coefficients,param_degree,C_degree)
        Cms.append(Cm[:,0])

    if np.all([len(Cm) == len(ts[0]) for Cm in Cms]):
        ## All arrays of the same size
        Cms = np.array(Cms)
    else:
        ## If one or more solutions are of different size, return 1e10
        Cms = 1e10*np.ones((len(Pps),len(ts[0])))

    return Cms

def MSE(a,b):
    """
    Calculate the mean squared error (MSE) between two arrays.

    Args:
        a (numpy.ndarray): The first array (e.g., data).
        b (numpy.ndarray): The second array (e.g., model output).

    Returns:
        float: The mean squared error between `a` and `b`.
    """
    assert a.shape == b.shape, "Arrays must have the same shape"
    return ((a - b)**2).mean()

def print_DE(xi,learned_param_degrees,learned_C_degrees):
    """
    Constructs and prints the symbolic representation of the differential
    equation (DE) learned by the model.

    The DE is represented as a linear combination of polynomial features
    of the form $x_i * P_p^{p_{deg}} * C^{c_{deg}}$.

    Args:
        xi (np.ndarray): The array of learned coefficients (xi).
        learned_param_degrees (np.ndarray): The array of parameter (Pp) degrees
                                            corresponding to each coefficient.
        learned_C_degrees (np.ndarray): The array of concentration (C) degrees
                                        corresponding to each coefficient.

    Returns:
        str: The string representation of the differential equation.
    """
    DE_list = [f"{round(x,2)}*P_p^{pdeg}*C^{cdeg}" for (x,pdeg,cdeg) in zip(xi,learned_param_degrees,learned_C_degrees)]
    DE_str = DE_list[0]

    for x,term in zip(xi[1:],DE_list[1:]):
        if x < 0:
            DE_str += (" - " + term[1:])
        else:
            DE_str += (" + " + term)

    print(DE_str)
    return(DE_str)

def unified_model_training(optimizer,files,param_degree=1,C_degree = 4):
    """
    Trains a SINDy model using data from multiple files.

    The function first builds the library matrix (Theta) and derivative data (Ct)
    by combining data across all files. It then defines the features and fits
    a SINDy model using the specified optimizer (e.g., Lasso).

    Args:
        optimizer (pysindy.optimizers.BaseOptimizer): The sparse regression
            optimizer (e.g., ps.STLSQ, Lasso) used by SINDy.
        files (list of str): List of file paths containing the system data.
        param_degree (int, optional): The maximum power of the reaction
            parameter (Pp) in the features. Defaults to 1.
        C_degree (int, optional): The maximum power of the concentration (C)
            in the features. Defaults to 4.

    Returns:
        pysindy.SINDy: The trained SINDy model object.
    """
    Theta, Ct, t = unified_library_build(files,param_degree,C_degree)

    lib = IdentityLibrary().fit(Theta)

    p_deg_mesh, C_deg_mesh = np.meshgrid(np.arange(1,param_degree+1),
                                              np.arange(1,C_degree+1),
                                              indexing = "ij")

    input_features = [f"P_p^{pdeg}*C^{cdeg}" for (pdeg,cdeg) in zip(p_deg_mesh.reshape(-1),
                                                                 C_deg_mesh.reshape(-1))]
    sindy_model = ps.SINDy(feature_library=lib,
                            optimizer=optimizer,
                          feature_names=input_features)

    sindy_model.fit(Theta, x_dot=Ct)#, t=t_train)

    return sindy_model

def unified_library_build(files,param_degree=1,C_degree = 4):
    """
    Constructs the unified library matrix (Theta) and derivative data (Ct)
    from multiple simulation files for SINDy training.

    It iterates through files, extracts concentration (C), derivative (Ct),
    and parameter (Pp), and builds the features of the form $P_p^{p_{deg}} * C^{c_{deg}}$.

    Args:
        files (list of str): List of file paths containing the system data.
        param_degree (int, optional): Maximum degree of the parameter (Pp).
            Defaults to 1.
        C_degree (int, optional): Maximum degree of the concentration (C).
            Defaults to 4.

    Returns:
        tuple: (Theta, Ct, t) where:
            Theta (np.ndarray): The unified feature library matrix.
            Ct (np.ndarray): The unified time derivative data ($\dot{C}$).
            t (np.ndarray): The unified time array.
    """

    p_deg_mesh, C_deg_mesh = np.meshgrid(np.arange(1,param_degree+1),
                                              np.arange(1,C_degree+1),
                                              indexing = "ij")

    initialize_library = True
    for file in files:

        mat = np.load(file,allow_pickle=True).item()

        C_ = mat['variables'][1]
        Ct_ = mat['variables'][2]
        t_ = mat['variables'][0]

        Pp = float(mat['rp'][0][0])
        Pd = float(mat['rd'][0][0])
        Pm = float(mat['rm'][0][0])

        # Calculate features for this file: (Pp^pdeg) * (C_^cdeg)
        Theta_ = np.array([(Pp**pdeg)*(C_**cdeg) for (pdeg,cdeg) in zip(p_deg_mesh.reshape(-1),
                                                                 C_deg_mesh.reshape(-1))])[:,:,0].T

        if initialize_library is True:

            Theta = Theta_
            Ct    = Ct_
            t     = t_

            initialize_library = False

        else:

            Theta = np.vstack([Theta, Theta_])
            Ct    = np.vstack([Ct,    Ct_    ])
            t     = np.vstack([t,     t_     ])

    return Theta, Ct, t


def model_training_CV(optimizer,
                      files_Train_list,
                      files_Test_list,
                      param_degree=1,
                      C_degree=4):
    """
    Performs Cross-Validation (CV) model training and testing for a specific
    SINDy optimizer configuration.

    The function iterates through the CV folds, trains a SINDy model on the
    training set, simulates the result on the test set, and calculates the
    Akaike Information Criterion (AIC) score.

    Args:
        optimizer (pysindy.optimizers.BaseOptimizer): The sparse regression
            optimizer instance.
        files_Train_list (list of list of str): List of training file lists for each CV fold.
        files_Test_list (list of list of str): List of testing file lists for each CV fold.
        param_degree (int, optional): Maximum degree of Pp for the library. Defaults to 1.
        C_degree (int, optional): Maximum degree of C for the library. Defaults to 4.

    Returns:
        tuple: (sindy_aic_list, sindy_model_coeffs_list) where:
            sindy_aic_list (list of float): The AIC score for each CV fold.
            sindy_model_coeffs_list (list of np.ndarray): The learned coefficients
                                                          for each CV fold.
    """
    sindy_aic_list = []
    sindy_model_coeffs_list = []

    for k in range(len(files_Train_list)):

        sindy_model = unified_model_training(optimizer,files_Train_list[k],param_degree,C_degree)
        coefficients = sindy_model.coefficients()

        # Load test data and simulate the model over the test conditions
        CDsTest, tsTest, PpsTest = tensor_data_build(files_Test_list[k])
        CmsTest = tensor_model_build(CDsTest,tsTest,PpsTest,coefficients,param_degree,C_degree)

        # Calculate AIC
        N = CmsTest.size
        MSEval = MSE(CmsTest,CDsTest)
        sindy_num_params = np.sum(coefficients[0]!=0)
        sindy_aic = N * np.log(MSEval) + 2 * sindy_num_params # AIC formula

        sindy_aic_list.append(sindy_aic)
        sindy_model_coeffs_list.append(coefficients)

    return sindy_aic_list, sindy_model_coeffs_list

def find_opt_with_threshold(aic_list,old_opt,max_param_size,coeff_list,CV_nums):
    """
    Adjusts the optimal index found by minimum AIC to ensure model coefficients
    do not exceed a maximum allowed magnitude threshold.

    This prevents selecting a model that might have a slightly lower AIC but
    unphysically large coefficient values.

    Args:
        aic_list (list of float): List of mean AIC scores across CV folds for
                                  different regularization parameters (lambdas).
        old_opt (int): The index corresponding to the minimum AIC score.
        max_param_size (float): The maximum allowed absolute value for a model coefficient.
        coeff_list (list of list of np.ndarray): The learned coefficients for
                                                 each regularization parameter and CV fold.
        CV_nums (int): The number of cross-validation folds.

    Returns:
        int: The adjusted optimal index (lambda index) that satisfies the
             coefficient magnitude constraint.
    """

    # '''Only consider AIC scores for values of lambda where no parameters exceed max_param_size=50'''

    for kk in range(old_opt,len(aic_list)+1):
        num_past_thresh = 0

        # Check all CV folds for the current regularization index (kk)
        for jj in range(CV_nums):
            sindy_model_coeffs = coeff_list[old_opt][jj][0]
            if np.sum(np.abs(sindy_model_coeffs)>max_param_size)>0:
                num_past_thresh+=1

        if num_past_thresh>0:
            old_opt = kk+1
        else:
            break

    return old_opt


def trans(x,N):
    """
    Converts a decimal number to its binary vector representation of a
    specified length N.

    This is typically used in SINDy model selection for converting a model's
    coefficient support (non-zero/zero pattern) into a unique integer identifier.

    Args:
        x (int): The decimal number to convert.
        N (int): The length of the binary representation (number of features).

    Returns:
        np.ndarray: A 2D numpy array representing the binary vector.
    """
    # '''
    # convert decimal number to binary representation

    # inputs:

    # x           : number
    # N           : length of binary representation

    # outputs:

    # w           : binary vector from x
    # '''

    y=np.copy(x)
    if y == 0: return[0]
    bit = []
    for i in np.arange(N):
        bit.append(y % 2)
        y >>= 1
    return np.atleast_2d(np.asarray(bit[::-1]))

#go from binary representation to decimal number
def trans_rev(x):
    """
    Converts a binary vector representation back to its decimal number.

    This is typically used to convert the unique integer identifier of a
    SINDy model back into a decimal number.

    Args:
        x (np.ndarray): The binary vector (e.g., the model support).

    Returns:
        int: The resulting decimal number.
    """
    # '''
    # convert binary representation to decimal number

    # inputs:

    # x           : binary vector

    # outputs:

    # dec         : decimal number
    # '''


    n = len(x)-1
    dec = 0
    for i in np.arange(n+1):
        dec = dec + x[i]*2**(n-i)
    return dec

def partition_data(data_type,CV_nums,IC,drp):
    """
    Generates a list of file paths based on data type and parameters,
    and then splits these files into cross-validation (CV) training and
    testing sets.

    The file paths are constructed using formatted reaction parameters (rp, rd)
    and initial conditions (IC).

    Args:
        data_type (str): Identifier for the data source (e.g., "ABM", "mean_field").
        CV_nums (int): The number of cross-validation folds to create.
        IC (float): The initial condition used in the simulation files.
        drp (float): The step size for generating the reaction parameter (rp) values.

    Returns:
        tuple: (files_Train_list, files_Test_list) where each is a list of
               file path lists corresponding to the CV folds.
    """


    rp_sparse = np.arange(0.01,5.01,drp)
    if data_type == "ABM":
        model_str = "ABM"
        file_header = "logistic_ABM_sim"
        file_ending = "_real25"
    # elif data_type == "smooth_ABM":
    #    model_str = "data_smooth_deriv"
    #    file_header = "logistic_ABM_sim"
    #    file_ending = ""
    elif "mean_field" in data_type:
        model_str = f"Data_{data_type}"
        file_header = "gen_mfld_data"
        file_ending = ""

    files = []
    for rp in rp_sparse:
        rd = rp/2
        rp, rd = format_rp_rd(rp,rd) # Assumes this function is available
        files.append(f'../../data/{model_str}/{file_header}_rp_{rp:.2f}_rd_{rd}_rm_1_m_{IC}{file_ending}.npy')

    files_Train_list = []
    files_Test_list = []

    i = 0
    for _ in range(CV_nums):

        (files_Train,
          files_Test) = train_test_split(files,train_size=0.8,test_size=0.2,random_state=i)

        files_Train_list.append(files_Train)
        files_Test_list.append(files_Test)

        i += 1

    return files_Train_list, files_Test_list

def perform_sindy_CV(files_Train_list,files_Test_list,param_degree=1, C_degree=4):
    """
    Performs the entire SINDy Cross-Validation procedure over a range of
    Lasso regularization parameters (lambdas).

    It calculates the mean AIC across all CV folds for each lambda value and
    stores the learned coefficients.

    Args:
        files_Train_list (list of list of str): Training file lists for CV folds.
        files_Test_list (list of list of str): Testing file lists for CV folds.
        param_degree (int, optional): Maximum degree of Pp for the library. Defaults to 1.
        C_degree (int, optional): Maximum degree of C for the library. Defaults to 4.

    Returns:
        tuple: (sindy_aic_list, sindy_model_coeffs_list) where:
            sindy_aic_list (list of float): The mean AIC score for each lambda.
            sindy_model_coeffs_list (list of list of np.ndarray): The learned
                                                                  coefficients
                                                                  for each lambda and CV fold.
    """

    sindy_aic_list = []
    sindy_model_coeffs_list = []

    #lower and upper log limits for lasso regularization parameter
    lower_log = -9
    upper_log = -1
    #lasso settings
    max_lasso_iter = 100000


    for lmb in np.logspace(lower_log,upper_log,20):

        optimizer = Lasso(alpha=lmb, max_iter=max_lasso_iter, fit_intercept=False)

        ### Then determine best hyperpar value.
        sindy_aic, sindy_model_coeffs = model_training_CV(optimizer,
                                                      files_Train_list,
                                                      files_Test_list,
                                                      param_degree,
                                                      C_degree)

        sindy_aic_list.append(np.mean(sindy_aic))
        sindy_model_coeffs_list.append(sindy_model_coeffs)

    return sindy_aic_list, sindy_model_coeffs_list

def select_sindy_aic(sindy_aic_list,sindy_model_coeffs_list,CV_nums):
    """
    Selects the optimal regularization parameter index based on the minimum
    AIC score, while also enforcing a maximum coefficient magnitude constraint.

    Args:
        sindy_aic_list (list of float): Mean AIC scores for each lambda value.
        sindy_model_coeffs_list (list of list of np.ndarray): Learned coefficients
                                                              for each lambda and CV fold.
        CV_nums (int): The number of cross-validation folds.

    Returns:
        int: The index of the optimal regularization parameter (lambda).
    """

    #threshold for max absolute value of parameter size
    max_param_size = 20

    # indices of the lasso regulatization parameter where the lowest AIC scores occur
    sindy_opt = np.argmin(sindy_aic_list)
    sindy_opt = find_opt_with_threshold(sindy_aic_list,sindy_opt,max_param_size,sindy_model_coeffs_list,CV_nums)

    return sindy_opt

def get_final_learned_eqn(xi_list):
    """
    Determines the final, most frequently selected SINDy model across all
    cross-validation folds and calculates the mean coefficients for that model.

    The model structure is identified by converting the support (non-zero/zero
    pattern) of the coefficients to a unique decimal number.

    Args:
        xi_list (list of np.ndarray): List of learned coefficient arrays
                                      from the CV folds for the optimal lambda.

    Returns:
        np.ndarray: The mean coefficient vector ($xi$) for the most frequently
                    selected model structure.
    """
    xi_vote = [[] for d in np.arange(len(xi_list))]

    # Extract how many times each model is learned in the test/train splits
    xi_vote_tmp = []
    for j in range(len(xi_list)):
        # Convert coefficient support to a unique decimal ID
        xi_vote_tmp.append(trans_rev((np.abs(xi_list[j][0]) > 1e-4)*1))
    num_eqns = 3
    # Find the most common model IDs
    xi_vote_tmp = Counter(xi_vote_tmp).most_common(num_eqns)
    xi_vote = [x[0] for x in xi_vote_tmp]

    # Set up for bookkeeping for obtaining mean param estimates
    matrix_vote_initialized = False
    A = [""]

    #loop through coefficient estimates and extract those corresponding
    #to the most popular model
    for j in np.arange(len(xi_list)):
        xi_full = xi_list[j]
        # Check if the coefficient support matches the most popular model ID
        match = trans_rev(np.abs(xi_full[0]) > 1e-4 )*1 == xi_vote[0]
        if np.any(match):
            if not matrix_vote_initialized:
                A[0] = xi_full
                matrix_vote_initialized = True
            else:
                A[0] = np.vstack((A[0],xi_full))

    # Save mean coefficients for the most popular equation
    xi_vote_params_sindy = np.mean(A[0],axis=0)

    return xi_vote_params_sindy

def perform_final_model_selection(data_type, xi_vote_params_sindy,drp,IC,param_degree,C_degree):
    """
    Takes the structure of the best SINDy model (from cross-validation) and
    refines its non-zero coefficients using non-linear least squares
    optimization (e.g., Nelder-Mead) over the entire dataset.

    This process minimizes the MSE between the model's forward simulation and
    the ground truth data.

    Args:
        data_type (str): Identifier for the data source (e.g., "ABM", "mean_field").
        xi_vote_params_sindy (np.ndarray): The mean coefficient vector for the
                                           most popular SINDy model structure.
        drp (float): The step size for generating the reaction parameter (rp) values
                     to build the full dataset.
        IC (float): The initial condition used in the simulation files.
        param_degree (int): Maximum degree of rp for the library.
        C_degree (int): Maximum degree of C for the library.

    Returns:
        tuple: (coeffs_sindy_opt, learned_C_degrees, learned_param_degrees) where:
            coeffs_sindy_opt (np.ndarray): The coefficients optimized via
                                           non-linear fitting.
            learned_C_degrees (np.ndarray): The concentration degrees of the
                                            non-zero terms in the final model.
            learned_param_degrees (np.ndarray): The parameter degrees of the
                                                non-zero terms in the final model.
    """
    ### Extract relevant features & coefficients for initial guess
    degs_sindy   = np.nonzero(xi_vote_params_sindy)[0]
    coeffs_sindy = xi_vote_params_sindy[degs_sindy]
    init_coeffs  = coeffs_sindy  # initial guess

    # Map the index of the non-zero terms back to the degrees used in the library
    # C_degree must be available here.
    learned_C_degrees = (degs_sindy%C_degree)+1
    learned_param_degrees = (degs_sindy//C_degree)+1

    rp_sparse = np.arange(0.01,5.01,drp)
    if data_type == "ABM":
        model_str = "ABM"
        file_header = "logistic_ABM_sim"
        file_ending = "_real25"
    elif "mean_field" in data_type:
        model_str = f"Data_{data_type}"
        file_header = "gen_mfld_data"
        file_ending = ""

    # Re-assemble the full dataset
    files = []
    for rp in rp_sparse:
        rd = rp/2
        rp, rd = format_rp_rd(rp,rd) # Assumes this function is available
        files.append(f'../../data/{model_str}/{file_header}_rp_{rp:.2f}_rd_{rd}_rm_1_m_{IC}{file_ending}.npy')

    CDs, ts, Pps = tensor_data_build(files) # Assumes this function is available

    ### To be optimized
    def cost_function(coeffs, CDs, ts, Pps, param_degree,
                                             C_degree): #print(coeffs)

        # Only use the learned degrees/structure for model simulation
        CMs = tensor_model_build(CDs,ts,Pps,[coeffs],learned_param_degrees,learned_C_degrees) # Assumes this function is available

        return MSE(CDs,CMs) # Assumes this function is available

    ### Perform optimization
    res = minimize(cost_function, init_coeffs, method='nelder-mead',
                     args=(CDs, ts, Pps, param_degree,C_degree), options={'xatol': 1e-8, 'disp': True,'maxfun':10**5})
    coeffs_sindy_opt = res.x

    return coeffs_sindy_opt, learned_C_degrees, learned_param_degrees