import glob, pdb, sys

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
    t_solve_span = (t[0], t[-1])

    return solve_ivp(
                      BDM_RHS, t_solve_span, IC, t_eval=t,
                        args = (Pp,coefficients,param_degree, C_degree)
                      ).y.T

def unified_model_training(optimizer,files,param_degree=1,C_degree = 4):

    Theta, Ct, t = unified_library_build(files,param_degree,C_degree)
    
    lib = IdentityLibrary().fit(Theta)
    
    p_deg_mesh, C_deg_mesh = np.meshgrid(np.arange(1,param_degree+1),
                                             np.arange(1,C_degree+1),
                                             indexing = "ij")
        
    input_features = [f"P_p^{pdeg}*C^{cdeg}" for (pdeg,cdeg) in zip(p_deg_mesh.reshape(-1),
                                                                C_deg_mesh.reshape(-1))] #+ [Pp**2*(C**p) for p in np.arange(1,deg+1)]
    sindy_model = ps.SINDy(feature_library=lib,
                           optimizer=optimizer,
                          feature_names=input_features)
    
    sindy_model.fit(Theta, x_dot=Ct)#, t=t_train)
    
    return sindy_model

def unified_library_build(files,param_degree=1,C_degree = 4):
    
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
        
        Theta_ = np.array([(Pp**pdeg)*(C_**cdeg) for (pdeg,cdeg) in zip(p_deg_mesh.reshape(-1),
                                                                C_deg_mesh.reshape(-1))])[:,:,0].T 
        
        # Theta_ = np.hstack([Pp*C_,    Pp*(C_**2),    Pp*(C_**3),    Pp*(C_**4)])
        #                     #Pp**2*C_, Pp**2*(C_**2), Pp**2*(C_**3), Pp**2*(C_**4)])

        if initialize_library is True:

            Theta = Theta_
            Ct    = Ct_
            t     = t_
            
            initialize_library = False

        else:

            Theta = np.vstack([Theta, Theta_])
            Ct    = np.vstack([Ct,    Ct_   ])
            t     = np.vstack([t,    t_   ])
            
    return Theta, Ct, t

def tensor_data_build(files):
    
    #initialize tensor
    Cds = []
    #time arrays
    ts = []
    #Pps
    Pps = []
    
    for file in files:

        mat = np.load(file,allow_pickle=True).item()

        C_ = mat['variables'][1]
        Ct_ = mat['variables'][2]
        t_ = mat['variables'][0]

        Pp_ = float(mat['rp'][0][0])
        Pd_ = float(mat['rd'][0][0])
        Pm_ = float(mat['rm'][0][0])
        
        Cds.append(C_[:,0])
        ts.append(t_[:,0])
        Pps.append(Pp_)

    Cds = np.array(Cds)
    
    return Cds, ts, Pps

def tensor_model_build(Cds,ts,Pps,coefficients,param_degree=1,C_degree=4):
    
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
    Calculate the mean squared error between two arrays.

    Args:
        a (numpy.ndarray): The first array.
        b (numpy.ndarray): The second array.

    Returns:
        float: The mean squared error between `a` and `b`.
    """
    assert a.shape == b.shape
    return ((a - b)**2).mean()   

def model_training_CV(optimizer,
                      files_Train_list,
                      files_Test_list,
                      param_degree=1,
                      C_degree=4):

    sindy_aic_list = []
    sindy_model_coeffs_list = []
    
    for k in range(len(files_Train_list)):
        
        sindy_model = unified_model_training(optimizer,files_Train_list[k],param_degree,C_degree)
        coefficients = sindy_model.coefficients()
    
        CDsTest, tsTest, PpsTest = tensor_data_build(files_Test_list[k])
        CmsTest = tensor_model_build(CDsTest,tsTest,PpsTest,coefficients,param_degree,C_degree)

        N = CmsTest.size
        MSEval = MSE(CmsTest,CDsTest)
        sindy_num_params = np.sum(coefficients[0]!=0)
        sindy_aic = N*np.log(MSEval) + 2*sindy_num_params

        sindy_aic_list.append(sindy_aic)
        sindy_model_coeffs_list.append(coefficients)
        
    return sindy_aic_list, sindy_model_coeffs_list

def find_opt_with_threshold(aic_list,old_opt,max_param_size,coeff_list,CV_nums):
    
    '''Only consider AIC scores for values of lambda where no parameters exceed max_param_size=50'''

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



def trans(x,N):

    '''
    convert decimal number to binary representation
    
    inputs:
    
    x           : number
    N           : length of binary representation
   
    outputs:

    w           : binary vector from x
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
    convert binary representation to decimal number
    
    inputs:
    
    x           : binary vector 
   
    outputs:

    dec         : decimal number
    '''


    n = len(x)-1
    dec = 0
    for i in np.arange(n+1):
        dec = dec + x[i]*2**(n-i)
    return dec

def partition_data(data_type,CV_nums,IC,drp):
    
    
    rp_sparse = np.arange(0.01,5.01,drp)
    if data_type == "ABM":
        model_str = "ABM"
        file_header = "logistic_ABM_sim"
        file_ending = "_real25"
    # elif data_type == "smooth_ABM":
    #     model_str = "data_smooth_deriv"
    #     file_header = "logistic_ABM_sim"
    #     file_ending = ""
    elif "mean_field" in data_type:
        model_str = f"Data_{data_type}"
        file_header = "gen_mfld_data"
        file_ending = ""
        
    files = []
    for rp in rp_sparse:
        rd = rp/2
        rp, rd = format_rp_rd(rp,rd)
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
    
    sindy_aic_list = []
    sindy_model_coeffs_list = []

    #lower and upper log limits for lasso regularization parameter
    lower_log = -9
    upper_log = -1
    #lasso settings
    max_lasso_iter = 100000
    #library for sindy
    

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

    #threshold for max absolute value of parameter size
    max_param_size = 20
    
    #indices of the lasso regulatization parameter where the lowest AIC scores occur
    sindy_opt = np.argmin(sindy_aic_list)
    sindy_opt = find_opt_with_threshold(sindy_aic_list,sindy_opt,max_param_size,sindy_model_coeffs_list,CV_nums)

    return sindy_opt

def get_final_learned_eqn(xi_list):
    xi_vote = [[] for d in np.arange(len(xi_list))]

    # Extract how many times each model is learned in the test/train splits
    xi_vote_tmp = []
    for j in range(len(xi_list)):
        xi_vote_tmp.append(trans_rev((np.abs(xi_list[j][0]) > 1e-4)*1))
    num_eqns = 3
    xi_vote_tmp = Counter(xi_vote_tmp).most_common(num_eqns)
    xi_vote = [x[0] for x in xi_vote_tmp]

    # Set up for bookkeeping for obtaining mean param estimates
    matrix_vote_initialized = False
    A = [""]

    #loop through coefficient estimates and extract those corresponding
    #to the most popular model
    for j in np.arange(len(xi_list)):
        xi_full = xi_list[j]
        match =  trans_rev(np.abs(xi_full[0]) > 1e-4 )*1 == xi_vote[0]
        if np.any(match):
            if not matrix_vote_initialized:
                A[0] = xi_full
                matrix_vote_initialized = True
            else:
                A[0] = np.vstack((A[0],xi_full))

    # Save mean coefficients for the most popular equation
    xi_vote_params_sindy = np.mean(A[0],axis=0)
    
    return xi_vote_params_sindy

def perform_final_model_selection(data_type, xi_vote_params_sindy,drp):
    
    ### Extract relevant features & coefficients for initial guess
    degs_sindy   = np.nonzero(xi_vote_params_sindy)[0]
    coeffs_sindy = xi_vote_params_sindy[degs_sindy]
    init_coeffs  = coeffs_sindy  # initial guess

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
    files = []
    for rp in rp_sparse:
        rd = rp/2
        rp, rd = format_rp_rd(rp,rd)
        files.append(f'../../data/{model_str}/{file_header}_rp_{rp:.2f}_rd_{rd}_rm_1_m_{IC}{file_ending}.npy')
        
    CDs, ts, Pps = tensor_data_build(files)
    
    ### To be optimized
    def cost_function(coeffs, CDs, ts, Pps, param_degree,
                                              C_degree): #print(coeffs)

        CMs = tensor_model_build(CDs,ts,Pps,[coeffs],learned_param_degrees,learned_C_degrees)

        return MSE(CDs,CMs)

    ### Perform optimization
    res = minimize(cost_function, init_coeffs, method='nelder-mead',
                       args=(CDs, ts, Pps, param_degree,C_degree), options={'xatol': 1e-8, 'disp': True,'maxfun':10**5})
    coeffs_sindy_opt = res.x
    
    return coeffs_sindy_opt, learned_C_degrees, learned_param_degrees

CV_nums = 10

param_degree = 1
C_degree = 10

data_type = "mean_field_lessnoise"#"mean_field_morenoise"#ABM, "smooth_ABM"

drps = [0.01, 0.1, 0.5, 1]
ICs = [0.05, 0.25]

index = int(sys.argv[1])
drp, IC = drps[index%4],ICs[index//4]

data_type = sys.argv[2]
assert data_type in ["ABM","mean_field_nonoise","mean_field_lessnoise"], "data_type must equal \"ABM\", \"mean_field_nonoise\", or \"mean_field_lessnoise\""

print(drp, IC)

files_Train_list, files_Test_list = partition_data(data_type,CV_nums,IC,drp)

sindy_aic_list, sindy_model_coeffs_list = perform_sindy_CV(files_Train_list,files_Test_list,param_degree, C_degree)

sindy_opt = select_sindy_aic(sindy_aic_list,sindy_model_coeffs_list,CV_nums)

selected_sindy_params_list = sindy_model_coeffs_list[sindy_opt]

xi_vote_params_sindy = get_final_learned_eqn(selected_sindy_params_list)

(coeffs_sindy_opt, 
 learned_C_degrees, 
 learned_param_degrees) = perform_final_model_selection(data_type,
                                                        xi_vote_params_sindy,
                                                        drp)

data = {"xi":coeffs_sindy_opt,
        "param_degree":param_degree,
        "C_degree":C_degree,
        "learned_C_degrees":learned_C_degrees, 
        "learned_param_degrees":learned_param_degrees}

np.save(f"../../results/ES-MEEQL/learned_coeffs_{data_type}_param_degree_{param_degree}_C_degree_{C_degree}_IC_{IC}_drp_{drp}.npy",data)