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

from helper_functions import *

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
                                                        drp, IC, param_degree, C_degree)


data = {"xi":coeffs_sindy_opt,
        "param_degree":param_degree,
        "C_degree":C_degree,
        "learned_C_degrees":learned_C_degrees, 
        "learned_param_degrees":learned_param_degrees}

np.save(f"../../results/ES-MEEQL/learned_coeffs_{data_type}_param_degree_{param_degree}_C_degree_{C_degree}_IC_{IC}_drp_{drp}.npy",data)