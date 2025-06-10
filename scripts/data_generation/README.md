The Python notebook Generate_MFM_data.ipynb can be used to generate the mean-field model datasets with variable amount of noise. 

The Matlab file Generate_ABM_data.m can be used to generate the ABM datasets for the birth-death-migration model. It uses functions Prolif_assay_func.m to run the model and Finite_diff_1d.m to calculate the derivatives. This saves datasets in .mat format. The python code ABM_data_convert_mat_to_npy.py is used to convert the datasets from .mat to .npy format.
