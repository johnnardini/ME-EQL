This folder contains files to run the Embedded structure (ES) Multi-experiment equation learning (ME-EQL) pipeline. 

The python file `Parameter_inclusive_learing.py` can be run by entering: ```python Parameter_inclusive_learning.py i data_type``` where `i` determines the number of model simulations used in the analysis and `data_type` determines the type of data being used. (If `i%4==0`, then 500 simulations are used. If `i%4==1`, then 50 simulations are used). If `i%4==2`, then 10 simulations are used). If `i%4==3`, then 5 simulations are used). The `data_type` options are `mean_field_nonoise`, `mean_field_lessnoise`, and `ABM`.

The jupyter notebook `sparsification_analysis.ipynb` can be run to visualize the results for each scenario. This includes visualizing mean-squared errors (MSEs) of the learned equations in predicting data over the proliferation parameter, and printing the learned equations.
