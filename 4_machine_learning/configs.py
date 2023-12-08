DPI        = 300
SEED       = 12345
MORPH_INFO = 'morphology'
PATHO_INFO = 'pathology'
CSV_PATH   = '' # '/workspace/data/feature_extraction/feature_results.csv'
HGH_PATH   = '' # '/data/egc/hgh.csv'
ISH_PATH   = '' # '/data/egc/ish.csv'
KUMC_PATH  = '' # '/data/egc/kumc.csv'
KBSMC_PATH = '' # '/data/egc/kbsmc.csv'
SAVE_PATH  = '' # '/workspace/data/figure_zip/3_machine_learning'

SGD_GRID = {
    'l1_ratio': [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
    'alpha': [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
}
XGB_GRID = {
    'max_depth'        : [3,4,5,7],
    'learning_rate'    : [0.1, 0.05, 0.01],
    'gamma'            : [0, 0.25, 1],
    'reg_lambda'       : [0, 1, 10],
    'scale_pos_weight' : [1, 3, 5],
    'subsample'        : [0.8],
    'colsample_bytree' : [0.5]
}
LGB_GRID = {
    'num_leaves': [5, 10, 15, 20, 30, 40, 50], 
    'min_child_samples': [5, 10, 15],
    'max_depth': [-1, 5, 10, 20],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'reg_alpha': [0.01, 0.05, 0.1, 0.15, 0.2]
}
CAT_GRID = {
    'learning_rate'       : [0.01, 0.05, 0.1],
    'bagging_temperature' : [0.01, 0.05, 0.1],
    'max_depth'           : [3, 5],
    'random_strength'     : [1, 5],
    'colsample_bylevel'   : [0.4, 0.5],
    'l2_leaf_reg'         : [1e-8, 1e-6],
}
SGL_GRID = {
    'group_reg': [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05], 
    'l1_reg': [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05], 
    'n_iter': [3, 5, 7, 10],
    'tol': [0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.0001, 0.00015, 0.0002]
}