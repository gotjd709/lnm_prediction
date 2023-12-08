import torch

DPI           = 300

# data EDA
D_HGH_LIST    = [] # glob.glob('/workspace/data/deep_learning/patch_level/train/hgh/*/input_x100/*.png')
D_KBSMC_LIST  = [] # glob.glob('/workspace/data/deep_learning/patch_level/train/kbsmc/*/input_x100/*.png')
D_SS_LIST     = [] # glob.glob('/workspace/data/deep_learning/patch_level/train/ss/*/input_x100/*.png')
D_ISH_LIST    = [] # glob.glob('/workspace/data/deep_learning/patch_level/ish/*/input_x100/*.png')
D_KUMC_LIST   = [] # glob.glob('/workspace/data/deep_learning/patch_level/kumc/*/input_x100/*.png')
D_SAVE_PATH   = '' # '/workspace/data/figure_zip/0_data_schema/01_color_distribution'

# UMAP
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
U_SAVE_PATH   = '' # '/workspace/data/figure_zip/1_deep_learning/12_encoder_model_quality'
MODEL_PATH    = '' # '/workspace/log/weight/pytorch/final_model.pth'
TEST_PATH     = [] # glob.glob('/workspace/data/deep_learning/patch_level/ish/*/input_y100/*.png') + glob.glob('/workspace/data/deep_learning/patch_level/kumc/*/input_y100/*.png')
TEST_PICKLE   = '' # '/workspace/data/deep_learning/patch_level/test_data.pickle'
PICKLE_PATH1  = '' # '/workspace/data/deep_learning/patch_level/target_df.pickle'
PICKLE_PATH2  = '' # '/workspace/data/deep_learning/patch_level/x_reduced_list.pickle'
MAX_PER       = 0.7
MIN_PER       = 0.05
TARGET        = 2
X_MAX         = 20
X_MIN         = 10
Y_MAX         = 25
Y_MIN         = 5

# slide EDA
S_HGH_LIST    = [] # glob.glob('/data/egc/HGH/*/*.svs')
S_KBSMC_LIST  = [] # glob.glob('/data/egc/KBSMC/*/*.tiff')
S_SS_LIST     = [] # glob.glob('/data/egc/SS/*/*.ndpi')
S_ISH_LIST    = [] # glob.glob('/data/egc/ISH/*/*.svs')
S_KUMC_LIST   = [] # glob.glob('/data/egc/KUMC/*/*.svs')
GT_LIST       = [] # glob.glob('/workspace/data/deep_learning/slide_level/ground_truth/*.png')
S_SAVE_PATH1  = '' # '/workspace/data/figure_zip/0_data_schema/00_slide_image'
S_SAVE_PATH2  = '' # '/workspace/data/figure_zip/1_deep_learning/11_slide_level_results'

# roc_curve
SEED          = 12345
MORPH_INFO    = 'morphology'
PATHO_INFO    = 'pathology'
CSV_PATH      = '' # '/workspace/data/feature_extraction/feature_results.csv'
SAVE_PATH     = '' # '/workspace/data/figure_zip/3_machine_learning'
SLIDE_PATH    = '' # '/workspace/data/deep_learning/slide_level/slide_image'
PRED_PATH     = '' # '/workspace/data/deep_learning/slide_level/slide_inference'

R_HGH_PATH    = '' # '/data/egc/hgh.csv'
R_ISH_PATH    = '' # '/data/egc/ish.csv'
R_KUMC_PATH   = '' # '/data/egc/kumc.csv'
R_KBSMC_PATH  = '' # '/data/egc/kbsmc.csv'