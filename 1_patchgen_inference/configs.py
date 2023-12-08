import torch

### patch generation

PATCH_SIZE   = 512
RESOLUTION   = 100
P_DOWN_LEVEL = 2
K            = 10
TUMOR_RATIO  = 0.05
TISSUE_RATIO = 0.3
P_SLIDE_LIST = []
P_SAVE_PATH  = ''


### WSI inference

W_DOWN_LEVEL  = -1
MODEL_WEIGHT  = '' #'/workspace/log/weight/pytorch/final_model.pth'
# data = pd.read_csv('/workspace/data/feature_extraction/path_table.csv', encoding='cp949')
# slide_list = list(data['slide_path'])[:1]
W_SLIDE_LIST  = []
W_SAVE_PATH   = '' # '/workspace/data/deep_learning/slide_level/slide_inference/test'
ANNO_PATH     = ''
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"