import torch

MODEL               = 'DeepLabV3Plus' 
ENCODER_NAME        = 'se_resnext101_32x4d' 
ENCODER_WEIGHTS     = 'imagenet' 
CLASSES             = 4 
ACTIVATION          = 'softmax' 
TRAIN_PATH          = '' # '/workspace/data/deep_learning/patch_level/train_data.pickle'
VALID_PATH          = '' # '/workspace/data/deep_learning/patch_level/valid_data.pickle'
TEST_PATH           = '' # '/workspace/data/deep_learning/patch_level/test_data.pickle'
EXTERNAL_PATCH_LIST = [] # glob.glob('/workspace/data/deep_learning/patch_level/ish/*/input_y100/*.png') + glob.glob('/workspace/data/deep_learning/patch_level/kumc/*/input_y100/*.png')
LOG_PATH            = '' # '/workspace/log/tensorboard'
WEIGHT_PATH         = '' # '/workspace/log/weight/pytorch'
RANDOM_SEED         = 333
INPUT_SHAPE         = (512,512)
PATIENCE            = 8 
SAMPLER             = True 
BATCH_SIZE          = 64
NUM_WORKER          = 4 
LOSS                = 'DiceLoss' 
DESCRIPTION         = '' # 'last_test_patch_check'
LR                  = 1e-4 
OPTIMIZER           = 'Adam' 
EPOCH               = 100 
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"