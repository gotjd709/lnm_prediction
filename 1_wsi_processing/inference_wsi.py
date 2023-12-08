from configs    import *
import utils

### Global Variables from config

if __name__ == '__main__':    
    main = utils.InferenceWSI(PATCH_SIZE, RESOLUTION, W_DOWN_LEVEL)
    for slide_path in W_SLIDE_LIST:
        main._slide_setting(slide_path, ANNO_PATH, W_SAVE_PATH)
        main.inference(weight=MODEL_WEIGHT)
