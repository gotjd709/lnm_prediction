from configs           import *
import multiprocessing
import utils

### Global Variables from config

def extract(main, slide_path, save_path):
    anno_path = '.'.join(slide_path.split('.')[:-1]) + '.xml'
    main._slide_setting(slide_path, anno_path, save_path)
    main.extract()


if __name__ == '__main__':
    # main setting
    main = utils.ExtractPatch(PATCH_SIZE, RESOLUTION, P_DOWN_LEVEL, TUMOR_RATIO, TISSUE_RATIO)

    # multi processing
    for i in range(0,len(P_SLIDE_LIST),K):
        slide_batch = P_SLIDE_LIST[i:i+K]
        for slide_path in slide_batch:
            p = multiprocessing.Process(target=extract, args=(main, slide_path, P_SAVE_PATH, ))
            p.start()
        p.join()

