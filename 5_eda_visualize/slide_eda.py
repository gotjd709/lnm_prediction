from configs                import *
import matplotlib.pyplot    as plt
import numpy                as np
import openslide
import cv2

### Global Variables from config

def save_slide_image(slide_path, i):
    slide = openslide.OpenSlide(slide_path)
    name = slide_path.split('/')[3]
    svs = np.array(slide.read_region(location=(0,0), level=slide.level_count-1, size=(slide.level_dimensions[-1]))).astype(np.uint8)[...,:3]
    plt.imshow(svs)
    plt.axis('off')
    plt.savefig(f'{S_SAVE_PATH1}/{name}_{i}.png', dpi=DPI)

def save_slide_results(idx, gt_path):
    gt = cv2.imread(gt_path,0)
    pr = cv2.imread(''+gt_path.split('/')[-1],0)
    img = cv2.imread(''+gt_path.split('/')[-1])
    pr = cv2.imread(f'{PRED_PATH}/'+gt_path.split('/')[-1],0)
    img = cv2.imread(f'{SLIDE_PATH}/'+gt_path.split('/')[-1])
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title('slide image')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(gt, vmin=0, vmax=3, cmap='Oranges')
    plt.title('ground turth')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(pr, vmin=0, vmax=3, cmap='Oranges')
    plt.title('prediction')
    plt.axis('off')
    plt.savefig(f'{S_SAVE_PATH2}/{idx}.png', dpi=DPI)
    plt.cla()

if __name__ == '__main__':
    slide_total_list = [S_HGH_LIST, S_KBSMC_LIST, S_SS_LIST, S_ISH_LIST, S_KUMC_LIST]
    for slide_list in slide_total_list:
        for i, slide_path in enumerate(slide_list):
            save_slide_image(slide_path, i)

    for idx, gt_path in enumerate(GT_LIST):
        save_slide_results(idx, gt_path)
