from scipy.stats         import mode
from configs             import *
import matplotlib.pyplot as plt
import cv2

### Global Variables from config

def draw_each_color_space(patch_list, save_name):
    ax = plt.axes(projection = '3d')

    x = []
    y = []
    z = []
    c = []

    for patch_path in patch_list:
        patch = cv2.imread(patch_path)
        r_mode = mode(patch[...,0].flatten())[0].item()
        g_mode = mode(patch[...,1].flatten())[0].item()
        b_mode = mode(patch[...,2].flatten())[0].item()

        x.append(r_mode)
        y.append(g_mode)
        z.append(b_mode)
        c.append((r_mode/255, g_mode/255, b_mode/255))

    ax.scatter(x,y,z, c=c)
    ax.set_xlabel('R')
    ax.set_xlim([0,255])
    ax.set_ylabel('G')
    ax.set_ylim([0,255])
    ax.set_zlabel('B')
    ax.set_zlim([0,255])
    ax.set_title(f'{save_name} color distribution')
    plt.savefig(f'{D_SAVE_PATH}/{save_name}.png', dpi=DPI)
    ax.cla()

def draw_total_color_space(label_color_dict):
    plt.figure(figsize=(15,15))
    ax = plt.axes(projection = '3d')

    for i in label_color_dict.keys():
        x = []
        y = []
        z = []
        
        for patch_path in label_color_dict[i][1]:
            patch = cv2.imread(patch_path)
            r_mode = mode(patch[...,0].flatten())[0].item()
            g_mode = mode(patch[...,1].flatten())[0].item()
            b_mode = mode(patch[...,2].flatten())[0].item()

            x.append(r_mode)
            y.append(g_mode)
            z.append(b_mode)

        ax.scatter(x,y,z, c=label_color_dict[i][2], label=label_color_dict[i][0], s=3)
    ax.set_xlabel('R')
    ax.set_xlim([0,255])
    ax.set_ylabel('G')
    ax.set_ylim([0,255])
    ax.set_zlabel('B')
    ax.set_zlim([0,255])
    ax.set_title('total organization color distribution')
    ax.legend()
    plt.savefig(f'{D_SAVE_PATH}/total.png', dpi=DPI)
    plt.cla()
    ax.cla()

if __name__ == '__main__':
    total_list = [[D_HGH_LIST,'hgh'], [D_KBSMC_LIST,'kbsmc'], [D_SS_LIST,'ss'], [D_ISH_LIST,'ish'], [D_KUMC_LIST,'kumc']]
    total_list = [total_list[0]]
    for organ in total_list:
       draw_each_color_space(organ[0], organ[1])
       plt.cla()

    train_list = D_HGH_LIST+D_KBSMC_LIST+D_SS_LIST
    test_list = D_ISH_LIST+D_KUMC_LIST
    total_color_dict = {0:['Train Datasets',train_list,(197/255,65/255,2/255)], 1:['Test Datasets',test_list,(253/255,167/255,98/255)]}
    draw_total_color_space(total_color_dict)