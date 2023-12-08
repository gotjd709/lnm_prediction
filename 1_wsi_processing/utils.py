from skimage                         import morphology, filters
from lxml                            import etree
from tqdm                            import trange
from configs                         import DEVICE
import numpy                         as np
import openslide
import torch
import cv2
import os



class WSIBase(object):
    '''
    Define basic methods needed to handle WSI
    '''
    def __init__(self, patch_size, resolution, down_level):
        self.patch_size = patch_size
        self.resolution = resolution
        self.down_level = down_level

    def _slide_setting(self, slide_path, anno_path, save_path):
        self.slide_path = slide_path
        self.anno_path  = anno_path
        self.save_path  = save_path

        self.slide = openslide.OpenSlide(self.slide_path)
        self.slide_name = self.slide_path.split('/')[-1].split('.')[0]

        self.level0_w, self.level0_h = self.slide.level_dimensions[0]
        self.adjust_term = 1 / float(self.slide.properties.get('openslide.mpp-x'))
        self.read_size = int(self.patch_size*self.adjust_term)

        self.level_min_w, self.level_min_h = self.slide.level_dimensions[self.down_level]
        self.level_min_img = np.array(self.slide.read_region((0,0), self.down_level if self.down_level > 0 else self.slide.level_count-1, size=(self.level_min_w, self.level_min_h)))
        self.zero2min = self.level0_h // self.level_min_h

    # get tissue mask method
    def get_tissue_mask(self, RGB_min=0):
        min_w, min_h = self.slide.level_dimensions[-1]
        min_img = np.array(self.slide.read_region((0,0), self.slide.level_count-1, size=(min_w, min_h)))
        hsv = cv2.cvtColor(min_img, cv2.COLOR_RGB2HSV)
        ## if more than threshold make Ture
        background_R = min_img[:, :, 0] > filters.threshold_otsu(min_img[:, :, 0])
        background_G = min_img[:, :, 1] > filters.threshold_otsu(min_img[:, :, 1])
        background_B = min_img[:, :, 2] > filters.threshold_otsu(min_img[:, :, 2])

        tissue_RGB = np.logical_not(background_R & background_G & background_B)
        tissue_S = hsv[:, :, 1] > filters.threshold_otsu(hsv[:, :, 1])

        min_R = min_img[:, :, 0] > RGB_min
        min_G = min_img[:, :, 1] > RGB_min
        min_B = min_img[:, :, 2] > RGB_min

        mask = tissue_S & (tissue_RGB + min_R + min_G + min_B)
        ret = morphology.remove_small_holes(mask, area_threshold=(min_h*min_w)//8)
        ret = np.array(ret).astype(np.uint8)
        
        kernel_size = 5
        tissue_mask = cv2.morphologyEx(ret*255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size)))  
        tissue_mask = cv2.resize(tissue_mask, (self.level_min_w, self.level_min_h), cv2.INTER_NEAREST)
        return tissue_mask

    # get sequence ratio method
    def get_seq_range(self, slide_width, slide_height, read_size, zero2two):
        y_seq = trange(int(((slide_height)) // int(read_size/zero2two)) + 1)
        x_seq = range(int(((slide_width)) // int(read_size/zero2two)) + 1)
        return y_seq, x_seq

    # get ratio mask method
    def get_ratio_mask(self, patch):
        h_, w_ = patch.shape[0], patch.shape[1]
        n_total = h_*w_
        n_cell = np.count_nonzero(patch)
        if (n_cell != 0):
            return n_cell*1.0/n_total*1.0
        else:
            return 0

    # save patch method
    def save_image(self, dir_path, file_name, img):
        os.makedirs(dir_path, exist_ok = True)
        cv2.imwrite(os.path.join(dir_path, file_name), img)



class ExtractPatch(WSIBase):
    '''
    Define methods to generate patches in WSI
    '''
    # init variable setting method
    def __init__(self, patch_size, resolution, down_level, tumor_ratio, tissue_ratio):
        super().__init__(
            patch_size  = patch_size,
            resolution  = resolution,
            down_level  = down_level           
        )
        self.tumor_ratio = tumor_ratio
        self.tissue_ratio  = tissue_ratio

    # save image patch & mask patch
    def execute_patch(self, patch_img, mask_img, patch_count, name, resolution):
        resize_image = cv2.resize(patch_img, (self.patch_size,self.patch_size), cv2.INTER_AREA)
        resize_mask = cv2.resize(mask_img.astype(np.uint8), (self.patch_size,self.patch_size), cv2.INTER_CUBIC).astype(np.uint8)
        
        self.save_image(self.save_path + f'/{self.slide_name}/input_x{str(resolution)}/', f'{patch_count}_{name}.png', resize_image)
        self.save_image(self.save_path + f'/{self.slide_name}/input_y{str(resolution)}/', f'{patch_count}_{name}.png', resize_mask)

    # make xml file to mask
    def asap2mask(self):
        bbox_dict = dict()
        anno_dict = dict()

        trees = etree.parse(self.anno_path).getroot()[0]
        for tree in trees:

            ## bbox coordinates extraction
            if str(tree.get('Type').split('_')[0]) == 'Rectangle':
                group = str(tree.get('Name')).split(' ')[0]
                regions = tree.findall('Coordinates')
                for region in regions:
                    coordinates = region.findall('Coordinate')
                    bbox = list()
                    for coord in coordinates:
                        x = round(float(coord.get('X')))
                        y = round(float(coord.get('Y')))
                        x = np.clip(round(x/self.zero2min), 0, self.level_min_w)
                        y = np.clip(round(y/self.zero2min), 0, self.level_min_h)
                        bbox.append((x, y))
                    if group in bbox_dict.keys():
                        bbox_dict[group].append(bbox)
                    else:
                        bbox_dict[group] = [bbox]

            elif (tree.get('Type').split('_')[0] == 'Spline'):
                if (tree.get('PartOfGroup') == 'Diff') or (tree.get('PartOfGroup').split('_')[0] == '01') or (tree.get('PartOfGroup').split('_')[0] == '02') or (tree.get('PartOfGroup').split('_')[0] == '06'):
                    group = 'tp_1'
                elif (tree.get('PartOfGroup') == 'Undiff') or (tree.get('PartOfGroup').split('_')[0] == '03') or (tree.get('PartOfGroup').split('_')[0] == '04') or (tree.get('PartOfGroup').split('_')[0] == '05') or (tree.get('PartOfGroup').split('_')[0] == '07') or (tree.get('PartOfGroup').split('_')[0] == '08'):
                    group = 'tp_2'
                else:
                    group = 'p_11'
                regions = tree.findall('Coordinates')

                # extract coordinate of x,y
                for region in regions:
                    coordinates = region.findall('Coordinate')
                    pts = list()
                    for coord in coordinates:
                        x = round(float(coord.get('X')))
                        y = round(float(coord.get('Y')))
                        x = np.clip(round(x/self.zero2min), 0, self.level_min_w)
                        y = np.clip(round(y/self.zero2min), 0, self.level_min_h)
                        pts.append((x, y))
                                        
                    if group in anno_dict.keys():
                        anno_dict[group].append(pts)
                    else:
                        anno_dict[group] = [pts]            

        if 'p_11' in anno_dict.keys():
            del anno_dict['p_11']
                
        # make masks
        tumor_mask = np.zeros((self.level_min_h, self.level_min_w))
        for key in anno_dict.keys():
            regions = anno_dict[key]
            for region in regions:
                pts = [np.array(region, dtype=np.int32)]
                tumor_mask = cv2.fillPoly(tumor_mask, pts, self.mask_tone(key))

        tissue_mask = self.get_tissue_mask()
        tissue_mask = np.where(tissue_mask>0, 1, 0)*1

        min_w, min_h = self.slide.level_dimensions[-1]

        total_mask = np.add(tissue_mask, tumor_mask)
        total_mask = cv2.resize(total_mask, (min_w, min_h), cv2.INTER_NEAREST)

        return bbox_dict, tumor_mask

    def return_color(self, classes):
        if classes == 1:
            return (0,255,0)
        elif classes == 2:
            return (255,0,0)

    def bbox_minslide(self, bbox):
        min_x = min(j[0] for j in bbox)
        max_x = max(j[0] for j in bbox)
        min_y = min(j[1] for j in bbox)
        max_y = max(j[1] for j in bbox)
        slide_w = max_x - min_x
        slide_h = max_y - min_y        
        return min_x, min_y, slide_w, slide_h

    def mask_tone(self, key):
        if key == 'tp_1':  
            return 1
        elif key == 'tp_2':
            return 2

    # extract patches corresponding with annoation mask method
    def extract(self):
        tissue_mask = self.get_tissue_mask()
        bbox_dict, tumor_mask = self.asap2mask()
        step = 1
        patch_count = 0


        min_x = 0; min_y = 0; slide_w = tumor_mask.shape[1]; slide_h = tumor_mask.shape[0]
        y_seq, x_seq = self.get_seq_range(slide_w, slide_h, self.read_size, self.zero2min)

        for y in y_seq:
            for x in x_seq:
                start_x = int(min_x + int(self.read_size/self.zero2min)*x)
                end_x = int(min_x + int(self.read_size/self.zero2min)*(x+step))
                start_y = int(min_y + int(self.read_size/self.zero2min)*y)
                end_y = int(min_y+ int(self.read_size/self.zero2min)*(y+step))

                tissue_mask_patch = tissue_mask[start_y:end_y, start_x:end_x]
                #sum_patch = np.zeros(((end_y - start_y), (end_x - start_x)))
                zero_patch = np.zeros(((end_y - start_y), (end_x - start_x)))
                name = ''
                
                if (self.get_ratio_mask(tissue_mask_patch) >= self.tissue_ratio):
                    img_patch = np.array(self.slide.read_region(
                        location = (int(start_x*self.zero2min), int(start_y*self.zero2min)),
                        level = 0,
                        size = (self.read_size, self.read_size)
                    )).astype(np.uint8)[...,:3]

                    tumor_patch = tumor_mask[start_y:end_y, start_x:end_x]
                    if (self.get_ratio_mask(tumor_patch) >= self.tumor_ratio):
                        tissue_patch = np.logical_or(tissue_mask_patch, zero_patch)*1
                        mix_patch = np.add(tumor_patch, tissue_patch)
                        tumor_cutoff_patch = (np.where(mix_patch==2,1,0) + np.where(mix_patch==3,2,0))
                        sum_patch = np.add(tumor_cutoff_patch, tissue_patch).astype(np.uint8)
                        name = '_tp_1' if np.max(sum_patch) == 2 else '_tp_2'
                        patch_count += 1 
                        self.execute_patch(img_patch, sum_patch, patch_count, name, self.resolution)

        # normal patch generation
        if len(bbox_dict) >= 1 and (('Normal' in bbox_dict.keys()) or ('normal' in bbox_dict.keys())):
            if 'Normal' in bbox_dict.keys():
                normal_bbox = bbox_dict['Normal']
            elif 'normal' in bbox_dict.keys():
                normal_bbox = bbox_dict['normal']           
            # normal_bbox = bbox_dict['Normal'] if 'Normal' in bbox_dict.keys() else bbox_dict['normal']

            for bbox in normal_bbox:
                min_x, min_y, slide_w, slide_h = self.bbox_minslide(bbox)
                y_seq, x_seq = self.get_seq_range(slide_w, slide_h, self.read_size, self.zero2min)

                for y in y_seq:
                    for x in x_seq:
                        start_x = int(min_x + int(self.read_size/self.zero2min)*x)
                        end_x = int(min_x + int(self.read_size/self.zero2min)*(x+step))
                        start_y = int(min_y + int(self.read_size/self.zero2min)*y)
                        end_y = int(min_y+ int(self.read_size/self.zero2min)*(y+step))

                        tissue_mask_patch = tissue_mask[start_y:end_y, start_x:end_x]
                        sum_patch = np.zeros(((end_y - start_y), (end_x - start_x)))
                        name = '_np_0'

                        if (self.get_ratio_mask(tissue_mask_patch) >= self.tissue_ratio):
                            img_patch = np.array(self.slide.read_region(
                                location = (int(start_x*self.zero2min), int(start_y*self.zero2min)),
                                level = 0,
                                size = (self.read_size, self.read_size)
                            )).astype(np.uint8)[...,:3]
                            # if self.blur_classifier(img_patch) > 300:     
                            normal = np.logical_or(tissue_mask_patch,sum_patch)*1
                            sum_patch = np.add(normal,sum_patch).astype(np.uint8)
                            patch_count += 1 
                            self.execute_patch(img_patch, sum_patch, patch_count, name, self.resolution)



class InferenceWSI(WSIBase):
    '''
    Define methods to infer WSI
    '''
    def __init__(self, patch_size, resolution, down_level):
        super().__init__(
            patch_size  = patch_size,
            resolution  = resolution,
            down_level  = down_level            
        )

    def get_image_patch(self, x, y):
        img_patch = np.array(self.slide.read_region(
            location = (x, y),
            level = 0,
            size = (self.read_size, self.read_size)
        )).astype(np.uint8)[...,:3]
        image_batch = cv2.resize(img_patch, (self.patch_size, self.patch_size), cv2.INTER_CUBIC)/255.0
        return image_batch

    def predict_patch(self, img_batch, model):
        x_tensor = torch.from_numpy(img_batch).float().to(DEVICE).unsqueeze(0).permute(0,3,1,2)               
        pr_mask = model(x_tensor).squeeze().cpu().detach().numpy().round()
        pr_mask = np.argmax(pr_mask, axis=0).astype(np.uint8)
        pr_mask = cv2.resize(pr_mask, (self.patch_size,self.patch_size), cv2.INTER_NEAREST) 
        return pr_mask

    def inference_patch(self, mask, x, y, tissue_mask, model, overlap):
        start_x = int(self.read_size*x) + int(self.read_size//2) if overlap else int(self.read_size*x)
        start_y = int(self.read_size*y) + int(self.read_size//2) if overlap else int(self.read_size*y)
        
        img_batch = self.get_image_patch(start_x, start_y)
        tissue_mask_patch = tissue_mask[start_y//self.zero2min:start_y//self.zero2min+self.read_size//self.zero2min, start_x//self.zero2min:start_x//self.zero2min+self.read_size//self.zero2min] 
        
        if self.get_ratio_mask(tissue_mask_patch) > 0.3:
            pr_mask = self.predict_patch(img_batch, model)
        else:
            pr_mask = np.zeros((self.patch_size,self.patch_size))
        try:
            mask[int(start_y//self.adjust_term):int(start_y//self.adjust_term+self.patch_size), int(start_x//self.adjust_term):int(start_x//self.adjust_term+self.patch_size)] = pr_mask
        except:
            pass
        return mask

    def processing_wsi(self, mask, mask_o):
        # overlapping
        mask = np.add(np.add(np.where(mask==2,3,0), np.where(mask==3,7,0)), np.where(mask==1,1,0))
        mask_o = np.add(np.add(np.where(mask_o==2,3,0), np.where(mask_o==3,7,0)), np.where(mask_o==1,1,0))
        overlap_mask = np.add(mask, mask_o)
        
        # remove nontissue region
        nontissue_region = cv2.resize(np.where(self.get_tissue_mask()<1,255,0), (int(self.level0_w/self.adjust_term), int(self.level0_h/self.adjust_term)), cv2.INTER_NEAREST)
        processed_mask = overlap_mask - nontissue_region
        processed_mask = np.add(np.add(np.where(processed_mask>0,1,0), np.where(processed_mask>2,1,0)), np.where(processed_mask>6,1,0)).astype(np.uint8)

        # resize to minimum level size
        resized_mask = cv2.resize(processed_mask, (self.level_min_w, self.level_min_h), cv2.INTER_NEAREST)     
        return resized_mask

    def inference(self, weight):        
        # get initial parameter setting
        y_seq = trange(int(int(self.level0_h) // (self.read_size)))
        x_seq = range(int(int(self.level0_w) // (self.read_size)))        

        # tissue_mask (level 2 size)
        tissue_mask = self.get_tissue_mask()

        # model setting
        model = torch.load(weight)

        # get zero mask, mask_o (level 2 size)
        mask = np.zeros((int(self.level0_h/self.adjust_term), int(self.level0_w/self.adjust_term)))
        mask_o = np.zeros((int(self.level0_h/self.adjust_term), int(self.level0_w/self.adjust_term)))      

        # inference patch
        for y in y_seq:
            for x in x_seq:
                mask = self.inference_patch(mask, x, y, tissue_mask, model, overlap=None)
                mask_o = self.inference_patch(mask_o, x, y, tissue_mask, model, overlap=True)

        # save postprocessed WSI
        processed_mask = self.processing_wsi(mask, mask_o)
        self.save_image(f'{self.save_path}/', f'{self.slide_name}.png', processed_mask)
