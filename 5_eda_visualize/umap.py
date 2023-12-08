from sklearn.preprocessing             import QuantileTransformer
from sklearn.pipeline                  import make_pipeline
from sklearn.impute                    import SimpleImputer
from configs                           import *
import matplotlib.pyplot               as plt
import pandas                          as pd
import numpy                           as np
import pickle
import torch
import umap
import cv2
    
### Global Variables from config

class FeatureMap(object):
    def __init__(self, model, path_zips, max_per, min_per):
        self.model     = model
        self.encoder   = model.module.__dict__['_modules']['encoder']
        self.path_zips = path_zips
        self.max_per   = max_per
        self.min_per   = min_per

    def calc_pixel_percent(self, i, mask):
        return np.sum(np.where(mask==i,1,0))/(mask.shape[0]*mask.shape[1])
    
    def check_pixel_percent(self, mask):
        pixel_3_percent = self.calc_pixel_percent(3, mask)
        pixel_2_percent = self.calc_pixel_percent(2, mask)
        pixel_1_percent = self.calc_pixel_percent(1, mask)
        
        if pixel_3_percent > self.max_per:
            return 2
        elif pixel_2_percent > self.max_per and pixel_3_percent < self.min_per:
            return 1
        elif pixel_1_percent > self.max_per and pixel_2_percent < self.min_per and pixel_3_percent < self.min_per:
            return 0
        else:
            return -1
    
    def extract_image_feature(self, df, feature, mask, path_zip):
        gap_tensor  = torch.mean(feature.view(feature.size(0), feature.size(1), -1), dim=2).squeeze().cpu().detach().numpy()
        batch_array = [x for x in gap_tensor]
        batch_array.append(self.check_pixel_percent(mask))
        batch_array.append(path_zip)
        feature_df = pd.DataFrame(batch_array).transpose()
        df = pd.concat([df, feature_df], axis=0)
        return df
    
    def record_image_features(self, df_list, path_zip):
        image    = cv2.imread(path_zip[0])/255
        mask     = cv2.imread(path_zip[1],0)
        tensor   = torch.from_numpy(image).float().to('cuda').unsqueeze(0).permute(0,3,1,2)
        features = self.encoder(tensor)
        for i in range(5):
            df_list[i] = self.extract_image_feature(df_list[i], features[i+1], mask, path_zip)
        return df_list
            
    def make_df_list(self):
        df_list = [pd.DataFrame(list())]*5
        for path_zip in self.path_zips:
            df_list = self.record_image_features(df_list, path_zip)
        return df_list
    
    def get_target_df(self, target):
        df = self.make_df_list()[target]
        column_name = [f'feature_{x+1}' for x in range(df.shape[1]-2)]
        column_name.append('label')
        column_name.append('path_zip')
        df.columns = column_name
        df = df[df['label'] > -1]
        df = df.reset_index(drop=True)
        return df
        
    def get_label_index(self, df):
        t2_index = df[df['label']==2].index
        t1_index = df[df['label']==1].index
        no_index = df[df['label']==0].index        
        return no_index, t1_index, t2_index
        
    def get_x_reduced(self, df):
        no_index, t1_index, t2_index = self.get_label_index(df)
        
        feature_info = df.drop(['label', 'path_zip'], axis=1)
        label_info   = df[['label']].values.flatten()
        
        pipe = make_pipeline(SimpleImputer(strategy='mean'), QuantileTransformer())
        feature_info = pipe.fit_transform(feature_info.copy())
        manifold = umap.UMAP().fit(feature_info, label_info)
        X_reduced = manifold.transform(feature_info)
        
        
        t2_list = []
        t1_list = []
        no_list = []
        
        for i, x_reduced in enumerate(X_reduced):
            if i in t2_index:
                t2_list.append(x_reduced)
            elif i in t1_index:
                t1_list.append(x_reduced)
            elif i in no_index:
                no_list.append(x_reduced)
        
        return no_list, t1_list, t2_list
        
    def draw_umap(self, df):
        no_list, t1_list, t2_list = self.get_x_reduced(df)
        X_reduced_list = [np.array(no_list), np.array(t1_list), np.array(t2_list)]

        legend_dict = {0:['normal',(253/255,185/255, 125/255)], 1:['differentiated tumor',(233/255,93/255,13/255)], 2:['undifferentiated tumor',(127/255,39/255,4/255)]}
        
        for i, x_reduced in enumerate(X_reduced_list):
            plt.scatter(x_reduced[:,0], x_reduced[:,1], c=legend_dict[i][1], label=legend_dict[i][0], s=3)
        plt.legend(fontsize=8)
        plt.title('Test Patch UMAP')
        plt.savefig(f'{U_SAVE_PATH}/121_UMAP/test_patch_UMAP.png', dpi=DPI)
        plt.cla()

        return X_reduced_list
        
    def get_target_outlier(self, df, X_reduced_list, target, x_max, x_min, y_max, y_min, patch_results):
        target_list = X_reduced_list[target]
        
        target_except_index = []
        for i, xy in enumerate(target_list):
            if (x_max > xy[0]) and (x_min < xy[0]) and (y_max > xy[1]) and (y_min < xy[1]):
                target_except_index.append(i)
        
        except_path = []
        
        target_index = self.get_label_index(df)[target]
        for except_index in target_except_index:
            path_zip = df['path_zip'][target_index[except_index]]
            except_path.append(path_zip)
            image   = cv2.imread(path_zip[0])/255
            mask    = cv2.imread(path_zip[1],0)
            tensor  = torch.from_numpy(image).float().to(DEVICE).unsqueeze(0).permute(0,3,1,2)
            pr_mask = np.argmax((self.model(tensor)).squeeze().cpu().detach().numpy().round(), axis=0).astype(np.uint8)
            plt.subplot(1,3,1)
            plt.imshow(image)
            plt.axis('off')
            plt.title('patch image')
            plt.subplot(1,3,2)
            plt.imshow(mask, vmin=0, vmax=3, cmap='Oranges')
            plt.axis('off')
            plt.title('ground truth')
            plt.subplot(1,3,3)
            plt.imshow(pr_mask, vmin=0, vmax=3, cmap='Oranges')
            plt.axis('off')
            plt.title('prediction')
            name = '122_tumor1_normal' if patch_results == 'normal' else '123_tumor1_tumor2'
            plt.savefig(f'{U_SAVE_PATH}/{name}/{except_index}.png', dpi=DPI)
            plt.cla()

        return except_path


if __name__ == '__main__':

    model = torch.load(MODEL_PATH)

    with open(TEST_PICKLE, 'rb') as fr:
        path_zips = pickle.load(fr)

    path_zip = [(x.replace('input_y100','input_x100'),x) for x in TEST_PATH]
    path_zips.extend(path_zip)

    featuremap = FeatureMap(model, path_zips, MAX_PER, MIN_PER)
    target_df = featuremap.get_target_df(4)
    with open(PICKLE_PATH1, 'wb') as fw:
        pickle.dump(target_df, fw)

    with open(PICKLE_PATH1, 'rb') as fr:
        target_df = pickle.load(fr)

    X_reduced_list = featuremap.draw_umap(target_df)

    with open(PICKLE_PATH2, 'wb') as fw:
        pickle.dump(X_reduced_list, fw)
        
    with open(PICKLE_PATH2, 'rb') as fr:
        X_reduced_list = pickle.load(fr)

    featuremap.get_target_outlier(target_df, X_reduced_list, TARGET, X_MAX, X_MIN, Y_MAX, Y_MIN, True)