from torch.utils.data                         import Dataset
from sklearn.utils                            import shuffle
import albumentations                         as A
import numpy                                  as np
import torch
import glob
import cv2


class PathSplit(object):
    def __init__(self, msk_path1, msk_path2, msk_path3, train_ratio, valid_ratio, test_ratio, random_state):
        self.msk_path1    = msk_path1
        self.msk_path2    = msk_path2
        self.msk_path3    = msk_path3
        self.train_len    = train_ratio
        self.valid_len    = train_ratio + valid_ratio
        self.test_len     = train_ratio + valid_ratio + test_ratio
        self.random_state = random_state

    def slice(self, start, end, msk_path):
        '''
        slice the msk_dir_path from start to end 
        '''
        msk_path = glob.glob(msk_path)
        # msk_path = shuffle(glob.glob(msk_path), random_state=self.random_state)
        slide_path = msk_path[int(start*len(msk_path)):int(end*len(msk_path))]
        #check_path = slide_path[0].split('/')[-3][:2] 
        return slide_path

    def zip(self, msk_path, img_dir_name='input_x100'):
        '''
        make a tuple list by zipping the mask directory path with the image direcotry path.
        ex) 
        msk_path1 = '.../input_y100/1_tp.png'
        img_path1 = '.../input_x100/1_tp.png'
        [msk_path1, msk_path2, ...] -> [(img_path1, msk_path1), (img_path2, msk_path2), ...] 
        '''
        return shuffle([('/'.join(x.split('/')[:-2])+f'/{img_dir_name}/'+x.split('/')[-1], x) for x in msk_path], random_state=self.random_state)

    def split(self):
        '''
        split msk_paths to train_img_msk_path, valid_img_msk_path, test_img_msk_path
        '''      
        train_msk_path = self.slice(0, self.train_len, self.msk_path1) + self.slice(0, self.train_len, self.msk_path2) + self.slice(0, self.train_len, self.msk_path3)
        valid_msk_path = self.slice(self.train_len, self.valid_len, self.msk_path1) + self.slice(self.train_len, self.valid_len, self.msk_path2) + self.slice(self.train_len, self.valid_len, self.msk_path3)
        test_msk_path = self.slice(self.valid_len, self.test_len, self.msk_path1) + self.slice(self.valid_len, self.test_len, self.msk_path2) + self.slice(self.valid_len, self.test_len, self.msk_path3)
        print('TRAIN TOTAL :', len(train_msk_path))
        print('VALID TOTAL :', len(valid_msk_path))
        print('TEST TOTAL :', len(test_msk_path))
        return self.zip(train_msk_path), self.zip(valid_msk_path), self.zip(test_msk_path)


class TensorData(Dataset):    
    def __init__(self, path_list, image_size, classes, augmentation=None):
        self.path_list = path_list
        self.image_size = image_size
        self.classes = classes
        self.augmentation = train_aug() if augmentation else test_aug()

    def get_labels(self):
        label_list = []
        for path in self.path_list:
            label_list.append(path[1].split('/')[-1][-8:-4])
        return label_list
        
    def __len__(self):
        return len(self.path_list)
        
    def __getitem__(self, index):
        batch_y = np.zeros(self.image_size + (self.classes,), dtype='uint8')
        img_path, mask_path = self.path_list[index]
        batch_x = cv2.imread(img_path).astype(np.float32)/255
        mask = cv2.imread(mask_path, 0)
        for i in range(self.classes):
            batch_y[...,i] = np.where(mask==i, 1, 0)
        sample = self.augmentation(image=batch_x, mask=batch_y)
        x_data, y_data = sample['image'], sample['mask']
        x_data = torch.FloatTensor(x_data)
        x_data = x_data.permute(2,0,1)
        y_data = torch.FloatTensor(y_data)
        y_data = y_data.permute(2,0,1)
        return x_data, y_data


def train_aug():
    ret = A.Compose(
        [   # transform & distortion augmentation
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=20, alpha_affine=15, interpolation=1, border_mode=1),
                A.GridDistortion(),
                A.OpticalDistortion(),
            ],p=0.8),
            # flip & rotate augmentation
            A.OneOf([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90(),
            ],p=1),
            ## color augmentation
            A.RandomBrightnessContrast(p=0.8),
        ]
    )
    return ret

def test_aug():
    ret = A.Compose(
        []
    )
    return ret