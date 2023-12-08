from utils              import FeatureExtraction
from collections        import deque
from configs            import *
from tqdm               import tqdm
import pandas           as pd
import multiprocessing
import time
import glob
import cv2
import os

### Global Variables from config

def extract(pack, feature_df, column_name, PROPERTIES, ij):
    mask_path, slide_path, diag_id, adjust, lnm = pack 
    slide_name = '.'.join(slide_path.split('/')[-1].split('.')[:-1])
    organization = slide_path.split('/')[3]
    mask = cv2.imread(mask_path, 0)
    mask_feature = FeatureExtraction(mask, adjust)
    values = deque([mask_feature.__dict__[prop] for prop in PROPERTIES])
    values.appendleft(diag_id)
    values.appendleft(slide_name)
    values.appendleft(organization)
    values.append(lnm)
    df = pd.DataFrame(list(values))
    df = df.transpose()
    feature_df = pd.concat([feature_df, df], axis=0)
    df = feature_df.copy()
    df.columns = column_name
    df.to_csv(SAVE_CSV_PATH+'/temp_dir' + f'/{ij}_' + slide_name + '.csv')



if __name__ == '__main__':
    # setting
    df = pd.read_csv(PATH_TABLE_PATH, encoding='cp949')
    pack_list = [x for x in zip(df['mask_path'], df['slide_path'], df['DIAG_ID'], df['adjust'], df['LNM'])]
    os.makedirs(SAVE_CSV_PATH, exist_ok=True)

    # preprocessing
    column_name = deque(PROPERTIES.copy())
    column_name.appendleft('DIAG_ID')
    column_name.appendleft('SLIDE_NAME')
    column_name.appendleft('ORGANIZATION')
    column_name.append('LNM')
    feature_df = pd.DataFrame([])
    os.makedirs(SAVE_CSV_PATH+'/temp_dir', exist_ok = True)

    # process each feature
    for i in tqdm(range(0,len(pack_list), K)):
        pack_batch = pack_list[i:i+K]
        for j, pack in enumerate(pack_batch):
            try: 
                p = multiprocessing.Process(target=extract, args=(pack, feature_df, column_name, PROPERTIES, i+j, ))
                p.start()
            except:
                pass
        p.join()

    # wait rest feature extraction dataframes
    time.sleep(300)

    # concatenate features
    csv_list = sorted(glob.glob(SAVE_CSV_PATH+'/temp_dir/*.csv'))
    csv_final = pd.DataFrame([])
    for csv in csv_list:
        csv_content = pd.read_csv(csv)
        csv_final = pd.concat([csv_final, csv_content], axis=0)
        os.remove(csv)
    csv_final.to_csv(SAVE_CSV_PATH + f'/feature_results.csv')