from imblearn.under_sampling          import RandomUnderSampler
from imblearn.over_sampling           import RandomOverSampler
from xgboost                          import XGBClassifier
from sklearn.linear_model             import SGDClassifier
from sklearn.model_selection          import train_test_split
from lightgbm                         import LGBMClassifier
from configs                          import *
from utils                            import total_roc, feature_importance_xgb, feature_importance_sgd, feature_importance_lgb, false_negative
import pandas                         as pd

### Global Variables from config

if __name__ == '__main__':
    ##### load data #####

    hgh = pd.read_csv(R_HGH_PATH)
    hgh = hgh[((hgh['SMI'] >= 0) & (hgh['SMI'] <=1)) & ((hgh['PNI'] >= 0) & (hgh['PNI'] <=1)) & ((hgh['LVI'] >= 0) & (hgh['LVI'] <=1)) & ((hgh['LNM'] >= 0) & (hgh['LNM'] <=1)) & (hgh['SIZE'] != '-1')]
    hgh['MAD'] = [max(map(float, x[:-2].split(' x '))) for x in list(hgh[hgh['SIZE'] != '-1']['SIZE'])]

    ish = pd.read_csv(R_ISH_PATH)
    ish = ish[((ish['SMI'] >= 0) & (ish['SMI'] <=1)) & ((ish['PNI'] >= 0) & (ish['PNI'] <=1)) & ((ish['LVI'] >= 0) & (ish['LVI'] <=1)) & ((ish['LNM'] >= 0) & (ish['LNM'] <=1))& ((ish['SIZE'] != '-1') & (ish['SIZE'] != '0'))]
    ish['MAD'] = [max(map(float, x[:-2].split(' x '))) for x in list(ish[(ish['SIZE'] != '-1') & (ish['SIZE'] != '0')]['SIZE'])]

    kumc = pd.read_csv(R_KUMC_PATH)
    kumc = kumc[((kumc['SMI'] >= 0) & (kumc['SMI'] <=1)) & ((kumc['PNI'] >= 0) & (kumc['PNI'] <=1)) & ((kumc['LVI'] >= 0) & (kumc['LVI'] <=1)) & ((kumc['LNM'] >= 0) & (kumc['LNM'] <=1)) & (kumc['SIZE'] != '-1')]
    kumc['MAD'] = [max(map(float, x[:-2].split(' x '))) for x in list(kumc[kumc['SIZE'] != '-1']['SIZE'])]

    kbsmc = pd.read_csv(R_KBSMC_PATH)
    kbsmc = kbsmc[((kbsmc['SMI'] >= 0) & (kbsmc['SMI'] <=1)) & ((kbsmc['PNI'] >= 0) & (kbsmc['PNI'] <=1)) & ((kbsmc['LVI'] >= 0) & (kbsmc['LVI'] <=1)) & ((kbsmc['LNM'] >= 0) & (kbsmc['LNM'] <=1)) & (kbsmc['SIZE'] != '-1')]
    kbsmc['MAD'] = [max(map(float,x[:-2].split(' x '))) for x in list(kbsmc['SIZE'])]

    total = pd.concat([hgh, ish, kumc, kbsmc], axis=0)
    total = total.reset_index()

    # X feature, y feature extraction 
    hgh = total[total['ORGAN']=='hgh']
    hgh_X = hgh[['SMI','PNI','LVI','MAD']]
    hgh_y = hgh[['LNM']]

    ish = total[total['ORGAN']=='ish']
    ish_X = ish[['SMI', 'PNI', 'LVI','MAD']]
    ish_y = ish[['LNM']]

    kumc = total[total['ORGAN']=='kumc']
    kumc_X = kumc[['SMI', 'PNI', 'LVI','MAD']]
    kumc_y = kumc[['LNM']]

    kbsmc = total[total['ORGAN']=='kbsmc']
    kbsmc_X = kbsmc[['SMI', 'PNI', 'LVI','MAD']]
    kbsmc_y = kbsmc[['LNM']]

    # train, test dataset split
    hgh_X_train, hgh_X_test, hgh_y_train, hgh_y_test = train_test_split(hgh_X, hgh_y, test_size=0.2, random_state=SEED, stratify=hgh_y)
    ish_X_train, ish_X_test, ish_y_train, ish_y_test = train_test_split(ish_X, ish_y, test_size=0.2, random_state=SEED, stratify=ish_y)
    kumc_X_train, kumc_X_test, kumc_y_train, kumc_y_test = train_test_split(kumc_X, kumc_y, test_size=0.2, random_state=SEED, stratify=kumc_y)
    kbsmc_X_train, kbsmc_X_test, kbsmc_y_train, kbsmc_y_test = train_test_split(kbsmc_X, kbsmc_y, test_size=0.2, random_state=SEED, stratify=kbsmc_y)

    X_train = pd.concat([hgh_X_train, ish_X_train, kumc_X_train, kbsmc_X_train], axis=0)
    X_train = (X_train-X_train.mean())/X_train.std()
    y_train = pd.concat([hgh_y_train, ish_y_train, kumc_y_train, kbsmc_y_train], axis=0)

    X_test = pd.concat([hgh_X_test, ish_X_test, kumc_X_test, kbsmc_X_test])
    X_test = (X_test-X_test.mean())/X_test.std()
    y_test = pd.concat([hgh_y_test, ish_y_test, kumc_y_test, kbsmc_y_test])

    hgh_X_test = X_test.loc[hgh_X_test.index]
    ish_X_test = X_test.loc[ish_X_test.index]
    kumc_X_test = X_test.loc[kumc_X_test.index]
    kbsmc_X_test = X_test.loc[kbsmc_X_test.index]

    # test datasets list
    X_test_list = [X_test, hgh_X_test, ish_X_test, kumc_X_test, kbsmc_X_test]
    y_test_list = [y_test, hgh_y_test, ish_y_test, kumc_y_test, kbsmc_y_test]

    # under sampling & over sampling datasets
    undersample = RandomUnderSampler(sampling_strategy='majority', random_state=SEED)
    X_under, y_under = undersample.fit_resample(X_train, y_train)

    oversample = RandomOverSampler(sampling_strategy='minority', random_state=SEED)
    X_over, y_over = oversample.fit_resample(X_train, y_train)


    ##### analysis & visualize #####


    ### pathology

    # XGBoost
    xgb_clf = XGBClassifier(colsample_bytree=0.5, gamma=1, learning_rate=0.01, max_depth=7, reg_lambda=10, scale_pos_weight=1, subsample=0.8, random_state=SEED)
    xgb_clf.fit(X_under, y_under)

    method = 'XGBoost'
    total_roc(xgb_clf, PATHO_INFO, method, X_test_list, y_test_list)
    feature_importance_xgb(xgb_clf, X_under, PATHO_INFO, method)