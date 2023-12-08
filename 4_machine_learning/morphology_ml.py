from imblearn.under_sampling          import RandomUnderSampler
from imblearn.over_sampling           import RandomOverSampler
from sklearn.model_selection          import train_test_split
from sklearn.linear_model             import SGDClassifier
from sklearn.cluster                  import AgglomerativeClustering
from group_lasso                      import LogisticGroupLasso
from lightgbm                         import LGBMClassifier
from catboost                         import CatBoostClassifier
from xgboost                          import XGBClassifier
from utils                            import model2grid
from configs                          import *
import pandas                         as pd

### Global Variables from config

if __name__ == '__main__':
    tot = pd.read_csv(CSV_PATH, encoding='cp949')

    # feature datatsets accroding to organization
    hgh = tot[tot['ORGANIZATION'] == 'HGH']
    hgh_X = hgh.iloc[:,:73]
    hgh_y = hgh.iloc[:,-1]
    ish = tot[tot['ORGANIZATION'] == 'ISH']
    ish_X = ish.iloc[:,:73]
    ish_y = ish.iloc[:,-1]
    kumc = tot[tot['ORGANIZATION'] == 'KUMC']
    kumc_X = kumc.iloc[:,:73]
    kumc_y = kumc.iloc[:,-1]
    kbsmc = tot[tot['ORGANIZATION'] == 'KBSMC']
    kbsmc_X = kbsmc.iloc[:,:73]
    kbsmc_y = kbsmc.iloc[:,-1]
    ss = tot[tot['ORGANIZATION'] == 'SS']
    ss_X = ss.iloc[:,:73]
    ss_y = ss.iloc[:,-1]


    # split datasets hierarchically
    hgh_X_train, hgh_X_test, hgh_y_train, hgh_y_test = train_test_split(hgh_X, hgh_y, test_size=0.2, random_state=SEED, stratify=hgh_y)
    ish_X_train, ish_X_test, ish_y_train, ish_y_test = train_test_split(ish_X, ish_y, test_size=0.2, random_state=SEED, stratify=ish_y)
    kumc_X_train, kumc_X_test, kumc_y_train, kumc_y_test = train_test_split(kumc_X, kumc_y, test_size=0.2, random_state=SEED, stratify=kumc_y)
    kbsmc_X_train, kbsmc_X_test, kbsmc_y_train, kbsmc_y_test = train_test_split(kbsmc_X, kbsmc_y, test_size=0.2, random_state=SEED, stratify=kbsmc_y)
    ss_X_train, ss_X_test, ss_y_train, ss_y_test = train_test_split(ss_X, ss_y, test_size=0.2, random_state=SEED, stratify=ss_y)

    X_train = pd.concat([hgh_X_train, ish_X_train, kumc_X_train, kbsmc_X_train, ss_X_train], axis=0)
    X_train = (X_train.iloc[:,3:73]-X_train.iloc[:,3:73].mean())/X_train.iloc[:,3:73].std()
    y_train = pd.concat([hgh_y_train, ish_y_train, kumc_y_train, kbsmc_y_train, ss_y_train], axis=0)

    X_test_label = pd.concat([hgh_X_test, ish_X_test, kumc_X_test, kbsmc_X_test, ss_X_test], axis=0)
    X_test = (X_test_label.iloc[:,3:73]-X_test_label.iloc[:,3:73].mean())/X_test_label.iloc[:,3:73].std()
    X_test_label = pd.concat([X_test_label.iloc[:,:3],X_test], axis=1)

    hgh_X_test = X_test_label[X_test_label['ORGANIZATION'] == 'HGH'].iloc[:,3:73]
    ish_X_test = X_test_label[X_test_label['ORGANIZATION'] == 'ISH'].iloc[:,3:73]
    kumc_X_test = X_test_label[X_test_label['ORGANIZATION'] == 'KUMC'].iloc[:,3:73]
    kbsmc_X_test = X_test_label[X_test_label['ORGANIZATION'] == 'KBSMC'].iloc[:,3:73]
    ss_X_test = X_test_label[X_test_label['ORGANIZATION'] == 'SS'].iloc[:,3:73]
    y_test = pd.concat([hgh_y_test, ish_y_test, kumc_y_test, kbsmc_y_test, ss_y_test], axis=0)


    # test datasets list
    X_test_list = [X_test, hgh_X_test, ish_X_test, kumc_X_test, kbsmc_X_test, ss_X_test]
    y_test_list = [y_test, hgh_y_test, ish_y_test, kumc_y_test, kbsmc_y_test, ss_y_test]


    # under sampling & over sampling datasets
    undersample = RandomUnderSampler(sampling_strategy='majority', random_state=SEED)
    X_under, y_under = undersample.fit_resample(X_train, y_train)

    oversample = RandomOverSampler(sampling_strategy='minority', random_state=SEED)
    X_over, y_over = oversample.fit_resample(X_train, y_train)


    # XGBoost, SGD(Stochastic Gradient Descent), LGB(Light Gradient Boosting), CatBoost, SGL(Sparse Group Lasso) model setting
    xgb_clf = XGBClassifier(random_state=SEED)
    sgd_clf = SGDClassifier(loss='log', penalty='elasticnet', max_iter=200, random_state=SEED, class_weight='balanced')
    lgb_clf = LGBMClassifier(random_state=SEED)
    cat_clf = CatBoostClassifier(random_state=SEED)
    X_trans = X_train.transpose()
    clustering = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='complete', compute_full_tree=True, distance_threshold=0.95).fit(X_trans)
    sgl_clf = LogisticGroupLasso(groups=list(clustering.labels_), old_regularisation=False, warm_start=True, random_state=SEED)


    # XGBoost
    method = 'XGBoost'
    model2grid(xgb_clf, X_train, y_train, X_test_list, y_test_list, XGB_GRID, MORPH_INFO, method, cv=5)
    model2grid(xgb_clf, X_over, y_over, X_test_list, y_test_list, XGB_GRID, MORPH_INFO, method, cv=5)
    model2grid(xgb_clf, X_under, y_under, X_test_list, y_test_list, XGB_GRID, MORPH_INFO, method, cv=5)


    # SGD
    method = 'SGD'
    model2grid(sgd_clf, X_train, y_train, X_test_list, y_test_list, SGD_GRID, MORPH_INFO, method, cv=5)
    model2grid(sgd_clf, X_over, y_over, X_test_list, y_test_list, SGD_GRID, MORPH_INFO, method, cv=5)
    model2grid(sgd_clf, X_under, y_under, X_test_list, y_test_list, SGD_GRID, MORPH_INFO, method, cv=5)


    # LGB
    method = 'LGB'
    model2grid(lgb_clf, X_train, y_train, X_test_list, y_test_list, LGB_GRID, MORPH_INFO, method, cv=5)
    model2grid(lgb_clf, X_over, y_over, X_test_list, y_test_list, LGB_GRID, MORPH_INFO, method, cv=5)
    model2grid(lgb_clf, X_under, y_under, X_test_list, y_test_list, LGB_GRID, MORPH_INFO, method, cv=5)


    # CatBoost
    method = 'CatBoost'
    model2grid(cat_clf, X_train, y_train, X_test_list, y_test_list, CAT_GRID, MORPH_INFO, method, cv=5)
    model2grid(cat_clf, X_over, y_over, X_test_list, y_test_list, CAT_GRID, MORPH_INFO, method, cv=5)
    model2grid(cat_clf, X_under, y_under, X_test_list, y_test_list, CAT_GRID, MORPH_INFO, method, cv=5)


    # SGL
    method = 'SGL'
    model2grid(sgl_clf, X_train, y_train, X_test_list, y_test_list, SGL_GRID, MORPH_INFO, method, cv=5)
    model2grid(sgl_clf, X_over, y_over, X_test_list, y_test_list, SGL_GRID, MORPH_INFO, method, cv=5)
    model2grid(sgl_clf, X_under, y_under, X_test_list, y_test_list, SGL_GRID, MORPH_INFO, method, cv=5)