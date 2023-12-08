from sklearn.metrics                  import roc_curve, roc_auc_score
from lightgbm                         import plot_importance
from configs                          import *
import matplotlib.pyplot              as plt
import pandas                         as pd
import cv2
import os

### Global Variables from config

def total_roc(model_clf, info, method, X_test_list, y_test_list):
    plt.figure(figsize=(15,5))
    plt.plot([0,1],[0,1],label='STANDARD')

    name_list = ['TOTAL_ROC', 'HGH_ROC', 'ISH_ROC', 'KUMC_ROC', 'KBSMC_ROC', 'SS_ROC']

    for (X_test, y_test, name) in zip(X_test_list, y_test_list, name_list):
        pred_positive_label = model_clf.predict_proba(X_test)[:,1]
        fprs, tprs, _       = roc_curve(y_test, pred_positive_label)
        plt.plot(fprs,tprs,label=f'{name} :: AUC {round(roc_auc_score(y_test, pred_positive_label),2)}')
    plt.legend()
    plt.grid()
    os.makedirs(f'{SAVE_PATH}/{info}/{method}/roc_curve/', exist_ok = True)
    plt.savefig(f'{SAVE_PATH}/{info}/{method}/roc_curve/total_merge.png', dpi=DPI)
    plt.cla()

def feature_importance_xgb(model_clf, X_train, info, method):
    value_list = [x for x in model_clf.feature_importances_]
    column_list = [x for x in X_train.columns.transpose()]
    zipped = zip(column_list, value_list)

    column_value = sorted(zipped, key=lambda x:x[1])
    column = [x[0] for x in column_value]
    value = [x[1] for x in column_value]

    coefs = pd.DataFrame(
        value, columns=['Feature Importance'], index=column
    )
    coefs.plot(kind='barh', figsize=(15,15))
    plt.title('XGBoost model')
    plt.axvline(x=0, color='.5')
    plt.subplots_adjust(left=.3)
    plt.legend(loc=4)
    os.makedirs(f'{SAVE_PATH}/{info}/{method}/feature_importance/', exist_ok = True)
    plt.savefig(f'{SAVE_PATH}/{info}/{method}/feature_importance/plot.png', dpi=DPI)
    plt.cla()

def feature_importance_sgd(model_clf, X_train, info, method):
    value_list = [x[0] for x in model_clf.coef_.transpose()]
    column_list = [x for x in X_train.columns.transpose()]
    zipped = zip(column_list, value_list)

    column_value = sorted(zipped, key=lambda x:x[1])
    column = [x[0] for x in column_value]
    value = [x[1] for x in column_value]

    coefs = pd.DataFrame(
        value, columns=['Coefficients'], index=column
    )
    coefs.plot(kind='barh', figsize=(15,15))
    plt.title('SGD model')
    plt.axvline(x=0, color='.5')
    plt.subplots_adjust(left=.3)
    plt.legend(loc=4)
    os.makedirs(f'{SAVE_PATH}/{info}/{method}/feature_importance/', exist_ok = True)
    plt.savefig(f'{SAVE_PATH}/{info}/{method}/feature_importance/plot.png', dpi=DPI)
    plt.cla()

def feature_importance_lgb(model_clf, info, method):
    plot_importance(booster=model_clf, figsize=(15,15))
    os.makedirs(f'{SAVE_PATH}/{info}/{method}/feature_importance/', exist_ok = True)
    plt.savefig(f'{SAVE_PATH}/{info}/{method}/feature_importance/plot.png', dpi=DPI)
    plt.cla()

def false_negative(tot, model_clf, X_test_list, y_test_list, info, method):
    X_test = X_test_list[0]
    y_test = y_test_list[0]
    y_pred = model_clf.predict(X_test)

    fn_name = []

    for (index, test, pred) in zip(y_test.index, y_test, y_pred):
        if test == 1 and pred == 0:
            slide_name = tot.loc[index]['SLIDE_NAME']
            fn_name.append(slide_name)

    for j, name in enumerate(fn_name):
        sl_path = f'{SLIDE_PATH}/{name}.png'
        pr_path = f'{PRED_PATH}/{name}.png'
        slide = cv2.imread(sl_path)
        pr_mask = cv2.imread(pr_path,0)
        plt.subplot(1,2,1)
        plt.imshow(slide)
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(pr_mask, vmin=0, vmax=3, cmap='Oranges')
        plt.axis('off')
        os.makedirs(f'{SAVE_PATH}/{info}/{method}/false_negative/', exist_ok = True)
        plt.savefig(f'{SAVE_PATH}/{info}/{method}/false_negative/{j}.png', dpi=DPI)