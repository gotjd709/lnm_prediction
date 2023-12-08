from sklearn.model_selection          import GridSearchCV
from sklearn.metrics                  import plot_confusion_matrix
from configs                          import *
import matplotlib.pyplot              as plt
import sklearn.metrics                as metrics
import pandas                         as pd
import os


def model2grid(model, X_train, y_train, X_test_list, y_test_list, param_grid, info, method, cv=5):
    model = GridSearchCV(model, param_grid=param_grid, \
                        scoring='roc_auc', \
                        cv=cv, verbose=2, n_jobs=5)

    model.fit(X_train, y_train)
    
    params = model.cv_results_['params']
    score = model.cv_results_['mean_test_score']
    results = pd.DataFrame(params)
    results['score'] = score
    results = results.sort_values('score', ascending=False)
    
    name_list = ['TOTAL', 'HGH', 'ISH', 'KUMC', 'KBSMC', 'SS']
    
    for i in range(len(X_test_list)):
        viz_results(model, info, method, name_list[i], X_test_list[i], y_test_list[i])

def viz_results(model, info, method, name, X_test, y_test):

    label = ['LM0', 'LM1']
    plot = plot_confusion_matrix(model,
                                X_test, y_test,
                                display_labels=label,
                                cmap=plt.cm.Blues,
                                normalize='true')
    plot.ax_.set_title('Confusion Matrix')
    os.makedirs(f'{SAVE_PATH}/{info}/{method}/confusion_matrix/', exist_ok = True)
    plt.savefig(f'{SAVE_PATH}/{info}/{method}/confusion_matrix/{name}.png', dpi=DPI)
    plt.cla()

    metrics.plot_roc_curve(model, X_test, y_test)
    os.makedirs(f'{SAVE_PATH}/{info}/{method}/roc_curve/', exist_ok = True)
    plt.savefig(f'{SAVE_PATH}/{info}/{method}/roc_curve/{name}.png', dpi=DPI)
    plt.cla()