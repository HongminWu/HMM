import sys
import os
import pandas as pd
import numpy as np
import shutil
import training_config
import generate_synthetic_data
import model_generation
import model_score
import util
import itertools
from sklearn.externals import joblib
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt

import ipdb

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    print classes
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_anomaly_by_label():
    X_train, X_test, y_train, y_test, class_names = [], [], [], [], []
    for X, y, fo, file_names in data_generator():
        X=np.concatenate(X)
        per_anomaly_len  = len(X)/len(y)
        Xdf = pd.DataFrame(X, columns=training_config.interested_data_fields)
        ax = Xdf.plot(subplots=True, legend=True, title=fo)
        for i in range(len(ax)):
            ax[i].set_xlabel('Time')
            for xc in range(per_anomaly_len, len(X), per_anomaly_len):
                ax[i].axvline(x=xc, color='k', linestyle='--')
                ax[i].legend(loc='best', framealpha=0.2)
    plt.show()

def get_all_anomalies_data_with_label():
    anomaly_data_path = training_config.anomaly_data_path    
    folders = os.listdir(anomaly_data_path)
    X, y, class_names= [], [], []
    for fo in folders:
        path = os.path.join(anomaly_data_path, fo)
        if not os.path.isdir(path):
            continue
        anomaly_data_group_by_folder_name = util.get_anomaly_data_for_labelled_case(training_config, path)
        temp = anomaly_data_group_by_folder_name.values()
        for i in range(len(temp)):
            X.append(temp[i][1])
            y.append(fo)
        class_names.append(fo)
    return X, y, class_names
    

def data_generator():
    anomaly_data_path = training_config.anomaly_data_path    
    folders = os.listdir(anomaly_data_path)
    for fo in folders:
        X, y= [],[]
        path = os.path.join(anomaly_data_path, fo)
        if not os.path.isdir(path):
            continue
        anomaly_data_group_by_folder_name = util.get_anomaly_data_for_labelled_case(training_config, path)
        temp = anomaly_data_group_by_folder_name.values()
        file_names = anomaly_data_group_by_folder_name.keys()
        for i in range(len(temp)):
            X.append(temp[i][1])
            y.append(fo)
        yield X, y, fo, file_names        

def predict_proba(X_test, class_names):
    # load trained anomaly models
    anomaly_model_group_by_label = {}
    anomaly_data_path = training_config.anomaly_data_path
    folders = os.listdir(anomaly_data_path)
    for fo in folders:
        anomaly_model_path = os.path.join(training_config.anomaly_model_save_path,
                                               fo,
                                               training_config.config_by_user['data_type_chosen'],
                                               training_config.config_by_user['model_type_chosen'],
                                               training_config.model_id)
        try:
            anomaly_model_group_by_label[fo] = joblib.load(anomaly_model_path + "/model_s%s.pkl"%(1,))
        except IOError:
            print 'anomaly model of  %s not found'%(fo,)
            ipdb.set_trace()
            raw_input("sorry! cann't load the anomaly model")
            continue

    predict_score = []
    calc_cofidence_resourse = []
    for i in range(len(X_test)):
        temp_loglik = []
        for model_label in class_names:
            one_log_curve_of_this_model = util.fast_log_curve_calculation(X_test[i], anomaly_model_group_by_label[model_label])
            temp_loglik.append(one_log_curve_of_this_model[-1])
        temp_score = temp_loglik / np.sum(temp_loglik)
        predict_score.append(temp_score)
    return np.array(predict_score)

def predict(X_test):
    # load trained anomaly models
    anomaly_model_group_by_label = {}
    anomaly_data_path = os.path.join(training_config.config_by_user['base_path'], 'all_anomalies')
    folders = os.listdir(anomaly_data_path)
    for fo in folders:
        anomaly_model_path = os.path.join(training_config.anomaly_model_save_path,
                                               fo,
                                               training_config.config_by_user['data_type_chosen'],
                                               training_config.config_by_user['model_type_chosen'],
                                               training_config.model_id)
        try:
            anomaly_model_group_by_label[fo] = joblib.load(anomaly_model_path + "/model_s%s.pkl"%(1,))
        except IOError:
            print 'anomaly model of  %s not found'%(fo,)
            ipdb.set_trace()
            raw_input("sorry! cann't load the anomaly model")
            continue

    predict_class = []
    calc_cofidence_resourse = []
    for i in range(len(X_test)):
        for model_label in anomaly_model_group_by_label:
            one_log_curve_of_this_model = util.fast_log_curve_calculation(X_test[i], 
                                                                        anomaly_model_group_by_label[model_label])
            calc_cofidence_resourse.append({
                'model_label': model_label,
                'culmulative_loglik': one_log_curve_of_this_model[-1],
                })
        sorted_list = sorted(calc_cofidence_resourse, key=lambda x:x['culmulative_loglik'])
        optimal_result = sorted_list[-1]
        classified_model = optimal_result['model_label']
        predict_class.append(classified_model)
    return predict_class



    
def fit(X, y, class_names):
    '''
    function: train all the anomalious models
    '''
    for model_name in class_names:
        indices = [i for i, label in enumerate(y) if label == model_name]
        train_data = [X[i] for i in indices]
        model_list, lengths = [], []
        for i in range(len(train_data)):
            lengths.append(len(train_data[i]))
        try:
            train_data = np.concatenate(train_data)
        except ValueError:
            print ('Oops!. something wrong...')
            ipdb.set_trace()
        lengths[-1] -=1
        model_generator = model_generation.get_model_generator(training_config.model_type_chosen, training_config.model_config)
        for model, now_model_config in model_generator:
            model = model.fit(train_data, lengths=lengths)  # n_samples, n_features
            score = model_score.score(training_config.score_metric, model, train_data, lengths)
            if score == None:
                print "scorer says to skip this model, will do"
                continue
            model_list.append({
                "model": model,
                "now_model_config": now_model_config,
                "score": score
            })
            print 'score:', score
            model_generation.update_now_score(score)        
        sorted_model_list = sorted(model_list, key=lambda x:x['score'])

        best = sorted_model_list[0]
        model_id = util.get_model_config_id(best['now_model_config'])

        anomaly_model_path = os.path.join(training_config.anomaly_model_save_path, 
                                                   model_name, 
                                                   training_config.config_by_user['data_type_chosen'], 
                                                   training_config.config_by_user['model_type_chosen'], 
                                                   training_config.model_id)
        
        if not os.path.isdir(anomaly_model_path):
            os.makedirs(anomaly_model_path)
            
        joblib.dump(
            best['model'],
            os.path.join(anomaly_model_path, "model_s%s.pkl"%(1,))
        )

def plot_roc_for_multiple_classes(X_test, y_test, class_names):
    from sklearn.preprocessing import label_binarize
    y_test = label_binarize(y_test, classes=class_names)
    n_classes = len(class_names)
    y_score = predict_proba(X_test, class_names)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    ##############################################################################
    # Plot of a ROC curve for a specific class
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


    ##############################################################################
    # Plot ROC curves for the multiclass problem

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

def run(if_plot):
    
    if if_plot:
        plot_anomaly_by_label()
        raw_input('Press any key to continue?')

    
    test_rate = 0.5
    X_train, X_test, y_train, y_test, class_names = [], [], [], [], []
    for X, y, fo,_ in data_generator():
        for i in range(int(len(y)*test_rate)):
            temp_X = X.pop()
            temp_y = y.pop()
            X_test.append(temp_X)
            y_test.append(temp_y)
        X_train.append(X)
        y_train.append(y)
        class_names.append(fo)
    X_train = np.concatenate(X_train).tolist()
    y_train = np.concatenate(y_train).tolist()
    

    '''
    X, y, class_names = get_all_anomalies_data_with_label()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=None, stratify=y)
    '''
    
    fit(X_train, y_train, class_names)

    # for plot roc
    plot_roc_for_multiple_classes(X_test, y_test, class_names)
    
    # for testing classifier
    y_pred = predict(X_test)
    print y_test
    print '\n'
    print y_pred

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred, labels=class_names)
    np.set_printoptions(precision=3)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.show()    

    
if __name__=='__main__':
    sys.exit(run(if_plot=False))
