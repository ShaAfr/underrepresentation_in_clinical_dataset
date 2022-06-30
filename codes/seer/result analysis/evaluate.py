import numpy as np
import pandas as pd
import os

from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, roc_auc_score, roc_curve, auc, \
    mean_absolute_error, precision_score, recall_score, classification_report, confusion_matrix,\
    balanced_accuracy_score, precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.isotonic import IsotonicRegression as IR
from sklearn.linear_model import LogisticRegression as LR


def platt(preds, labels, test_preds):
    """
    preds: validation preds
    labels: validation labels
    test_preds: test preds
    
    validation set used to fit the calibrater
    return the calibrated validation and test probabilities
    """
    preds = np.array(preds)
    labels = np.array(labels)
    test_preds = np.array(test_preds)

    lr = LR()                                                       
    lr.fit( preds.reshape( -1, 1 ), labels )     # LR needs X to be 2-dimensional, preds -- X_train, labels -- y_train

    p_calibrated_v = lr.predict_proba( preds.reshape( -1, 1 ))[:,1]
    p_calibrated_t = lr.predict_proba( test_preds.reshape( -1, 1 ))[:,1]
    return p_calibrated_v, p_calibrated_t
    
 
def isotonic(preds, labels, test_preds):
    """
    preds: validation preds
    labels: validation labels
    test_preds: test preds
    
    validation set used to fit the calibrater
    return the calibrated validation and test probabilities
    """
    preds = np.array(preds)
    labels = np.array(labels)
    test_preds = np.array(test_preds)

    ir = IR(out_of_bounds='clip')
    ir.fit( preds, labels )

    p_calibrated_v = ir.transform( preds )
    p_calibrated_t = ir.transform( test_preds )   # or ir.fit( p_test ), that's the same thing

    return p_calibrated_v, p_calibrated_t
    
    
def calibrate_results(folder_name, epochs):
    for e in range(1, epochs+1):
        df_valid = pd.read_csv(folder_name + '/valid_results_data_frame_' + str(e) + '_epo.csv')
        df_test = pd.read_csv(folder_name + '/test_results_data_frame_' + str(e) + '_epo.csv')

        # calibration 
        #platt_calibrated = platt(df['score y'], df['true y'], df_test_test['score y'])
        isotonic_calibrated = isotonic(df_valid['score y'], df_valid['true y'], df_test['score y'])

        # calibrated validation
        #df_valid['score y'] = platt_calibrated[0]
        #df_valid['predict y'] = (platt_calibrated[0] >= 0.5).astype(np.int)
        #df_valid.to_csv(folder_name + '/test_results_data_frame_platt.csv', index=False)

        os.makedirs(folder_name + '/epo ' + str(e), exist_ok=True)

        df_valid['score y'] = isotonic_calibrated[0]
        df_valid['predict y'] = (isotonic_calibrated[0] >= 0.5).astype(np.int)
        df_valid.to_csv(folder_name + '/epo ' + str(e) + '/valid_results_data_frame_isotonic.csv', index=False)

        # calibrated test
        #df_test['score y'] = platt_calibrated[1]
        #df_test['predict y'] = (platt_calibrated[1] >= 0.5).astype(np.int)
        #df_test.to_csv(folder_name + '/test_test_results_data_frame_platt.csv', index=False)

        df_test['score y'] = isotonic_calibrated[1]
        df_test['predict y'] = (isotonic_calibrated[1] >= 0.5).astype(np.int)
        df_test.to_csv(folder_name + '/epo ' + str(e) + '/test_results_data_frame_isotonic.csv', index=False)

   
def write_results_to_file(results_list, filename):
    df_to_write = pd.DataFrame(data=results_list, \
                               index=['Whole', 'Gender_Male', 'Gender_Female',\
                                     'Ethnicity_White', 'Ethnicity_Black',\
                                     'Ethnicity_Hispanic', 'Ethnicity_Asian',\
                                     'Age<30', '30<=Age<40', '40<=Age<50',\
                                     '50<=Age<60', '60<=Age<70',\
                                     '70<=Age<80', '80<=Age<90', 'Age>=90'],\
                              columns=['Test_data', 'Accuracy', 'Balanced_Accuracy',\
                                      'Precision_C1', 'Precision_C0', 'Recall_C1',\
                                      'Recall_C0', 'F1_C1', 'F1_C0', 'AUC_ROC', 'FPR',\
                                      'FNR', 'AUC_PR_C1', 'AUC_PR_C0'])
    #print(df_to_write)
    df_to_write.to_csv(filename)
    
    
def evaluate(true_y, pred_y, score_y):
    res = []
    res.append(true_y.shape[0]) # test cases
    res.append(accuracy_score(true_y, pred_y)) # acc
    res.append(balanced_accuracy_score(true_y, pred_y)) 
    
    tn, fp, fn, tp = confusion_matrix(true_y, pred_y, labels=[0,1]).ravel()
    recall0 = tn / (tn + fp)
    precision0 = tn / (tn + fn)
    
    res.append(precision_score(true_y, pred_y))
    res.append(precision0)
    res.append(recall_score(true_y, pred_y))
    res.append(recall0)
    
    res.append(f1_score(true_y, pred_y))
    res.append(2 * precision0 * recall0 / (precision0 + recall0))
    
    try:
        res.append(roc_auc_score(true_y, score_y))
    except:
        res.append(0)
    
    res.append(fp / (fp + tn))
    res.append(fn / (fn + tp))
    
    try:
        curve_precision, curve_recall, _ = precision_recall_curve(true_y, score_y)
        res.append(auc(curve_recall, curve_precision))
    except:
        res.append(0)

    true_y_filp = ((true_y) == 0).astype(np.int)
    if (score_y < 0).sum() > 0: # negative value, logr decision scores
        score_y_filp = np.negative(score_y)
    else: # mlp scores
        score_y_filp = (1-score_y)
    
    try:
        curve_precision0, curve_recall0, _ = precision_recall_curve(true_y_filp, score_y_filp)
        res.append(auc(curve_recall0, curve_precision0))
    except:
        res.append(0)
    
    # num test cases, acc, balanced acc, precision 1 and 0, recall 1 and 0, f1 1 and 0, roc auc, FPR, FNR, auc pr 1 and 0
    return res
    
    
    
def generate_all_threshold_for_subgroups(folder_name, valid_or_test):
    """
    after generating files for each threshold, generate files for a subgroup containing all thresholds
    
    folder_name: the directory containing the valid/test result files
    valid_or_test: to generate files for valid results or test results (pass 'valid' or 'test' as values)
    """
    
    subgroups = {'Whole': [], 'Gender_Male': [], 'Gender_Female': [], \
                 'Ethnicity_White': [], 'Ethnicity_Black': [],\
                 'Ethnicity_Hispanic': [], 'Ethnicity_Asian': [],\
                 'Age<30': [], '30<=Age<40': [], '40<=Age<50': [],\
                 '50<=Age<60': [], '60<=Age<70': [],\
                 '70<=Age<80': [], '80<=Age<90': [], 'Age>=90': []}
    
    thresholds = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, \
              0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    
    
    # extract results
    for t in thresholds:
        df = pd.read_csv(folder_name + '/' + valid_or_test + '_metrics_threshold_' + str(t) + '.csv')
        for s in subgroups:
            temp = []
            
            df1 = df.loc[df['Unnamed: 0'] == s]
            
            #temp.append(t)
            temp.append(df1['Recall_C1'].iloc[0])
            temp.append(df1['Precision_C1'].iloc[0])
            temp.append(df1['AUC_PR_C1'].iloc[0])
            temp.append(df1['F1_C1'].iloc[0])
            temp.append(df1['Recall_C0'].iloc[0])
            temp.append(df1['Precision_C0'].iloc[0])
            temp.append(df1['AUC_PR_C0'].iloc[0])
            temp.append(df1['F1_C0'].iloc[0])
            temp.append(df1['Accuracy'].iloc[0])
            temp.append(df1['Balanced_Accuracy'].iloc[0])
            temp.append(df1['AUC_ROC'].iloc[0])
            
            subgroups[s].append(temp) # threshold with metrics
            
    # write to file
    for s in subgroups:
        ss = s.replace('<', '_').replace('=', '_').replace('>', '_')
        cols = ['Rec_C1', 'Prec_C1', 'PR_C1', 'F1_C1', 'Rec_C0', 'Prec_C0', 'PR_C0', 'F1_C0', 'Acc', 'Bal_Acc', 'ROC']
        to_write = pd.DataFrame(data=subgroups[s], index=thresholds, columns=cols)
        to_write.to_csv(folder_name + '/' + valid_or_test + '_' + ss + '_all_thresholds.csv')
        
        
def evaluate_all_subgroup_all_threshold(df_results, cancer, folder_name, valid_or_test):
    """
    generate all metrics for all threshold, containing all subgroup results
    for one file (valid or test, for a single epoch/model)
    
    true_y: the label
    score_y: predicted score/probability
    folder_name: directory of the epoch/model, result file will be written to this dir
    """
    if cancer =='breast':
        # flip bcs classes for mlp models
        df_results['true y'] = (df_results['true y'] == 0).astype(np.int)
        df_results['score y'] = 1 - df_results['score y']

        
    """
    for whole group table
    takes a data frame contains all results
    returns a list with subgroups analysis
    """
    # divde into groups
    # sex groups, 0 - female, 1 - male
    male = df_results[df_results['Sex 1'] == 1]
    female = df_results[df_results['Sex 1'] == 0]
    # race groups
    # recode 1 - white, 2 - black, 4 - asian, origin recode 1 - hispanic
    white = df_results[df_results['Race recode Y 1'] == 1]
    black = df_results[df_results['Race recode Y 2'] == 1]
    hispanic = df_results[df_results['Origin Recode NHIA 1'] == 1]
    asian = df_results[df_results['Race recode Y 4'] == 1]
    # age groups
    _30_and_below = df_results[df_results['Age at diagnosis continuous'] < 30]
    _31_to_40 = df_results[(df_results['Age at diagnosis continuous'] >= 30) & \
                                   (df_results['Age at diagnosis continuous'] < 40)]
    _41_to_50 = df_results[(df_results['Age at diagnosis continuous'] >= 40) & \
                                   (df_results['Age at diagnosis continuous'] < 50)]
    _51_to_60 = df_results[(df_results['Age at diagnosis continuous'] >= 50) & \
                                   (df_results['Age at diagnosis continuous'] < 60)]
    _61_to_70 = df_results[(df_results['Age at diagnosis continuous'] >= 60) & \
                                   (df_results['Age at diagnosis continuous'] < 70)]
    _71_to_80 = df_results[(df_results['Age at diagnosis continuous'] >= 70) & \
                                   (df_results['Age at diagnosis continuous'] < 80)]
    _81_to_90 = df_results[(df_results['Age at diagnosis continuous'] >= 80) & \
                                   (df_results['Age at diagnosis continuous'] < 90)]
    _90_and_above = df_results[df_results['Age at diagnosis continuous'] >= 90]
    
    group_list = [df_results, male, female, white, black, hispanic, asian, _30_and_below,\
                  _31_to_40, _41_to_50, _51_to_60, _61_to_70, _71_to_80,\
                 _81_to_90, _90_and_above]
    
    thresholds = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, \
              0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    
    for t in thresholds:
        results_list = []
        for g in group_list:
            new_pred_y = (np.array(g['score y']) >= t).astype(np.int)
            results_list.append(evaluate(np.array(g['true y']),\
                                         new_pred_y,\
                                         np.array(g['score y'])))
    
        
        write_results_to_file(results_list, folder_name + '/' + valid_or_test + '_metrics_threshold_' + str(t) + '.csv')
        
    
    generate_all_threshold_for_subgroups(folder_name, valid_or_test)
    #generate_all_threshold_for_subgroups(folder_name, 'test')
    
    # return
    # nothing need to be returned, results are written to files
    
    
def threshold_selection_F1_C1_Bal_Acc(filename):
    """
    filename: the file containing all thresholds
    
    select top 3 F1_C1 then top 1 balanced accuracy
    
    return the selected threshold and corresponding metrics
    """
    df = pd.read_csv(filename)
    # top 3 F1_C1
    df_top = df.nlargest(3, 'F1_C1')
    # top 1 bal_acc
    df_selected = df_top.nlargest(1, 'Bal_Acc')
    # covert to list
    results = df_selected.values[0]
    # return threshold and metrics
    return results[0], results[1:]
    
    
def metrics_for_a_threshold(filename, thre):
    """
    filename: the file containing all threshold
    thre: a threshold given
    
    return the metrics using the given threshold
    """
    df = pd.read_csv(filename)
    df_selected = df.loc[df['Unnamed: 0'] == thre]
    #print(df_selected)
    results = df_selected.values[0]
    # return threshold and metrics
    return results[0], results[1:]
    
    
def generate_everything_all_epochs(folder_name, epochs, cancer):
    """
    evalaute on calibrated results!!!
    """
    
    for e in range(1, epochs+1):
        # 1 generate both valid and test results for later use
        df_valid = pd.read_csv(folder_name + '/epo ' + str(e) + '/valid_results_data_frame_isotonic.csv')
        evaluate_all_subgroup_all_threshold(df_valid, cancer, folder_name + '/epo ' + str(e), 'valid')
        
        df_test = pd.read_csv(folder_name + '/epo ' + str(e) + '/test_results_data_frame_isotonic.csv')
        evaluate_all_subgroup_all_threshold(df_test, cancer, folder_name + '/epo ' + str(e), 'test')
        
        # 2 pick whole group and subgroup threshold
        # generate summary files, one for each threhsold

        results_whole_valid = []
        results_whole_test = []
        
        selected_sub_thre = []
        results_sub_valid = []
        results_sub_test = []
        
        # whole group
        whole_thre, whole_performance_valid = threshold_selection_F1_C1_Bal_Acc(folder_name + '/epo ' + str(e) + '/valid_Whole_all_thresholds.csv')
        _, whole_performance_test = metrics_for_a_threshold(folder_name + '/epo ' + str(e) + '/test_Whole_all_thresholds.csv', whole_thre)
        
        selected_sub_thre.append(whole_thre)
        results_sub_valid.append(whole_performance_valid)
        results_sub_test.append(whole_performance_test)
        
        results_whole_valid.append(whole_performance_valid)
        results_whole_test.append(whole_performance_test)
        
        # subgroups
        subgroups = ['Gender_Male', 'Gender_Female', 'Ethnicity_White', 'Ethnicity_Black',\
                 'Ethnicity_Hispanic', 'Ethnicity_Asian',\
                 'Age<30', '30<=Age<40', '40<=Age<50',\
                 '50<=Age<60', '60<=Age<70',\
                 '70<=Age<80', '80<=Age<90', 'Age>=90']
        for s in subgroups:
            ss = s.replace('<', '_').replace('=', '_').replace('>', '_')
            
            sub_thre, sub_performance_valid = threshold_selection_F1_C1_Bal_Acc(folder_name + '/epo ' + str(e) + '/valid_' + ss + '_all_thresholds.csv')
            _, sub_performance_test = metrics_for_a_threshold(folder_name + '/epo ' + str(e) + '/test_' + ss + '_all_thresholds.csv', sub_thre)
            selected_sub_thre.append(sub_thre)
            results_sub_valid.append(sub_performance_valid)
            results_sub_test.append(sub_performance_test)
            
            _, sub_performance_on_whole_valid = metrics_for_a_threshold( \
                folder_name + '/epo ' + str(e) + '/valid_' + ss + '_all_thresholds.csv', whole_thre)
            _, sub_performance_on_whole_test = metrics_for_a_threshold( \
                folder_name + '/epo ' + str(e) + '/test_' + ss + '_all_thresholds.csv', whole_thre)
            results_whole_valid.append(sub_performance_on_whole_valid)
            results_whole_test.append(sub_performance_on_whole_test)
            
        # subgroup threshold summary file
        rows = ['Whole', 'Gender_Male', 'Gender_Female', 'Ethnicity_White', 'Ethnicity_Black', 'Ethnicity_Hispanic', 'Ethnicity_Asian',\
                'Age<30', '30<=Age<40', '40<=Age<50', '50<=Age<60', '60<=Age<70', '70<=Age<80', '80<=Age<90', 'Age>=90']
        cols = ['Rec_C1', 'Prec_C1', 'PR_C1', 'F1_C1', 'Rec_C0', 'Prec_C0', 'PR_C0', 'F1_C0', 'Acc', 'Bal_Acc', 'ROC']
        
        to_write_sub_valid = pd.DataFrame(data=results_sub_valid, index=rows, columns=cols)
        to_write_sub_valid['threshold'] = selected_sub_thre
        
        to_write_sub_test = pd.DataFrame(data=results_sub_test, index=rows, columns=cols)
        to_write_sub_test['threshold'] = selected_sub_thre
        
        to_write_sub_valid.to_csv(folder_name + '/epo ' + str(e) + '/valid_subgroup_threshold_summary.csv')
        to_write_sub_test.to_csv(folder_name + '/epo ' + str(e) + '/test_subgroup_threshold_summary.csv')
        
        # whole group threshold summary file
        to_write_whole_valid = pd.DataFrame(data=results_whole_valid, index=rows, columns=cols)
        to_write_whole_valid['threshold'] = [whole_thre for i in range(len(rows))]
        
        to_write_whole_test = pd.DataFrame(data=results_whole_test, index=rows, columns=cols)
        to_write_whole_test['threshold'] = [whole_thre for i in range(len(rows))]
        
        to_write_whole_valid.to_csv(folder_name + '/epo ' + str(e) + '/valid_whole_group_threshold_summary.csv')
        to_write_whole_test.to_csv(folder_name + '/epo ' + str(e) + '/test_whole_group_threshold_summary.csv')
        
        
        

