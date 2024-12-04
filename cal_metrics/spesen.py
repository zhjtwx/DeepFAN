from sklearn.metrics import roc_auc_score, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import numpy as np
def calc_metrics_for_fine_tuning_threshold(act, pred, verbose=False):
        epsilon = 1e-6
        tp = epsilon
        fp = epsilon
        fn = epsilon
        tn = epsilon
        act = list(act)
        pred = list(pred)
        for i in range(len(act)):
            if act[i] == 1 and pred[i] == 1:
                tp += 1
            elif act[i] == 1 and pred[i] == 0:
                fn += 1
            elif act[i] == 0 and pred[i] == 1:
                fp += 1
            else:
                tn += 1
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f = 2 * p * r / (p + r)
        acc = (tp + tn) / (fp + fn + tp + tn)
        sen = tp / (tp + fn)
        spe = tn / (tn + fp)
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
        if verbose:
            # print('        AUC\t', auc_val)
            print('Sensitivity\t', sen)
            print('Specificity\t', spe)
            print('         F1\t', f)
            print('  Precision\t', p)
            print('     Recall\t', r)
            print('   Accuracy\t', acc)
            print('        PPV\t', ppv)
            print('        NPV\t', npv)
        result = {}

        result['pre'] = p
        result['rec'] = r
        result['f1'] = f
        result['Sen'] = sen
        result['Spe'] = spe
        result['Acc'] = acc
        result['PPV'] = ppv #
        result['NPV'] = npv #
        return result
def tuning_threshold(y_label, y_score, threshold_find=0.99, threshold=None, verbose=True):
    choose_threshold = []
    y_score = y_score[:,1]
    y_label = y_label
    specificity_thred = None
    if threshold == None:
        fpr, tpr, threshold = metrics.roc_curve(y_label, y_score, drop_intermediate=False)
        min_threshold, max_threshold = min(threshold), max(threshold)
        thresholds = np.linspace(min_threshold, max_threshold, 500)[::-1]
        for index, thred in enumerate(thresholds):
            y_pred = (y_score >= thred).astype(int)
            current_result = calc_metrics_for_fine_tuning_threshold(y_label, y_pred, verbose=False)
            sensitivity, specificity = current_result['Sen'], current_result['Spe']
            choose_threshold.append((sensitivity, specificity, thred))
        for index in range(len(choose_threshold)):
            if index != 0:
                #spe
                if choose_threshold[index - 1][1] >= threshold_find and choose_threshold[index][1] <= threshold_find:
                    sensitivity_thred = (choose_threshold[index - 1][0], choose_threshold[index - 1][-1])
                #sen
                if choose_threshold[index - 1][0] <= threshold_find and choose_threshold[index][0] >= threshold_find:
                    specificity_thred = (choose_threshold[index][1], choose_threshold[index][-1])
        if specificity_thred == None:
            return {'threshold': threshold_find, 'sen': 0, 'spe': 0, 'sen_th': 0, 'spe_th': 0}
        if verbose == True:
            pass
            # print((sensitivity_thred, specificity_thred))
        result = {'threshold': threshold_find, 'sen': sensitivity_thred[0], 'spe': specificity_thred[0], 'sen_th': sensitivity_thred[1], 'spe_th': specificity_thred[1]}
        return result