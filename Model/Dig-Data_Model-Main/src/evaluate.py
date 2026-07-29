from copy import deepcopy
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_score, confusion_matrix
import pandas as pd
# import cleanlab

def multiclass_acc(preds, truths):
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

def weighted_acc(preds, truths, verbose):
    preds = preds.view(-1)
    truths = truths.view(-1)

    total = len(preds)
    tp = 0
    tn = 0
    p = 0
    n = 0
    for i in range(total):
        if truths[i] == 0:
            n += 1
            if preds[i] == 0:
                tn += 1
        elif truths[i] == 1:
            p += 1
            if preds[i] == 1:
                tp += 1
    try:
        w_acc = (tp * n / p + tn) / (2 * n)
    except:
        w_acc = 0

    if verbose:
        fp = n - tn
        fn = p - tp
        recall = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        f1 = 2 * recall * precision / (recall + precision + 1e-8)
        print('TP=', tp, 'TN=', tn, 'FP=', fp, 'FN=', fn, 'P=', p, 'N', n, 'Recall', recall, "f1", f1)

    return w_acc

def eval_mosei_senti(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    acc7 = multiclass_acc(test_preds_a7, test_truth_a7)
    acc5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f1 = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)
    acc2 = accuracy_score(binary_truth, binary_preds)

    return mae, acc2, acc5, acc7, f1, corr


def eval_mosei_emo(preds, truths, best_thresholds=None, verbose=False):
    '''
    CMU-MOSEI Emotion is a multi-label classification task
    preds: (bs, num_emotions)
    truths: (bs, num_emotions)
    '''

    total = preds.size(0)
    num_emo = preds.size(1)

    preds = preds.cpu().detach()
    truths = truths.cpu().detach()
    truths = (truths > 0).float()

    preds = torch.sigmoid(preds)

    try:
        aucs = roc_auc_score(truths, preds, labels=list(range(num_emo)), average=None).tolist()
        aucs.append(np.average(aucs))
    except ValueError:
        aucs = [0,0,0,0,0,0,0]
        # pass

    # aucs = roc_auc_score(truths, preds, labels=list(range(num_emo)), average=None).tolist()
    # aucs.append(np.average(aucs))

    if best_thresholds is None:
        # select the best threshold for each emotion category, based on F1 score
        thresholds = np.arange(0.05, 1, 0.05)
        _f1s = []
        for t in thresholds:
            _preds = deepcopy(preds)
            _preds[_preds > t] = 1
            _preds[_preds <= t] = 0

            this_f1s = []

            for i in range(num_emo):
                pred_i = _preds[:, i]
                truth_i = truths[:, i]
                this_f1s.append(f1_score(truth_i, pred_i))

            _f1s.append(this_f1s)
        _f1s = np.array(_f1s)
        best_thresholds = (np.argmax(_f1s, axis=0) + 1) * 0.05
    print('best_thresholds: ', best_thresholds)
    best_thresholds = torch.tensor(best_thresholds)
    preds[preds > best_thresholds] = 1
    preds[preds <= best_thresholds] = 0
    # ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    accs = []
    f1s = []
    for emo_ind in range(num_emo):
        preds_i = preds[:, emo_ind]
        truths_i = truths[:, emo_ind]
        accs.append(weighted_acc(preds_i, truths_i, verbose=verbose))
        # f1s.append(f1_score(truths_i, preds_i, average='weighted'))
        f1s.append(f1_score(truths_i, preds_i, average='binary'))

    accs.append(np.average(accs))
    f1s.append(np.average(f1s))

    acc_strict = 0
    acc_intersect = 0
    acc_subset = 0
    for i in range(total):
        if torch.all(preds[i] == truths[i]):
            acc_strict += 1
            acc_intersect += 1
            acc_subset += 1
        else:
            is_loose = False
            is_subset = False
            for j in range(num_emo):
                if preds[i, j] == 1 and truths[i, j] == 1:
                    is_subset = True
                    is_loose = True
                elif preds[i, j] == 1 and truths[i, j] == 0:
                    is_subset = False
                    break
            if is_subset:
                acc_subset += 1
            if is_loose:
                acc_intersect += 1

    acc_strict /= total # all correct
    acc_intersect /= total # at least one emotion is predicted
    acc_subset /= total # predicted is a subset of truth
    print('accs: ', accs)
    print('f1s: ', f1s)
    # print('aucs: ', aucs)
    print('acc_strict: ', acc_strict)
    print('acc_subset: ', acc_subset)
    print('acc_intersect: ', acc_intersect)
    # return accs, f1s, aucs, [acc_strict, acc_subset, acc_intersect]
    return (accs, f1s, aucs), best_thresholds


def eval_iemocap(preds, truths, best_thresholds=None):
    # emos = ["Happy", "Sad", "Angry", "Neutral"]
    '''
    preds: (bs, num_emotions)
    truths: (bs, num_emotions)
    '''

    num_emo = preds.size(1)

    preds = preds.cpu().detach()
    truths = truths.cpu().detach()

    preds = torch.sigmoid(preds)

    aucs = roc_auc_score(truths, preds, labels=list(range(num_emo)), average=None).tolist()
    aucs.append(np.average(aucs))

    if best_thresholds is None:
        # select the best threshold for each emotion category, based on F1 score
        thresholds = np.arange(0.05, 1, 0.05)
        _f1s = []
        for t in thresholds:
            _preds = deepcopy(preds)
            _preds[_preds > t] = 1
            _preds[_preds <= t] = 0

            this_f1s = []

            for i in range(num_emo):
                pred_i = _preds[:, i]
                truth_i = truths[:, i]
                this_f1s.append(f1_score(truth_i, pred_i))

            _f1s.append(this_f1s)
        _f1s = np.array(_f1s)
        best_thresholds = (np.argmax(_f1s, axis=0) + 1) * 0.05
    print('best_thresholds: ', best_thresholds)
    # th = [0.5] * truths.size(1)
    for i in range(num_emo):
        pred = preds[:, i]
        pred[pred > best_thresholds[i]] = 1
        pred[pred <= best_thresholds[i]] = 0
        preds[:, i] = pred

    accs = []
    recalls = []
    precisions = []
    f1s = []
    for i in range(num_emo):
        pred_i = preds[:, i]
        truth_i = truths[:, i]

        # acc = weighted_acc(pred_i, truth_i, verbose=False)
        acc = accuracy_score(truth_i, pred_i)
        recall = recall_score(truth_i, pred_i)
        precision = precision_score(truth_i, pred_i)
        f1 = f1_score(truth_i, pred_i)

        accs.append(acc)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    accs.append(np.average(accs))
    recalls.append(np.average(recalls))
    precisions.append(np.average(precisions))
    f1s.append(np.average(f1s))

    return (accs, recalls, precisions, f1s, aucs), best_thresholds

def eval_iemocap_ce(preds, truths):
    # emos = ["Happy", "Sad", "Angry", "Neutral"]
    '''
    preds: (num_of_data, 4)
    truths: (num_of_data,)
    '''
    preds = preds.argmax(-1)
    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds, average='macro')
    r = recall_score(truths, preds, average='macro')
    p = precision_score(truths, preds, average='macro')
    return (acc, r, p, f1)

def test_iemocap_ce(preds, truths):
    # emos = ["Happy", "Sad", "Angry", "Neutral"]
    '''
    preds: (num_of_data, 4)
    truths: (num_of_data,)
    '''
    # pred_list = (np.argmax(preds, axis=1)).tolist()
    # truth_list = truths.numpy()
    # truths = truths.argmax(-1)
    preds = preds.argmax(-1)
    wrong_answer = pd.DataFrame(columns=['filename', 'predict', 'truth'])
    print('truth_list', truths)
    print('pred_list', preds)
    mat_confusion = confusion_matrix(truths, preds)
    print('mat_confusion: \n', mat_confusion)
        # 用 seaborn 畫
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    # return {'ang': 0, 'dis': 1, 'fea': 2, 'hap': 3, 'sad': 4, 'sur': 5}
    labels=['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    # labels=['anger', 'fear', 'happiness', 'sadness', 'surprise']
    # labels=['anger', 'happiness', 'sadness', 'surprise']
    # labels=['fear', 'sadness']
    # labels=['ang-fru-neu-sad', 'neu-sad']
    sns.heatmap(
        mat_confusion, xticklabels=labels, yticklabels=labels,
        annot=True, linewidths=0.1, fmt='d', cmap='YlGnBu')
    plt.title('Confusion Matrix', fontsize=15)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    plt.savefig('./savings/wrong_stat/confusion_matrix.png')

    # preds = preds.argmax(-1)
    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds, average='macro')
    r = recall_score(truths, preds, average='macro')
    p = precision_score(truths, preds, average='macro')
    return (acc, r, p, f1)


def test_iemocap(preds, truths, uttranceId, best_thresholds=None):
    # emos = ["Happy", "Sad", "Angry", "Neutral"]
    '''
    preds: (bs, num_emotions)
    truths: (bs, num_emotions)
    '''
    # print('preds: ', preds)
    pred_list = (np.argmax(preds, axis=1)).tolist()
    truth_list = (np.argmax(truths, axis=1)).tolist()
    wrong_answer = pd.DataFrame(columns=['filename', 'predict', 'truth'])

    mat_confusion = confusion_matrix(truth_list, pred_list)
    print('mat_confusion: \n', mat_confusion)
        # 用 seaborn 畫
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    labels=['anger', 'excited', 'frustrated', 'happiness', 'neutral', 'sadness']
    # labels=['anger', 'frustrated']
    # labels=['ang-fru', 'neu-sad']
    sns.heatmap(
        mat_confusion, xticklabels=labels, yticklabels=labels,
        annot=True, linewidths=0.1, fmt='d', cmap='YlGnBu')
    plt.title('Confusion Matrix', fontsize=15)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    plt.savefig('./savings/wrong_stat/confusion_matrix.png')

    print(type(truths), type(preds))
    a = truths.numpy()
    b = preds.numpy()
    _ = cleanlab.dataset.health_summary(truth_list, b, class_names=['anger', 'excited', 'frustrated', 'happiness', 'neutral', 'sadness'])

    emotion_trans = {0: 'anger', 1: 'excited', 2: 'frustrated', 3: 'happiness', 4: 'neutral', 5: 'sadness'}
    # emotion_trans = {0: 'anger', 1: 'frustrated'}
    # emotion_trans = {0: 'ang-fru', 1: 'neu-sad'}
    for i in range(0, len(pred_list)):
        if pred_list[i] != truth_list[i]:
            wrong_answer.loc[len(wrong_answer)] = {'filename': uttranceId[i], 'predict': emotion_trans[pred_list[i]], 'truth': emotion_trans[truth_list[i]]}
    wrong_answer.to_csv('./savings/wrong_stat/wrong.csv', index=False)
    print('wrong_answer: \n', wrong_answer)

    num_emo = preds.size(1)

    preds = preds.cpu().detach()
    truths = truths.cpu().detach()

    preds = torch.sigmoid(preds)
    aucs = roc_auc_score(truths, preds, labels=list(range(num_emo)), average=None).tolist()
    aucs.append(np.average(aucs))

    if best_thresholds is None:
        # select the best threshold for each emotion category, based on F1 score
        thresholds = np.arange(0.05, 1, 0.05)
        _f1s = []
        for t in thresholds:
            _preds = deepcopy(preds)
            _preds[_preds > t] = 1
            _preds[_preds <= t] = 0

            this_f1s = []

            for i in range(num_emo):
                pred_i = _preds[:, i]
                truth_i = truths[:, i]
                this_f1s.append(f1_score(truth_i, pred_i))

            _f1s.append(this_f1s)
        _f1s = np.array(_f1s)
        best_thresholds = (np.argmax(_f1s, axis=0) + 1) * 0.05

    # th = [0.5] * truths.size(1)
    for i in range(num_emo):
        pred = preds[:, i]
        pred[pred > best_thresholds[i]] = 1
        pred[pred <= best_thresholds[i]] = 0
        preds[:, i] = pred

    accs = []
    recalls = []
    precisions = []
    f1s = []
    for i in range(num_emo):
        pred_i = preds[:, i]
        truth_i = truths[:, i]
        # print('pred  ', i, ': ', pred_i)
        # print(neutral'truth ', i, ': ', truth_i)
        # acc = weighted_acc(pred_i, truth_i, verbose=False)
        acc = accuracy_score(truth_i, pred_i)
        recall = recall_score(truth_i, pred_i)
        precision = precision_score(truth_i, pred_i)
        f1 = f1_score(truth_i, pred_i)

        accs.append(acc)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    accs.append(np.average(accs))
    recalls.append(np.average(recalls))
    precisions.append(np.average(precisions))
    f1s.append(np.average(f1s))

    return (accs, recalls, precisions, f1s, aucs), best_thresholds