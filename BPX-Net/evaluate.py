'''
author: jun wang
copyright:hzcu
date:2025.05.20
'''
import numpy as np
import torch
import sklearn.metrics as skm
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import os.path as osp
import os

def start_evaluate(model, dataset, topr, categores, save_path, save_predictions=True, feature_names=None, device='cpu'):
    '''
    Runing test and calculating metrics for the classification tasks.     
    Inputs
    model: the deepomic-trained model to evaluate
    dataset: the dataset used to evaluate the model
    topr: only topr (0.0~1.0) features were fed into the model (remaining feature values are set to -1).
          the features are ranked via the ema coefficients of the deepomic-trained model
    categores: the category names of the classification task
    save_path: save evaluation results to the directory
    save_predictions: if set to True, saving ema coefficeints, individual coefficents and predicted labels
    feature_names: the features names saved the columns to the coefficents csv files
    Outputs: 
    metrics.csv, if set save_predictions to True: ema_coeffs.csv, individual_coeffs.csv, predictions.csv
             These csv files are saved to the save_path
    '''
    num_classes = len(categores)
    assert num_classes>=2, 'num_classes must ≥ 2!'
    os.makedirs(save_path, exist_ok=True)

    model.eval()
    batch_size= 512
    val_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1, persistent_workers=True, shuffle=False)
   
    mask_p = 1.0-F.softmax(torch.abs(model.ANFS.ema))

    pred_labels = []
    pred_probs = []
    pred_coeffs = []
    gt_labels = []
    
    for xs, ys in val_dataloader:
        xs = torch.tensor(xs, dtype=torch.float32).to(device)               
        ys = torch.tensor(ys, dtype=torch.long).to(device)
        masked_xs,_ = random_masking(xs, mask_p, topr, device)
        with torch.no_grad():
            coeffs, out_scores = model(masked_xs)
        
        pred_labels.append(torch.argmax(out_scores, dim=-1).detach().cpu().numpy())
        pred_probs.append(out_scores.detach().cpu().numpy())
        pred_coeffs.append(coeffs.detach().cpu().numpy())
        gt_labels.append(ys.detach().cpu().numpy())

    pred_labels = np.concatenate(pred_labels, axis=0)
    pred_probs = np.concatenate(pred_probs, axis=0)
    pred_coeffs = np.concatenate(pred_coeffs, axis=0)
    gt_labels = np.concatenate(gt_labels, axis=0)

    metrics = calc_metrics(gt_labels, pred_labels, pred_probs, num_classes)

    df = pd.DataFrame(metrics, index=['value', '95%CI_low', '95%CI_high']).T   
    
    df.to_csv(osp.join(save_path,'metrics.csv'))

    if save_predictions:      
        if feature_names is None:
            feature_names = ['feature_{o}'.format(i) for i in range(pred_coeffs.shape[1])]
        df = pd.DataFrame([model.ANFS.ema.detach().cpu().numpy()], columns=feature_names)
        df.to_csv(osp.join(save_path,'ema_coeffs.csv'), index=False)

        df = pd.DataFrame(pred_coeffs, columns=feature_names)
        df.to_csv(osp.join(save_path,'individual_coeffs.csv'), index=False)

        predictions = np.concatenate([pred_probs, pred_labels[...,None], gt_labels[...,None]],axis=-1)

        df = pd.DataFrame(predictions, columns=categores.extend(['pred labels', 'gt labels']))

        df.to_csv(osp.join(save_path,'predictions.csv'),index=False)

    return metrics

def random_masking(variables, mask_sampling_weights, topr=0.0, device='cpu'):
    '''
    variables: variables tensor (batch_size, num_features)
    mask_sampling_weights: sampling probabilities correspond to each num_features
    topr: control the number of kept variables for predction
    '''
  
    mask = torch.zeros_like(variables, dtype=torch.long, requires_grad=False).to(device)
    if topr>=1.0:
        return variables, mask  
   
    assert topr>=0.0, 'topr must ≥ 0.0'
    assert topr<=1.0, 'topr must ≤ 1.0'
    
    sorted_ind = torch.argsort(mask_sampling_weights)        
    n_keep = int(topr*variables.shape[1])
    
    #masking for each sample
    for i in range(variables.shape[0]):
        mask_ratio = np.random.uniform(0, 1.0)
        
        n_sample = int(mask_ratio*variables.shape[1])
        if n_sample==0: continue

        #sampling equally
        prob = torch.ones(mask_sampling_weights.size(0)) / mask_sampling_weights.size(0)
        position = torch.multinomial(prob, n_sample)
        mask[i,position] = 1

        #number of kept vital variables         
        mask[i,sorted_ind[0:n_keep]]= 0

    masked_variables = torch.where(mask==1, torch.tensor(-1.0).to(device), variables)#set masked variables to -1
    mask = torch.where(variables==-1, torch.tensor(0).to(device), mask)#ignore the variables with missing values
    return masked_variables, mask

def CI95_metrics_cls_task(pred_labels, pred_probs, ys, score_function, num_classes):
    '''
    pred_labels: the predicted labels
    pred_probs:  the predicted probablities
    ys: the ground-truth label
    score_function: the function used to calculate metrics
    '''
    alpha = 0.05 
    n_bootstraps = 100 
    scores = []   

    inds = np.array([i for i in range(ys.shape[0])])

    for _ in range(n_bootstraps):
        resampled_inds = np.random.choice(inds, size=len(inds), replace=True)
        resampled_preds = pred_labels[resampled_inds]
        resampled_pred_probs = pred_probs[resampled_inds]
        resampled_ys = ys[resampled_inds]
        if score_function == skm.roc_auc_score:
            if num_classes>2:
                scores.append(score_function(label_binarize(resampled_ys,classes=[0, 1, 2]), 
                                             resampled_pred_probs, average='macro', multi_class='ovr'))
            else:
                scores.append(score_function(resampled_ys, 
                                             resampled_pred_probs[:,-1]))
        elif score_function == skm.accuracy_score:
            scores.append(score_function(resampled_ys, resampled_preds))
        else:
            scores.append(score_function(resampled_ys, resampled_preds, average='macro'))


    lower_bound = np.percentile(scores, (alpha / 2) * 100)
    upper_bound = np.percentile(scores, (1 - alpha / 2) * 100)
    return lower_bound, upper_bound


def calc_metrics(gt_labels, pred_labels, pred_probs, num_classes):
    '''
    gt_labels:the groud-truth label
    pred_labels:the predicted label by model
    pred_probs:the predicted probabilities by model
    num_classes: the number of categories of the task
    '''
    accuracy = skm.accuracy_score(gt_labels, pred_labels)
    acc_low, acc_high = CI95_metrics_cls_task(pred_labels, pred_probs, gt_labels, skm.accuracy_score, num_classes)
    print('accuracy:', accuracy, '95%CI:', acc_low, acc_high)

    precision = skm.precision_score(gt_labels, pred_labels, average='macro')
    pre_low, pre_high = CI95_metrics_cls_task(pred_labels, pred_probs, gt_labels, skm.precision_score, num_classes)
    print('precision:', precision, '95%CI:', pre_low, pre_high )

    f1score = skm.f1_score(gt_labels, pred_labels, average='macro')
    f1_low, f1_high = CI95_metrics_cls_task(pred_labels, pred_probs, gt_labels, skm.f1_score, num_classes)
    print('f1score:', f1score, '95%CI:', f1_low,f1_high )

    recall = skm.recall_score(gt_labels, pred_labels, average='macro')
    rec_low, rec_high = CI95_metrics_cls_task(pred_labels, pred_probs, gt_labels, skm.recall_score, num_classes)
    print('recall:', recall, '95%CI:', rec_low, rec_high)

    if num_classes == 2:
        auc = skm.roc_auc_score(gt_labels, pred_probs[:, -1])        
    else:
        auc = skm.roc_auc_score(label_binarize(gt_labels,classes=[0, 1, 2]), 
                                             pred_probs, average='macro', multi_class='ovr')
    
    auc_low, auc_high = CI95_metrics_cls_task(pred_labels, pred_probs, gt_labels, skm.roc_auc_score, num_classes)
    print('auc:', auc, '95%CI:', auc_low, auc_high )

    metrics = {
        'accuracy': (accuracy, acc_low, acc_high),
        'precision': (precision, pre_low, pre_high),
        'f1score': (f1score, f1_low, f1_high),
        'recall': (recall, rec_low, rec_high),
        'auc': (auc, auc_low, auc_high)
        }
    
    return metrics


