'''
author: jun wang
copyright:hzcu
date:2025.06.12
'''
import torch
import pandas as pd
import numpy as np
import torch.utils
from datasets.csv_dataset import CsvDataset
from torch.utils.data import Subset
from sklearn.model_selection import KFold
from models.classifiers import ClassifierNames
from models.BPXNet import BPXNet
from trainers.classifier_trainers import ClassifierTrainer
import os.path as osp
from evaluate import start_evaluate, calc_metrics
import warnings
warnings.filterwarnings('ignore')

def build_fold_path(task, classifier, BPRD_gama, use_prior, fold):
    return '{0}_{1}_gama{2}_{3}_prior/fold{4}'.format(task, classifier, BPRD_gama, use_prior, fold)


if __name__ == '__main__':
    #define hyperparameters
    classifier = ClassifierNames.kan #option: transformer, kan, longformer, mamba, resnet18, resnet50
    BPRD_gama = 0.2#the ratio to keep biomarkers during dropout
    use_prior = 'expert' #option: expert, foundation, None
    categories = ['<12m', '>=12m']
    num_classes = len(categories)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  

    prior_root = 'datasets/priors/ccrcc'
    #loading prior weights to guide the training
    if use_prior is None: 
        prior_weights = None 
    else:
        prior_weights_file = 'expert_importance.csv' if use_prior=='expert' else 'foundation_importance.csv'
    
        prior_weights = pd.read_csv(osp.join(prior_root, prior_weights_file)).drop(columns=[],
                                                                                            axis=1)
        prior_weights = np.squeeze(np.array(prior_weights))   
        print('prior loaded!!!!')  

    #loading data samples, ignoire these columns in csv
    ignores = ['']
    
    maxv_for_norm = pd.read_csv(osp.join('datasets', 'MMIST-ccRCC-MaxValues.csv'), encoding='gbk').drop(columns=ignores, axis=1)
    maxv_for_norm.pop('vital_status_12')       
    maxv_for_norm = np.array(maxv_for_norm)[0,...]
    
    data_file = ['datasets/MMIST-ccRCC.csv']
    dataset = CsvDataset(data_file,
                                label_column='vital_status_12',
                                ignores=ignores,
                                max_norm=True, #maximum normalization
                                maxv_for_norm=maxv_for_norm,
                                repeat_fews=True)    
   
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (traind_ids, val_ids) in enumerate(kfold.split(dataset)):
        print('Folds:{0}/{1}'.format(fold+1, 5))
        train_subset = Subset(dataset, traind_ids)
        val_subset = Subset(dataset, val_ids)        

        model = BPXNet(dataset.getNumFeatures(),
                        bprd_gama = BPRD_gama,
                        fill_v=-1,
                        anfs_hiden_features=128,
                        anfs_out_activation='tanh',
                        anfs_ema_alpha=0.98,
                        classifier_name = classifier, 
                        classifier_layers = [dataset.getNumFeatures(), 128, 128, 128],
                        num_classes= num_classes,
                        prior_knowledge= prior_weights,
                        device=device)
        
        #define trainer for model training
        trainer = ClassifierTrainer(model, optimizer='adamw', batch_size=512, epochs=100, 
                                    lr=0.001, lr_decay_steps=50, use_focal_loss=True, device=device)
        
        ckpoint_save_path = 'checkpoints/'+ build_fold_path('ccrcc', classifier, BPRD_gama, use_prior, fold+1)

        trainer.start_train(train_subset, ckpoint_save_path, val_subset, 1)

        #start evaluation for model performance
        print('Start testing')
        model.load_state_dict(torch.load(osp.join(ckpoint_save_path, "best_weights.pth"), map_location=device), strict=True)
        for topr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            metric_save_path = 'results/' + build_fold_path('ccrcc', classifier, BPRD_gama, use_prior, fold+1) + '/topr{0}'.format(topr)
            start_evaluate(model, val_subset, topr, categories, metric_save_path, True, dataset.feature_names, device)


    
    
    #