# BPX-Net
# Code for Biomarker-Preserved Explainable Networks for Disease Diagnosis and Prognosis with Multi-omics

> This is a deep learning framework for medical multimodal data modeling and biomarker discovery

### my enviroment
- Winows 10
- Anaconda python 3.12.7
- Pytorch 2.6.0+cu126
## dataset
Data samples for each cohort can be found at subdir 'datasets' in csv files.Full version of data is vailable via email:wjcy19870122@163.com


## training
``` bash
(1)Get full datasets and put to datasets
(2) cmd run: python pneu_cross_validation_5folds.py to train pneumonia diagnoiss
(3) cmd run: python gdm_cross_validation_5folds.py to train gdm stratification
(4) cmd run: ccrcc_cross_validation_5folds.py to train ccrcc prognsis
(5) cmd run: aml_cross_validation_5folds.py to train aml treatment response
note: models are saved at checkpoints, and evluation metrics are saved at results
```

## training using custom dataset
``` bash
(1) put your data in a csv file with each line for sample, each column for features and the label
(2) from datasets.csv_dataset import CsvDataset
    data_file = ['path/your_data.csv']
    ignores=['feature name'] #feature columns to ignore
    dataset = CsvDataset(data_file,
                                label_column='label',#specify the column name of label 
                                ignores=ignores,
                                max_norm=True, #maximum normalization
                                maxv_for_norm=None,#None or numpy array of max values for feature normalization 
                                repeat_fews=True) #set true to alliviate data imbalance   
(3) define model:
    from models.BPXNet import BPXNet
    model = BPXNet(dataset.getNumFeatures(),#input_features
                        bprd_gama = BPRD_gama,#biomarker preservation ratio for BPRD
                        fill_v=-1, #missing values set to -1
                        anfs_hiden_features=128, #the dimension for hiden layers in ANFS
                        anfs_out_activation='tanh', #the activation function for feature importance
                        anfs_ema_alpha=0.98, #EMA weights for updating statistically significant feature importance
                        classifier_name = 'kan', #decision-maker, optional:kan,transformer,resnet18-50, longformer 
                        classifier_layers = [dataset.getNumFeatures(), 128, 128, 128], #the depth of decison-maker
                        num_classes= num_classes,#output number of classes
                        prior_knowledge= prior_weights, #None or preloaded prior feature importance
                        device=device)#device:cpu or cuda

note: details how to use can be referred to pneu_cross_validation_5folds.py
