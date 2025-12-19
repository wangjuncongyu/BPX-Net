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

