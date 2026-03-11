# BPX-Net
# Code for Biomarker-Preserved Explainable Networks for Disease Diagnosis and Prognosis with Multi-omics

> This is a deep learning framework for medical multimodal data modeling and biomarker discovery

### my enviroment
- Winows 10
- Anaconda python 3.10
- Pytorch 2.9.1+cu128
- Scikit-learn 1.1.0
- NumPy 1.26.4
- Pandas 2.2.3
## dataset
Data samples for each cohort can be found at subdir 'datasets' in csv files.Full version of data is vailable via email:wjcy19870122@163.com

## Installation
#### with pip
``` bash
pip install bpx-net
```
#### with conda
``` bash
conda activate your_env_name
conda install pip
pip install bpx-net
```

## How to use it?

(1) put your data in a csv file with each line for sample, each column for features and the label
``` bash
    from datasets.csv_dataset import CsvDataset
    data_file = ['path/your_data.csv']
    ignores=['feature name'] #feature columns to ignore
    dataset = CsvDataset(data_file,
                                label_column='label',#specify the column name of label 
                                ignores=ignores,
                                max_norm=True, #maximum normalization
                                maxv_for_norm=None,#None or numpy array of max values for feature normalization 
                                repeat_fews=True) #set true to alliviate data imbalance
``` 
(2) training using custom dataset
``` bash
    from bpx_net import BPXNetClassifier
    import os.path as osp
    import sklearn.metrics as skm
    from sklearn.preprocessing import label_binarize
    
    # define model
    model = BPXNetClassifier(
        num_variables=dataset.getNumFeatures(),
        bprd_gama=BPRD_gama,
        fill_v=-1,
        anfs_hiden_features=128,
        classifier_name=classifier,
        classifier_layers=[dataset.getNumFeatures(), 128, 128, 128],
        num_classes=num_classes,
        prior_knowledge=prior_weights,
        epochs=100,
        batch_size=512,
        lr=0.001,
        device=device
    )
    
    # fit
    model.fit(X_train, y_train, eval_set=(X_val, y_val), save_model_dir=ckpoint_save_path)
    
    # load model
    model.load_model(osp.join(ckpoint_save_path, "best_weights.pth")) 
    
    # predict
    y_pred = model.predict(X_val)
    accuracy = skm.accuracy_score(y_val, y_pred)
    print('accuracy:', accuracy)
    
    # predict proba
    y_pred_proba = model.predict_proba(X_val)
    if num_classes == 2:
        auc = skm.roc_auc_score(y_val, y_pred_proba[:, -1])        
    else:
        auc = skm.roc_auc_score(label_binarize(y_val,classes=[0, 1, 2]), 
                                            y_pred_proba, average='macro', multi_class='ovr')
    print('auc:', auc)
``` 
