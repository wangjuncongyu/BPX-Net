# BPX-Net
## Code for Biomarker-Preserved Explainable Networks for Disease Diagnosis and Prognosis with Multi-omics

> This is a deep learning framework for medical multimodal data modeling and biomarker discovery

## My enviroment
- Winows 10
- Anaconda python 3.10
- Pytorch 2.9.1+cu128
- Scikit-learn 1.1.0
- NumPy 1.26.4
- Pandas 2.2.3
- einops 0.8.1
## Dataset
Data samples for each cohort can be found at subdir 'datasets' in csv files.Full version of data is vailable via email:wjcy19870122@163.com

## Installation
#### with pip
``` bash
pip install bpx-net
```
#### with conda
``` bash
conda create -n bpx python=3.10 -y
conda activate bpx
pip install bpx-net
```
## Use custom dataset
```python
import numpy as np
from datasets.csv_dataset import CsvDataset
from sklearn.model_selection import train_test_split

data_file = ['data/MMIST-ccRCC.csv']
label_column = 'vital_status_12' 
ignores = [] 

dataset = CsvDataset(
    data_file,
    label_column=label_column,
    ignores=ignores,
    max_norm=True,
    repeat_fews=False  
)

X_norm = dataset.samples[:, 0:-1]
y = dataset.samples[:, -1]
```
or
```python
import pandas as pd
df = pd.read_csv('data/MMIST-ccRCC.csv', encoding='gbk')

if ignores:
    df = df.drop(columns=ignores)

X_raw = df.drop(columns=[label_column]).values
y = df[label_column].values

max_vals = np.max(X_raw, axis=0)
max_vals[max_vals == 0] = 1e-5
X_norm = np.where(X_raw == -1, -1, X_raw / max_vals)
```
```python
X_train, X_val, y_train, y_val = train_test_split(
    X_norm, 
    y, 
    test_size=0.2,       
    random_state=42, 
    stratify=y           
)
```
## How to use it?

(1) put your data in a csv file with each line for sample, each column for features and the label
(2) training using custom dataset
``` python
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
    prior_knowledge=prior_weights, # If not, you can set None.
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
## How to get feature importance?
```python
EMA = torch.abs(model.network.ANFS.ema)
```
## Model parameters
- <code>num_variables</code>: int (default=120)  
number of input_features  
- <code>bprd_gama</code>: float (default=0.2)   
biomarker preservation ratio for BPRD  
- <code>fill_v</code>: int (default=-1)  
missing values set to -1
- <code>anfs_hiden_features</code>: int (default=128)  
the dimension for hiden layers in ANFS
- <code>anfs_out_activation</code>: str (default = 'tanh')  
the activation function for feature importance
- <code>anfs_ema_alpha</code>: float (default=0.98)  
EMA weights for updating statistically significant feature importance
- <code>classifier_name</code>: str (default = 'kan')  
decision-maker, optional:kan,transformer,resnet18-50, longformer 
- <code>classifier_layers</code>: unit (default = None)  
the depth of decison-maker
- <code>num_classes</code>: int (default=3)  
output number of classes
- <code>prior_knowledge</code>: unit (default = None)  
None or preloaded prior feature importance
- <code>epochs</code>: int (default=100)    
number of epochs for trainng
- <code>batch_size</code>: int (default=512)  
number of examples per batch
- <code>lr</code>: float (default=0.001)  
learning rate
- <code>lr_decay_steps</code>: int (default=50)  
learning rate decay steps
- <code>lr_decay_rate</code>: float (default=0.98)  
learning rate decay rate
- <code>use_focal_loss</code>: bool (default=True)  
if use focal_loss
- <code>optimizer_name</code>: str (default = 'adamw')  
optimizer name
- <code>device</code>: str (default = 'auto')  
device:cpu or cuda

## Citing us
If you use BPX-Net, we would appreciate your references to [our paper](https://link.springer.com/article/10.1186/s13040-026-00537-1).
