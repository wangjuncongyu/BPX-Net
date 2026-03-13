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
- einops 0.8.1
## dataset
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
