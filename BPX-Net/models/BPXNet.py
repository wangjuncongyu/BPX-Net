'''
author: Jun Wang 
copyright: Hangzhou City University
times:2025.1.13
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .classifiers import GetClassifier

class BPRDModule(nn.Module):
    '''
    the biomarker preserved random dropout module for simulating missing values
    '''
    def __init__(self, num_variables, gama=0.2, fill_v=-1, prior_knowledge=None, device='cpu'):
        super(BPRDModule, self).__init__()
        '''      
        num_variables:the number of variables for each sample  
        gama: the ratio to preserve biomarkers that should not dropout
        fill_v: the value used to replace the missing values, default set to -1
        device:cpu or cuda
        '''
        assert gama>=0.0 and gama<=1.0, \
            'unmask_topk_ratio must in range of [0.0, 1.0], but given ' + str(gama)
        self.num_variables = num_variables
        self.gama = gama
        self.fill_v= torch.tensor(fill_v, dtype=torch.float32).to(device)
        self.device = device       

        self.prior_knowledge = prior_knowledge
        if not self.prior_knowledge is None:          
            assert self.prior_knowledge.shape[0] == num_variables, 'Size of expert_knowledge mismatch with in features'
            if isinstance(self.prior_knowledge, np.ndarray):
                self.prior_knowledge = torch.tensor(self.prior_knowledge, dtype=torch.float32, device=self.device)
          
            self.prior_knowledge = self.prior_knowledge/torch.max(self.prior_knowledge)
          
        #the dropout probability 
        self.feature_importance = torch.nn.Parameter(torch.ones([num_variables]), requires_grad=False).to(device)*1.0/num_variables


    def forward(self, variables, model_kownledge=None):
        '''
        variabels: variables tensor (batch_size, num_features) 
        model_kownledge: the model weights of each variable (feedback from the ema of ANFS module)
        '''
        #dropout only activated on training stage
        if self.training:
            if model_kownledge is not None:                
                #model_kownledge = torch.where(torch.isnan(model_kownledge), 0.0, model_kownledge)
                
                self.feature_importance = F.softmax(torch.abs(model_kownledge), dim=0)
                #print(self.dropout_p)
                if torch.isnan(self.feature_importance).any():
                    print('feature_importance has nan')
                   
                if torch.isinf(self.feature_importance).any():
                    print('feature_importance has inf')              

            if self.prior_knowledge is not None:
                self.feature_importance = torch.where(self.feature_importance>self.prior_knowledge, self.feature_importance, self.prior_knowledge)                         
            
            variables = self.__weighted_random_dropout__(variables, self.gama, self.feature_importance)
        return variables
    
    def __weighted_random_dropout__(self, x, r, w, p=None):
        """
        Drop out least important features with randomness, while always keeping top-r fraction.

        Args:
            x: Tensor of shape (B, D) — input feature vectors
            r: float in (0, 1) — fraction of most important features to always keep
            w: Tensor of shape (B, D) or (D,) — importance scores        
            p: float in [0, 1 - r] or None — random keep rate for remaining features.
            If None, p is randomly sampled per batch.

        Returns:
            x_masked: Tensor of shape (B, D) — masked feature vectors
        
        """
        B, D = x.shape
        k = int(D * r)  # number of features to always keep

        # Broadcast w to shape (B, D) if needed
        if w.dim() == 1:
            w = w.unsqueeze(0).expand(B, -1)

        # Get indices of top-k features per sample
        _, topk_idx = torch.topk(w, k=k, dim=1)

        # Create binary mask initialized to zeros
        mask = torch.zeros_like(x).to(self.device)

        # Always keep top-k features
        mask.scatter_(1, topk_idx, 1.0)

        # Handle the rest: features not in top-k
        remaining_mask = 1.0 - mask

        # If p not specified, sample a random keep rate for remaining features
        if p is None:
            p = torch.rand(1).item() * (1 - r)

        # Generate random mask for remaining features (same shape)
        random_keep = torch.bernoulli(torch.full_like(x, p))

        # Apply random keep only to non-top-k features
        mask = mask + (remaining_mask * random_keep)

        # Apply mask to input
        x_masked = torch.where(mask==0, torch.tensor(self.fill_v).to(self.device), x)

        return x_masked
    
class ANFSModule(nn.Module):
    def __init__(self, num_variables, hiden_features=128, out_activation='tanh', ema_alpha = 0.99, device='cpu'):
        super(ANFSModule, self).__init__()
        self.num_variables = num_variables
        self.register_buffer('ema',torch.nn.Parameter(torch.zeros([num_variables]), requires_grad=True).to(device)*1.0 )
        self.ema_alpha= ema_alpha
       
        self.feature_selector = nn.Sequential(
           nn.Linear(num_variables, hiden_features),
           nn.ReLU(),

           nn.Linear(hiden_features, hiden_features),
           nn.ReLU(), 

           nn.Linear(hiden_features, hiden_features),
           nn.ReLU(), 

           nn.Linear(hiden_features, num_variables),
           self.__get_activation__(out_activation)
           )
        self.to(device)
        
    def __get_activation__(self, activation_name):
        if activation_name=='tanh':
            return nn.Tanh()
        elif activation_name == 'sigmoid':
            return nn.Sigmoid()
        elif activation_name == 'softmax':
            return nn.Softmax()
        else:
            NotImplemented

    def forward(self, variables):        
        correlation_coefficents = self.feature_selector(variables)
        selected_variables = variables*correlation_coefficents
        if self.training:
            #update ema 
            self.ema = self.ema_alpha * self.ema + (1-self.ema_alpha)*torch.mean(correlation_coefficents, dim=0)
          
        return correlation_coefficents, selected_variables


class BPXNet(nn.Module):
    '''
    the deep model for end-to-end biomarker discovery and disease modeling
    '''
    def __init__(self, 
                 num_variables=120,
                 bprd_gama=0.2,
                 fill_v = -1,
                 anfs_hiden_features=128, 
                 anfs_out_activation= 'tanh',
                 anfs_ema_alpha = 0.99,
                 classifier_name='transformer',
                 classifier_layers = [120, 128, 128, 128],
                 num_classes=3,
                 prior_knowledge=None,
                 device='cpu'):
        super(BPXNet, self).__init__()
        '''
        num_variables: the number of variables 
        bprd_gama: the perservation ratio of biomarkers in the BPRD module
        fill_v: the value used to fill the missing values, default -1
        anfs_hiden_features: the hiden featrues for the subnetwork in the ANFS module. Note value 0 means not using this module
        anfs_out_activation: the activation used to get the coefficients, default tanh
        anfs_ema_alpha: the hyperparameter used to control the EMA of coefficients
        classifier_name: kan, mamba, or transformer that used for the classification task
        classifier_layers: used to control the depth and features of the classifier
        num_classes: the number of categories for the classification task
        expert_knowledge: the coefficents reflecting variabel importance by experts, default None
        device: cpu or cuda
        '''

        self.device = device
      
        self.BPRD = BPRDModule(num_variables=num_variables, 
                               gama=bprd_gama,
                               fill_v=fill_v,
                               prior_knowledge=prior_knowledge,
                               device = device)
        if anfs_hiden_features == 0 : self.ANFS = None
        else:
            self.ANFS = ANFSModule(num_variables=num_variables,
                               hiden_features=anfs_hiden_features,
                               out_activation=anfs_out_activation,
                               ema_alpha=anfs_ema_alpha,
                               device=device)

        self.classifier = GetClassifier(classifier_name, classifier_layers, num_classes, 0.2, device)        

        self.to(device)
        
    def forward(self, variables, return_feature = False):
        '''
        input: samples (N, num_variables)
        mask: (N, num_variables) indicates which variable with missing value
        '''        
        if self.ANFS is None:
            correlation_coefficents = torch.zeros_like(variables).to(self.device)#make the coeff to zeros
            selected_variables = variables
        else:
          
            variables = self.BPRD(variables, self.ANFS.ema)
            correlation_coefficents, selected_variables = self.ANFS(variables)
        if return_feature:
            embeddings, outcome_logits = self.classifier(selected_variables, return_feature=True)
            outcome_scores = F.softmax(outcome_logits, dim=-1)
            return correlation_coefficents, outcome_scores, embeddings
        else:
            outcome_logits = self.classifier(selected_variables)
            outcome_scores = F.softmax(outcome_logits, dim=-1)
            return correlation_coefficents, outcome_scores
       

#unit testing of the DeepBioDis
if __name__ == '__main__':
    data = torch.rand(4,10)
    data[0,0] = -1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BPXNet(10, 0.2, -1, 64, 'tanh', 0.99, 'kan', [10, 64, 64, 64], 3, None, device=device)
    
    data = data.to(device)

    model.train()
    print('1:', model.BPRD.dropout_p)
    coefficients, outcome_scores = model(data)
    print('=========== training#1 =================')
    print('\n', model.BPRD.dropout_p)

    model.eval()
    coefficients, outcome_scores = model(data)
    
    print('=========== testing#1 =================')
    print('\n', model.BPRD.dropout_p)

    model.train()
    print('1:', model.BPRD.dropout_p)
    coefficients, outcome_scores = model(data)
    print('=========== training#2 =================')
    print('\n', model.BPRD.dropout_p)

    model.eval()
    coefficients, outcome_scores = model(data)
    
    print('=========== testing#1 =================')
    print('\n', model.BPRD.dropout_p)
