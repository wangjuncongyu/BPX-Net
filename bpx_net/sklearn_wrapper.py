import os
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset
from .focal_loss import FocalLoss

# 导入你的底层网络和优化器获取函数
from .models.BPXNet import BPXNet
from .trainers.classifier_trainers import GetOptimizer

class BPXNetClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn 风格的 BPXNet 分类器包装类。
    """
    def __init__(
        self,
        # 模型架构参数
        num_variables=120,
        bprd_gama=0.2,
        fill_v=-1,
        anfs_hiden_features=128,
        anfs_out_activation='tanh',
        anfs_ema_alpha=0.98,
        classifier_name='kan',
        classifier_layers=None,
        num_classes=3,
        prior_knowledge=None,
        # 训练控制参数
        epochs=100,
        batch_size=512,
        lr=0.001,
        lr_decay_steps=50,
        lr_decay_rate=0.98,
        use_focal_loss=True,
        optimizer_name='adamw',
        device='auto'
    ):
        if classifier_layers is None:
            classifier_layers = [num_variables, 128, 128, 128]
            
        # Sklearn 规范：__init__ 必须且只能做参数赋值
        self.num_variables = num_variables
        self.bprd_gama = bprd_gama
        self.fill_v = fill_v
        self.anfs_hiden_features = anfs_hiden_features
        self.anfs_out_activation = anfs_out_activation
        self.anfs_ema_alpha = anfs_ema_alpha
        self.classifier_name = classifier_name
        self.classifier_layers = classifier_layers
        self.num_classes = num_classes
        self.prior_knowledge = prior_knowledge
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate
        self.use_focal_loss = use_focal_loss
        self.optimizer_name = optimizer_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        
        self.network = None
        self.classes_ = np.arange(num_classes) # Sklearn ClassifierMixin 需要的属性

    def fit(self, X_train, y_train, eval_set=None, save_model_dir=None):
        """
        训练模型，支持验证集和自动保存最优权重。
        eval_set: tuple (X_val, y_val)
        """
        # 1. 初始化 PyTorch 实体网络
        self.network = BPXNet(
            num_variables=self.num_variables,
            bprd_gama=self.bprd_gama,
            fill_v=self.fill_v,
            anfs_hiden_features=self.anfs_hiden_features,
            anfs_out_activation=self.anfs_out_activation,
            anfs_ema_alpha=self.anfs_ema_alpha,
            classifier_name=self.classifier_name,
            classifier_layers=self.classifier_layers,
            num_classes=self.num_classes,
            prior_knowledge=self.prior_knowledge,
            device=self.device
        )
        self.network.to(self.device)

        # 2. 准备 DataLoader
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                      torch.tensor(y_train, dtype=torch.long))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if eval_set is not None:
            val_dataset = TensorDataset(torch.tensor(eval_set[0], dtype=torch.float32), 
                                        torch.tensor(eval_set[1], dtype=torch.long))
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # 3. 准备优化器和损失函数
        optimizer = GetOptimizer(self.optimizer_name, self.network, self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_decay_steps, gamma=self.lr_decay_rate) if self.lr_decay_steps > 0 else None
        loss_func = FocalLoss(gamma=4.0) if self.use_focal_loss else torch.nn.CrossEntropyLoss()

        # 4. 训练日志与早停初始化
        if save_model_dir:
            os.makedirs(save_model_dir, exist_ok=True)
            log_path = os.path.join(save_model_dir, "training_summary.csv")
            pd.DataFrame(columns=['time', 'step', 'loss', 'accuracy']).to_csv(log_path, index=False)
            
        max_acc = -1.0
        best_weights = None

        # 5. 核心训练循环
        for epoch in range(1, self.epochs + 1):
            self.network.train()
            losses = []
            t1 = time.time()
            
            for xs, ys in train_loader:
                xs, ys = xs.to(self.device), ys.to(self.device)
                
                # 前向传播 (BPXNet 返回 coeffs 和 out_scores)
                coeffs, out_scores = self.network(xs)
                loss = loss_func(out_scores, ys)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                
            if scheduler: scheduler.step()
            train_loss = np.mean(losses)
            
            # 6. 验证逻辑
            acc = -1.0
            if eval_set is not None:
                acc = self._evaluate(val_loader)
                
            t2 = time.time()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{self.epochs} | Loss: {train_loss:.4f} | Lr: {current_lr:.6f} | Time: {t2-t1:.2f}s | Acc: {acc:.4f}")

            # 7. 记录日志与保存最优模型
            if save_model_dir:
                pd.DataFrame([[datetime.now(), f"Step[{epoch}]", train_loss, acc]]).to_csv(log_path, mode='a', header=False, index=False)
                
                if acc > max_acc or acc == -1.0:
                    max_acc = acc
                    best_weights = self.network.state_dict().copy()
                    torch.save(best_weights, os.path.join(save_model_dir, "best_weights.pth"))

        # 训练结束后，自动加载最优权重
        if best_weights is not None:
            self.network.load_state_dict(best_weights)

        return self

    def _evaluate(self, dataloader):
        self.network.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xs, ys in dataloader:
                xs, ys = xs.to(self.device), ys.to(self.device)
                _, out_scores = self.network(xs)
                predicted = torch.argmax(out_scores, dim=1)
                correct += (predicted == ys).sum().item()
                total += ys.size(0)
        return correct / total

    def predict_proba(self, X):
        """返回各个类别的预测概率矩阵 (N, num_classes)"""
        self.network.eval()
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        all_scores = []
        with torch.no_grad():
            for xs in loader:
                xs = xs[0].to(self.device)
                _, out_scores = self.network(xs) # BPXNet 内部已经做了 F.softmax
                all_scores.append(out_scores.cpu().numpy())
                
        return np.vstack(all_scores)

    def predict(self, X):
        """返回预测的类别标签数组 (N,)"""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def save_model(self, filepath):
        """
        安全地保存模型，避免直接 pickle 整个类导致的 unpickling error。
        只保存 sklearn 的初始化参数和底层网络的 state_dict。
        """
        save_dict = {
            'init_params': self.get_params(),
            'model_state_dict': self.network.state_dict() if self.network is not None else None
        }
        torch.save(save_dict, filepath)

    def load_model(self, filepath):
        """加载模型权重和参数"""
        # checkpoint = torch.load(filepath, map_location=self.device)
        # self.set_params(**checkpoint['init_params'])
        state_dict = torch.load(filepath, map_location=self.device)
        # 重新初始化网络
        if self.network is None:
            self.network = BPXNet(
                num_variables=self.num_variables, bprd_gama=self.bprd_gama, fill_v=self.fill_v,
                anfs_hiden_features=self.anfs_hiden_features, anfs_out_activation=self.anfs_out_activation,
                anfs_ema_alpha=self.anfs_ema_alpha, classifier_name=self.classifier_name,
                classifier_layers=self.classifier_layers, num_classes=self.num_classes,
                prior_knowledge=self.prior_knowledge, device=self.device
            )
            self.network.to(self.device)
        
        self.network.load_state_dict(state_dict, strict=True)
        self.network.eval()