import torch.nn as nn
import time, copy, torch
from collections import defaultdict
import torch.nn.functional as F

from average import agg_func
from allEnds.endBase import EndBase
from allModel.model import FedAvgCNN


class EndGKT(EndBase):
    def __init__(self, index, args, model):
        super().__init__(index, args, model)
        self.device = args.device
        self.lamda = 1
        self.loss = nn.CrossEntropyLoss()
        self.num_classes = args.num_classes

        self.server_logits_dict = dict()
        self.temperature = 1
        self.criterion_KL = KL_Loss(self.temperature)

        # 初始化待传给服务器
        self.extracted_feature_dict = dict()
        self.logits_dict = dict()
        self.labels_dict = dict()

    def train(self):
        self.model.train()
        start_time = time.time()

        max_local_epochs = self.local_epoch

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                logs_probs = self.model(x)
                loss_true = self.loss(logs_probs, y)
                # 用全局模型的Logits来辅助训练
                if len(self.server_logits_dict) != 0:
                    large_model_logits = torch.from_numpy(self.server_logits_dict[i]).to(self.device)
                    loss_kd = self.criterion_KL(logs_probs, large_model_logits)
                    loss = loss_true + self.temperature * loss_kd
                else:
                    loss = loss_true

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

        for i, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            log_probs, extracted_features = self.model.forward_features(x)
            self.extracted_feature_dict[i] = extracted_features.cpu().detach().numpy()
            log_probs = log_probs.cpu().detach().numpy()
            self.logits_dict[i] = log_probs
            self.labels_dict[i] = y.cpu().detach().numpy()
        
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    # 从服务器获得服务器的logits
    def receive_server_logits(self, logits):
        self.server_logits_dict = logits

# 蒸馏损失
class KL_Loss(nn.Module):
    def __init__(self, temperature=1):
        super(KL_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        # output_batch  -> B X num_classes
        # teacher_outputs -> B X num_classes

        # loss_2 = -torch.sum(torch.sum(torch.mul(F.log_softmax(teacher_outputs,dim=1), F.softmax(teacher_outputs,dim=1)+10**(-7))))/teacher_outputs.size(0)
        # print('loss H:',loss_2)

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)

        loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs)

        # Same result KL-loss implementation
        # loss = T * T * torch.sum(torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch)))/teacher_outputs.size(0)
        return loss
