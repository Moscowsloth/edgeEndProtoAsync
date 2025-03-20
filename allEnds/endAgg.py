import torch.nn as nn
import time, copy,torch
from collections import defaultdict
import torch.nn.functional as F
from torch.optim import lr_scheduler

from allEnds.endBase import EndBase
from allModel.model import FedAvgCNN, BaseHeadSplit
from allModel.autoencoder_pretrained import create_autoencoder

class EndAgg(EndBase):
    def __init__(self, index, args, model):
        super().__init__(index, args, model)
        self.loss_mse = nn.MSELoss()
        self.device = args.device
        self.lamda = 2
        self.num_classes = args.num_classes

        # 设置noises和labels
        self.noises = []
        self.labels = []
        # 设置autoencoder
        self.autoencoder=create_autoencoder().cuda()

    # 端节点为子节点，边节点为父节点
    def BSBODP_from_end_to_edge(self, args):
        T_agg = 3.0
        noises = self.parent.noises if len(self.parent.noises)<len(self.noises) else self.noises
        labels = self.parent.labels if len(self.parent.labels)<len(self.labels) else self.labels
        crit_non_leaf = Loss_Non_Leaf(T_agg)
        crit_leaf = Loss_Leaf(T_agg)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,80,100], gamma=0.9)

        for idx,(noise,label) in enumerate(zip(noises,labels)):
            optimizer.zero_grad()
            fake_data = self.parent.autoencoder.decoder(noise)
            nei_logits, _ = self.parent.model.fedagg_forward_features_server(fake_data)
            logits_fake, _ = self.model.fedagg_forward_features_server(fake_data)
            loss=0.0

            for temp_idx, (temp_img, temp_label_) in enumerate(self.train_loader):
                img, label_ = temp_img, temp_label_
                if temp_idx == idx:
                    break
                
            img, label_ = img.cuda(), label_.cuda()
            logits_true,_ = self.model.fedagg_forward_features_server(img)
            loss = loss + crit_leaf(logits_fake, nei_logits, logits_true, label.long())
            # 做更新
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()

    # 端节点为父节点，边节点子子节点
    def BSBODP_from_edge_to_end(self, args):
        T_agg = 3.0
        noises = self.noises if len(self.noises)<len(self.parent.noises) else self.parent.noises
        labels = self.labels if len(self.labels)<len(self.parent.labels) else self.parent.labels
        crit_non_leaf = Loss_Non_Leaf(T_agg)
        crit_leaf = Loss_Leaf(T_agg)
        optimizer = torch.optim.SGD(self.parent.model.parameters(), lr=args.lr, momentum=0.9)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,80,100], gamma=0.9)
        for idx,(noise,label) in enumerate(zip(noises,labels)):
            optimizer.zero_grad()
            fake_data = self.autoencoder.decoder(noise)
            nei_logits, _ = self.model.fedagg_forward_features_server(fake_data)
            logits_fake, _ = self.parent.model.fedagg_forward_features_server(fake_data)
            loss=0.0
            loss=loss+crit_non_leaf(logits_fake,nei_logits,label.long())
            # 做更新
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()

    def train(self, args):
        self.model.train()
        # 调用BSBODP以端视角进行端边联合训练
        self.BSBODP_from_end_to_edge(args)
        self.BSBODP_from_edge_to_end(args)

    def get_noises_labels(self):
        # 初始化操作遍历node.dataset里的idx和data
        for idx,data in enumerate(self.train_loader):
            # 获得图像和标签
            img, label = data
            img = img.to(self.device)
            label = label.to(self.device)
            # 生成noise并添加
            noise = self.autoencoder.encoder(img)
            self.noises.append(noise)
            self.labels.append(label)
        # 为自己的父节点增加noises和labels
        self.parent.noises.extend(self.noises)
        self.parent.labels.extend(self.labels)

class KL_Loss(nn.Module):
    def __init__(self, temperature=3.0):
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

class CE_Loss(nn.Module):
    def __init__(self, temperature=1):
        super(CE_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        # output_batch      -> B X num_classes
        # teacher_outputs   -> B X num_classes

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1)

        # Same result CE-loss implementation torch.sum -> sum of all element
        loss = -self.T * self.T * torch.sum(torch.mul(output_batch, teacher_outputs)) / teacher_outputs.size(0)

        return loss

class Loss_Non_Leaf(nn.Module):
    def __init__(self, temperature=1, alpha=10):
        super(Loss_Non_Leaf, self).__init__()
        self.alpha = alpha
        self.kl_loss_crit=KL_Loss(temperature)
        self.ce_loss_crit=nn.CrossEntropyLoss()

    def forward(self, output_batch, teacher_outputs, label):
        
        loss_ce=self.ce_loss_crit(output_batch,label.long())
        loss_kl=self.kl_loss_crit(output_batch,teacher_outputs.detach())
        loss_true=loss_ce+self.alpha*loss_kl
        return loss_true

class Loss_Leaf(nn.Module):
    def __init__(self, temperature=1, alpha=1, alpha2=1):
        super(Loss_Leaf, self).__init__()
        self.alpha = alpha
        self.alpha2 = alpha2
        self.non_leaf_loss_crit=Loss_Non_Leaf(temperature,alpha)
        self.ce_loss_crit=nn.CrossEntropyLoss()
    def forward(self, output_fake, teacher_outputs_fake, output_true, label):
        loss_leaf=self.non_leaf_loss_crit(output_fake, teacher_outputs_fake.detach(), label.long())
        loss_ce=self.ce_loss_crit(output_true,label.long())
        loss_true=loss_leaf+self.alpha2*loss_ce
        return loss_true