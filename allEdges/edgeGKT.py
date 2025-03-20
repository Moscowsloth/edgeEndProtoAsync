import torch.nn as nn
import time, copy, torch
import numpy as np

from allEdges.edgeBase import EdgeBase

class EdgeGKT(EdgeBase):
    # 边缘索引、所属端、参数、边缘模型
    def __init__(self, index, dids, args, model):
        super().__init__(index, dids, args, model)
        self.eval_term = args.eval_term
        self.num_classes = args.num_classes
        self.aggregated_edge = False    # 是否是聚合边缘
        self.device = args.device

        # 测试和训练记录
        self.rs_test_acc = []
        self.rs_train_loss = []

        # 设置损失函数
        self.loss = nn.CrossEntropyLoss()
        # 客户端数据提取的feature_map，以端的索引为key
        self.client_extracted_feauture_dict = dict()
        # 客户端数据提取的logits，以端的索引为key
        self.client_logits_dict = dict()
        # 客户端数据提取的labels，以端的索引为key
        self.client_labels_dict = dict()
        # 服务器提取的logits，要传给客户端
        self.server_logits_dict = dict()

    def train(self, index, f):
        if  index % self.eval_term == 0:
            edge_index, train_loss_list, test_acc_list, std_accs_list = self.all_evaluate(index)
            f.write("communication " + str(index) +" :\n")
            f.write("Edge " + str(self.index) +" :\n")
            f.write("train_loss "+str(train_loss_list)+ "\n")
            f.write("test_acc " + str(test_acc_list) +"\n")
            f.write("std_accs " + str(std_accs_list) + "\n")
            f.write("\n")
        # 当前所属端进行训练，获得各个端的信息
        for end in self.ends_registration:
            end.train()
            extracted_feature_dict = end.extracted_feature_dict
            logits_dict = end.logits_dict
            labels_dict = end.labels_dict

            # 为每个客户端准备一个s_logits_dict，记录服务器自己的logits
            s_logits_dict = dict()
            self.server_logits_dict[end.index] = s_logits_dict

            # 遍历batch
            for batch_index in extracted_feature_dict.keys():
                # 取batch_index对应的 batch_feature_map_x 和对应的标签 batch_labels
                batch_feature_map_x = torch.from_numpy(extracted_feature_dict[batch_index]).to(self.device)
                batch_labels = torch.from_numpy(labels_dict[batch_index]).long().to(self.device)
                # 获得自己模型的输出
                output_batch = self.model.forward_features_server(batch_feature_map_x)
                loss_true = self.loss(output_batch, batch_labels).to(self.device)
                loss = loss_true

                # 边侧模型更新
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 设置好服务器提取的logits，要传给每个客户端的zs
                s_logits_dict[batch_index] = output_batch.cpu().detach().numpy()

            # 为每个客户端传递自己的logits
            end.server_logits_dict = self.send_global_logits(end.index)

    def send_global_logits(self, client_index):
        return self.server_logits_dict[client_index]


            


