import torch.nn as nn
import time, copy, torch
import numpy as np

from allEdges.edgeBase import EdgeBase
from allModel.autoencoder_pretrained import create_autoencoder

class EdgeAgg(EdgeBase):
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

        self.autoencoder = create_autoencoder().cuda()

        # 边缘的noises和labels
        self.noises = []
        self.labels = []

    def train(self, index, f, args):
        if  index % self.eval_term == 0:
            edge_index, train_loss_list, test_acc_list, std_accs_list = self.all_evaluate(index)
            f.write("communication " + str(index) +" :\n")
            f.write("Edge " + str(self.index) +" :\n")
            f.write("train_loss "+str(train_loss_list)+ "\n")
            f.write("test_acc " + str(test_acc_list) +"\n")
            f.write("std_accs " + str(std_accs_list) + "\n")
            f.write("\n")
        # 当前边缘下，每个端调用
        for end in self.ends_registration:
            end.train(args)

    # # 边缘调用，node和汇总节点相互学习
    # def train_FedAgg(node, args):
    #     BSBODP(node, node.parent, args)            

    