import torch.nn as nn
import time, copy, torch
import numpy as np

import multiprocessing
from multiprocessing import Manager
import queue
import schedule

from allEdges.edgeBase import EdgeBase
from average import proto_aggregation, edge_proto_aggregation

class EdgeProAgg(EdgeBase):
    # 边缘索引、所属端、参数、边缘模型
    def __init__(self, index, dids, args, model):
        super().__init__(index, dids, args, model)

        self.eval_term = args.eval_term
        self.num_classes = args.num_classes
        self.aggregated_edge = False    # 是否是聚合边缘
        self.global_protos = []
        self.uploaded_ids = []
        self.uploaded_protos = []

        # 测试和训练记录
        self.rs_test_acc = []
        self.rs_train_loss = []


    def train(self, index, f):
        # 开始时间
        s_time = time.time()
        # 测试
        if  index % self.eval_term == 0:
            edge_index, train_loss_list, test_acc_list, std_accs_list = self.all_evaluate(index)
            f.write("communication " + str(index) +" :\n")
            f.write("Edge " + str(self.index) +" :\n")
            f.write("train_loss "+str(train_loss_list)+ "\n")
            f.write("test_acc " + str(test_acc_list) +"\n")
            f.write("std_accs " + str(std_accs_list) + "\n")
            f.write("\n")

        # 当前端做训练，每个边缘获得对应的protos
        for end in self.ends_registration:
            end.train()
        self.receive_protos()
        # 边缘做聚合获得边缘原型
        for item in self.uploaded_protos:
            for value in item.values():
                print(str(type(value)))
                break
        self.edge_protos = proto_aggregation(self.uploaded_protos)

        # 如果是聚合边缘，用聚合边缘控制邻居边缘的下发
        if self.aggregated_edge == True:
            # 聚合边缘调用evaluate方法展示结果
            
            # if index > 0 and index % self.eval_term == 0:
            #     edge_index, train_loss_list, test_acc_list, std_accs_list = self.all_evaluate(index)
            #     f.write("communication " + str(index) +" :\n")
            #     f.write("train_loss "+str(train_loss_list)+ "\n")
            #     f.write("test_acc " + str(test_acc_list) +"\n")
            #     f.write("std_accs " + str(std_accs_list) + "\n")
            #     f.write("\n")

            self.edge_protos_all = [copy.deepcopy(self.edge_protos)]
            for neigh in self.neigh_registration:
                self.edge_protos_all.append(neigh.edge_protos)
            # 获得全局原型
            self.global_protos = edge_proto_aggregation(self.edge_protos_all)
            # 发送全局原型给所有边缘
            for neigh in self.neigh_registration:
                neigh.global_protos = self.global_protos
                print(f"Communication round %d, Normal Edge %d has received global protos from edge %d." % (index, neigh.index, self.index))
            # 边缘获得全局原型后, 发给自己的客户端
            if self.global_protos is not None:
                # 将全局原型发给端
                for neigh in self.neigh_registration:
                    for end in neigh.ends_registration:
                        end.receive_protos(self.global_protos)
                    print(f"Communication round %d, Normal Edge %d has sent global protos to its devices" % (index, neigh.index))
                for end in self.ends_registration:
                    end.receive_protos(self.global_protos)
                print(f"Communication round %d, Aggregtaion Edge %d has sent global protos to its devices" % (index, self.index))

        # 待补充，每个边缘加载预训练模型，用全局原型生成数据进行训练
        if index % 10 == 0 and len(self.global_protos)>0:
            pass




    # 边缘从所属客户端接收原型
    def receive_protos(self):
        for client in self.ends_registration:
            self.uploaded_ids.append(client.index)
            self.uploaded_protos.append(client.sender_knowledges)

