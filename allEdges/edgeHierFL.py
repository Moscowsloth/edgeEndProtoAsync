import torch.nn as nn
import time, copy, torch
import numpy as np

from allEdges.edgeBase import EdgeBase

class EdgeHierFL(EdgeBase):
    # 边缘索引、所属端、参数、边缘模型
    def __init__(self, index, dids, args, model):
        super().__init__(index, dids, args, model)
        self.eval_term = args.eval_term
        self.num_classes = args.num_classes
        self.aggregated_edge = False    # 是否是聚合边缘

        # 测试和训练记录
        self.rs_test_acc = []
        self.rs_train_loss = []
    
    def train(self, index, f):
        if  index % self.eval_term == 0:
            edge_index, train_loss_list, test_acc_list, std_accs_list = self.all_evaluate(index)
            f.write("communication " + str(index) +" :\n")
            f.write("Edge " + str(self.index) +" :\n")
            f.write("train_loss "+str(train_loss_list)+ "\n")
            f.write("test_acc " + str(test_acc_list) +"\n")
            f.write("std_accs " + str(std_accs_list) + "\n")
            f.write("\n")
            
        # 当前端做训练
        for end in self.ends_registration:
            end.train()
        print(f"Communication round %d, All the ends in edge %d has trained!" % (index, self.index))
        # 获得self.uploaded_ids、self.uploaded_weights、self.uploaded_models
        self.receive_models()
        # 单个边缘获得自己的全局模型
        self.aggregate_parameters()

        if self.aggregated_edge == True:
            # 聚合边缘调用evaluate方法展示结果，all_evaluate待实现
            # if  index % self.eval_term == 0:
            #     edge_index, train_loss_list, test_acc_list, std_accs_list = self.all_evaluate(index)
            #     f.write("communication " + str(index) +" :\n")
            #     f.write("train_loss "+str(train_loss_list)+ "\n")
            #     f.write("test_acc " + str(test_acc_list) +"\n")
            #     f.write("std_accs " + str(std_accs_list) + "\n")
            #     f.write("\n")
            # 进行边缘级的模型聚合，聚合后的模型是self.end_global_model
            # 获得self.uploaded_edge_ids、self.uploaded_edge_weights、self.uploaded_edge_models
            self.receive_models_from_edges()
            self.aggregate_parameters_edge()
            # 将聚合后的模型发给自己的各客户端，给self.global_model结果也是不变的！
            self.sender_knowledges = self.end_global_model
            self.send_to_ends_model()
            print(f"Communication round %d, Aggregation Edge %d has sent global models to its devices" % (index, self.index))
            # 邻居边缘将聚合后的全局模型发给各客户端
            for neigh in self.neigh_registration:
                neigh.sender_knowledges = self.end_global_model
                neigh.send_to_ends_model()
                print(f"Communication round %d, Normal Edge %d has received global models from edge %d." % (index, neigh.index, self.index))

            

            # print(len(self.all_edge_global_model))
            #     self.edge_protos_all.append(neigh.edge_protos)
        # self.receive_models()
