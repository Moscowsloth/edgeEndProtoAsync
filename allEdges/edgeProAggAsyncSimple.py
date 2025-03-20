from collections import defaultdict

import torch.nn as nn
import time, copy, torch
import numpy as np

import multiprocessing
from multiprocessing import Manager
import queue
import schedule

from allEdges.edgeBase import EdgeBase
from average import proto_aggregation, edge_proto_aggregation
from allEnds import updatedProtoWithWeight


class EdgeProAggAsyncSimple(EdgeBase):
    # 边缘索引、所属端、参数、边缘模型
    def __init__(self, index, dids, args, model, epoch, period):
        super().__init__(index, dids, args, model)

        self.eval_term = args.eval_term
        self.num_classes = args.num_classes
        self.aggregated_edge = False  # 是否是聚合边缘
        self.global_protos = []
        self.uploaded_ids = []
        self.uploaded_protos = []
        self.uploaded_protos_with_index = {}

        # 测试和训练记录
        self.rs_test_acc = []
        self.rs_train_loss = []

        # 维护客户端状态
        self.end_delay_comm = {}  # 客户端的通信时延
        self.end_delay_comp = {}  # 客户端的计算时延
        self.end_delay = {}  # 客户端的总时延
        self.end_isUpdated = {}  # 客户端是否更新
        self.end_staleness = {}  # 客户端的staleness

        # 维护边缘的聚合信息
        self.edge_epoch = epoch  # 客户端的更新轮次
        self.edge_period = period  # 客户端的更新周期
        self.edge_need_update = 0  # 标识符，是否需要聚合
        self.edge_protos = None  # 边缘原型

    def asyncTrain(self, index, f, training_time):
        # 开始时间
        edge_start_time = time.time()
        # 测试
        if index % self.eval_term == 0:
            edge_index, train_loss_list, test_acc_list, std_accs_list = self.all_evaluate(index)
            f.write("communication " + str(index) + " :\n")
            f.write("Edge " + str(self.index) + " :\n")
            f.write("train_loss " + str(train_loss_list) + "\n")
            f.write("test_acc " + str(test_acc_list) + "\n")
            f.write("std_accs " + str(std_accs_list) + "\n")
            f.write("\n")

        cur_epoch = 0  # 记录当前的训练轮次
        for end in self.ends_registration:
            self.end_staleness[end.index] = 0  # 初始化：staleness均为0
            self.end_isUpdated[end.index] = 1  # 初始化：均参加第一轮的训练


        while cur_epoch < self.edge_epoch:
            # 当前端做训练，每个边缘获得对应的protos
            for end in self.ends_registration:
                # 判断：上一边缘周期是否更新
                if self.end_isUpdated[end.index] == 1:  # 上一周期更新则训练
                    end.train()
                    self.end_delay[end.index] = end.delay_comm + end.delay_comp
                    # 判断：是否能在本周期完成训练和上传
                    if self.end_delay[end.index] <= self.edge_period:  # 能够在周期内完成则上传
                        # todo 如果总时延小于边缘的聚合周期，则上传原型。
                        print("End " + str(end.index) + " is now updating its proto!")
                        self.end_isUpdated[end.index] = 1  # 设置上传标识
                    else:  # 无法在本周期完成
                        print("End " + str(end.index) + " can not finish training!")
                        self.end_isUpdated[end.index] = 0  # 不准上传
                        self.end_delay[end.index] -= self.edge_period  # 更新延迟，减去本周期的时长
                        self.end_staleness[end.index] += 1  # staleness+1
                else:  # 上一周期没有更新
                    if self.end_delay[end.index] <= self.edge_period:  # 有staleness，但这一周期能够完成
                        print("End " + str(end.index) + " is now updating its proto, but with staleness!!")
                        self.end_isUpdated[end.index] = 1  # 允许上传
                    else:  # 无法在本周期完成
                        print("End " + str(end.index) + " can not finish training!")
                        self.end_isUpdated[end.index] = 0  # 不准上传
                        self.end_delay[end.index] -= self.edge_period  # 更新延迟，减去本周期的时长
                        self.end_staleness[end.index] += 1  # staleness+1

            self.edge_need_update = 0  # 标识符置0
            for value in self.end_isUpdated.values():
                if value == 1:
                    self.edge_need_update = 1
                    print("Edge need proto aggregation!!!")
                    break

            self.receive_protos()  # 每个边缘轮次结束，边缘接收端设备的原型

            if self.edge_need_update == 1:
                self.weightedProtoAggregation()  # 根据权重聚合原型，得到这一轮次的边缘原型
                # 下发原型
                for end in self.ends_registration:
                    if self.end_isUpdated[end.index] == 1:  # 如果本轮进行了更新，则接受原型
                        end.receive_protos(self.edge_protos)
            cur_epoch += 1

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
                print(f"Communication round %d, Normal Edge %d has received global protos from edge %d." % (
                index, neigh.index, self.index))
            # 边缘获得全局原型后, 发给自己的客户端
            if self.global_protos is not None:
                # 将全局原型发给端
                for neigh in self.neigh_registration:
                    for end in neigh.ends_registration:
                        end.receive_protos(self.global_protos)
                    print(f"Communication round %d, Normal Edge %d has sent global protos to its devices" % (
                    index, neigh.index))
                for end in self.ends_registration:
                    end.receive_protos(self.global_protos)
                print(f"Communication round %d, Aggregtaion Edge %d has sent global protos to its devices" % (
                index, self.index))

        # 待补充，每个边缘加载预训练模型，用全局原型生成数据进行训练
        if index % 10 == 0 and len(self.global_protos) > 0:
            pass

        training_time += self.edge_epoch * self.edge_period  # 记录训练所需时间
        edge_end_time = time.time





    # 边缘从所属客户端接收原型
    def receive_protos(self):
        # for client in self.ends_registration:
        #     self.uploaded_ids.append(client.index)
        #     self.uploaded_protos.append(client.sender_knowledges)
        for end in self.ends_registration:
            if self.end_isUpdated[end.index] == 1:
                self.uploaded_protos_with_index[end.index] = end.sender_knowledges  # 将端设备的原型保存


    # 只接收更新结束的客户端的原型
    def receiveProtoFromUpdated(self):
        for client in self.ends_registration:
            if self.end_isUpdated[client.index] == 1:
                self.uploaded_ids.append(client.index)
                self.uploaded_protos.append(client.sender_knowledges)

    def weightedProtoAggregation(self):
        # 根据self.uploaded_protos_with_index记录的内容，能够同时找到proto和index
        weight_dict = {}
        w_sum = 0  # 权重和
        for index, proto in self.uploaded_protos_with_index.items():
            if self.end_staleness[index] == 0:
                print("End " + str(index) + " does not have staleness!")
                weight_dict[index] = 1
            else:
                print("End " + str(index) + " has staleness of " + str(self.end_staleness[index]) + "!")
                weight_dict[index] = (self.end_staleness[index] + 1) ** -0.5
            w_sum += weight_dict[index]  # 累加总权重
        print("The weight dict of the uploaded ends is:")
        print(weight_dict)

        aggregated_proto = defaultdict(list)
        wsum_for_label = defaultdict(int)  # 记录标签对应的总权重


        for index, protolist in self.uploaded_protos_with_index.items():
            for label, proto in protolist.items():
                aggregated_proto[label].append(proto * weight_dict[index])
                wsum_for_label[label] += weight_dict[index]
        for label, protolist in aggregated_proto.items():
            if len(protolist) > 1:
                proto = 0 * protolist[0].data
                for i in protolist:
                    proto += i.data
                aggregated_proto[label] = proto / wsum_for_label[label]
            else:
                aggregated_proto[label] = protolist[0].data

        if self.edge_protos is not None:  # 如果边缘原型非空，则将边缘原型一起融合进来
            for label, proto in aggregated_proto.items():  # 可能有label的原型仍然为空！！
                # 部分label的原型为空，需要做特殊处理。空的label直接访问会导致创建一个空的list，带来麻烦。
                if label not in self.edge_protos:
                    aggregated_proto[label] = proto  # 如果某个标签的原型还不存在就直接用上传结果赋值
                else:
                    aggregated_proto[label] = proto * 0.9 + self.edge_protos[label].data * 0.1  # 如果已有前序结果就融合

        if len(aggregated_proto) > 0:
            self.edge_protos = aggregated_proto




