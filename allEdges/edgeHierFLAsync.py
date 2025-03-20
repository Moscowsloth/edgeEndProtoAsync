from collections import defaultdict

import torch.nn as nn
import time, copy, torch
import numpy as np

from allEdges.edgeBase import EdgeBase


class EdgeHierFLAsync(EdgeBase):
    # 边缘索引、所属端、参数、边缘模型
    def __init__(self, index, dids, args, model, epoch, period):
        super().__init__(index, dids, args, model)
        self.eval_term = args.eval_term
        self.num_classes = args.num_classes
        self.aggregated_edge = False  # 是否是聚合边缘

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

    def train(self, index, f):
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
                if self.end_isUpdated[end.index] == 1:  # 上一周期更新
                    end.train()
                    self.end_delay[end.index] = end.delay_comm + end.delay_comp
                    # 判断：是否能在本周期完成训练和上传
                    if self.end_delay[end.index] <= self.edge_period:  # 能够在周期内完成
                        # todo 如果总时延小于边缘的聚合周期，则上传原型。
                        # print("End " + str(end.index) + " is now updating its proto!")
                        self.end_isUpdated[end.index] = 1  # 设置上传标识
                    else:  # 无法在本周期完成
                        # print("End " + str(end.index) + " can not finish training!")
                        self.end_isUpdated[end.index] = 0  # 不准上传
                        self.end_delay[end.index] -= self.edge_period  # 更新延迟，减去本周期的时长
                        self.end_staleness[end.index] += 1  # staleness+1
                else:  # 上一周期没有更新
                    if self.end_delay[end.index] <= self.edge_period:  # 有staleness，但这一周期能够完成
                        # print("End " + str(end.index) + " is now updating its proto, but with staleness!!")
                        self.end_isUpdated[end.index] = 1  # 允许上传
                    else:  # 无法在本周期完成
                        # print("End " + str(end.index) + " can not finish training!")
                        self.end_isUpdated[end.index] = 0  # 不准上传
                        self.end_delay[end.index] -= self.edge_period  # 更新延迟，减去本周期的时长
                        self.end_staleness[end.index] += 1  # staleness+1

            self.edge_need_update = 0  # 更新标识符置0
            # 判断边缘是否需要更新
            for value in self.end_isUpdated.values():
                if value == 1:
                    self.edge_need_update = 1
                    print("Edge needs proto aggregation!!!")
                    break
            if self.edge_need_update == 1:
                # print("Now edge " + str(self.index) + " updating!")
                # 接收模型
                self.receive_models()
                # 聚合模型
                staleness_weight = self.calculateWeightWithStaleness()  # 计算出staleness权重并保存
                self.aggregate_parameters_with_staleness(staleness_weight)

                for end in self.ends_registration:
                    if self.end_isUpdated[end.index] == 1:
                        global_model_statedict = self.global_model.state_dict()
                        end.model.load_state_dict(global_model_statedict)
                        # for param1, param2 in zip(self.global_model.parameters(), end.model.parameters()):
                        #     if not torch.equal(param1.data, param2.data):
                        #         print("not equal!!!!!!!!==========")
                        # print("==========================")
                        # print("Aggregated model has been sent to " + str(end.index) + " in epoch " + str(cur_epoch) + " of edge " + str(self.index) + "!!!")

            cur_epoch += 1

        print(f"Communication round %d, All the ends in edge %d has trained!" % (index, self.index))
        # # 获得self.uploaded_ids、self.uploaded_weights、self.uploaded_models
        # self.receive_models()
        # # 单个边缘获得自己的全局模型
        # self.aggregate_parameters()

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
            print(f"Communication round %d, Aggregation Edge %d has sent global models to its devices" % (
            index, self.index))
            # 邻居边缘将聚合后的全局模型发给各客户端
            for neigh in self.neigh_registration:
                neigh.sender_knowledges = self.end_global_model
                neigh.send_to_ends_model()
                print(f"Communication round %d, Normal Edge %d has received global models from edge %d." % (
                index, neigh.index, self.index))

            # print(len(self.all_edge_global_model))
            #     self.edge_protos_all.append(neigh.edge_protos)
        # self.receive_models()

    def receive_models(self):
        # 总样本数的记录
        tot_samples = 0
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        for end in self.ends_registration:
            if self.end_isUpdated[end.index] == 1:
                tot_samples += end.train_samples
                self.uploaded_ids.append(end.index)
                self.uploaded_weights.append(end.train_samples)
                self.uploaded_models.append(end.model)
            else:  # 如果这一周期没有上传，则将其样本数和权重置0
                tot_samples += 0
                self.uploaded_ids.append(end.index)
                self.uploaded_weights.append(0)
                self.uploaded_models.append(end.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def calculateWeightWithStaleness(self):
        # 跟原型不一样，直接用staleness对weight进行衰减，然后用写好的聚合函数操作即可
        weight_list = []
        wsum = 0
        for end in self.ends_registration:
            if self.end_isUpdated[end.index] == 1:  # 如果已经更新
                if self.end_staleness[end.index] == 0:  # 没有staleness，权重置1
                    # print("Calculating weight from staleness... End " + str(end.index) + " does not have staleness!!")
                    weight = 1
                else:  # 有staleness，进入计算流程，得出权重
                    # print("Calculating weight from staleness... End " + str(end.index) + " has staleness of " + str(self.end_staleness[end.index]) + str("!!"))
                    weight = (self.end_staleness[end.index] + 1) ** -0.5
            else:  # 如果没有更新
                weight = 0
            wsum += weight
            # print(wsum)
            weight_list.append(weight)
        # for w in weight_list:
        #     w /= wsum   # 归一化
            # print(w)
        normalized_weight_list = [x / wsum for x in weight_list]
        # print(normalized_weight_list)
        return normalized_weight_list  # 将weightlist返回，注意，weightlist跟原本代码里面的weight列表一样，在遍历的时候乘进去就好了

    def aggregate_parameters_with_staleness(self, weight_list):  # 带有staleness参数的聚合
        print("Staleness weight list:" + str(weight_list))  # 打印staleness的权重衰减
        print("Now the length of uploaded models list is: " + str(len(self.uploaded_models)))
        # print(self.global_model)
        if self.global_model is None:
            self.global_model = copy.deepcopy(self.uploaded_models[0])
            for param in self.global_model.parameters():
                param.data.zero_()
            for w, client_model, w_staleness in zip(self.uploaded_weights, self.uploaded_models, weight_list):
                # w *= w_staleness
                self.add_parameters(w, client_model)
        else:
            with torch.no_grad():  # 将所有的参数都*0.1
                for p in self.global_model.parameters():
                    p.mul_(0.1)
            for w, client_model, w_staleness in zip(self.uploaded_weights, self.uploaded_models, weight_list):
                # w *= w_staleness
                w *= 0.9
                self.add_parameters(w, client_model)
        # for w, client_model, w_staleness in zip(self.uploaded_weights, self.uploaded_models, weight_list):
        #     w = w * w_staleness  # 给聚合权重乘上staleness参数
        #     if self.global_model is None:
        #         self.add_parameters(w, client_model)
        #     else:
        #         with torch.no_grad():  # 将所有的参数都*0.1
        #             for p in self.global_model.parameters():
        #                 p.mul_(0.1)
        #             # torch.multiply(self.global_model.parameters(), 0.1, out=self.global_model.parameters())
        #         self.add_parameters(w * 0.9, client_model)
            # for p in self.global_model.parameters():
            #     print(p)




