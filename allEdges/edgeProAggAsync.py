import sys
from collections import defaultdict
from datetime import datetime

import torch.nn as nn
import time, copy, torch
import numpy as np

import torch.multiprocessing
from torch import multiprocessing
from torch.multiprocessing import Pool, Manager
import queue
import schedule

import average
from allEdges.edgeBase import EdgeBase
from average import proto_aggregation, edge_proto_aggregation
from allEnds.endProAgg import EndProAgg
from allEnds.endProAggAsync import EndProAggAsync
from allEnds.updateInfo import UpdateInfo

import threading


class EdgeProAggAsync(EdgeBase):
    # 边缘索引、所属端、参数、边缘模型
    def __init__(self, index, dids, args, model, edgeepoch):
        super().__init__(index, dids, args, model)

        self.edge_protos = None
        self.eval_term = args.eval_term
        self.num_classes = args.num_classes
        self.aggregated_edge = False  # 是否是聚合边缘
        self.global_protos = []
        self.uploaded_ids = []
        self.uploaded_protos = []
        self.edge_epoch = edgeepoch
        self.current_epoch = 0  # 这个是边缘的更新轮次
        self.edge_T = args.edge_t  # 这个是边缘的更新周期
        # todo 留出了接口，则需要在每次全局训练时给每个边缘分配合适的更新周期

        # 测试和训练记录
        self.rs_test_acc = []
        self.rs_train_loss = []
        self.update_time = 0

        # 创建end_state字典，方便计算staleness
        self.end_index = []
        self.ends_state = {}
        self.isUpdated = 0
        self.updatedInfoList = []
        # for end in self.ends_registration:
        #     self.end_index.append(end.index)
        # self.ends_state = {i: -1 for i in self.end_index}
        # print("Client state test:")
        # print(self.ends_state)
    # todo: 第一件事：捋顺代码逻辑 一个是训练函数需要确定 一个是update函数需要确定 还有一个问题 没完成的训练能否继续进行？
    def asyn_train(self, index, f):
        # 开始时间
        s_time = time.time()
        print("train async!")       # 打印开始信息
        print("##################################")
        # 测试
        if index % self.eval_term == 0:
            edge_index, train_loss_list, test_acc_list, std_accs_list = self.all_evaluate(index)
            f.write("communication " + str(index) + " :\n")
            f.write("Edge " + str(self.index) + " :\n")
            f.write("train_loss " + str(train_loss_list) + "\n")
            f.write("test_acc " + str(test_acc_list) + "\n")
            f.write("std_accs " + str(std_accs_list) + "\n")
            f.write("\n")

        # manager = Manager()
        # client_pool = multiprocessing.Pool(len(self.ends_registration))
        # client_info = manager.Queue()  # 存储客户端信息
        # client_proto_list = manager.Queue()  # 原型列表

        self.edgeStart()

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

        if index % 10 == 0 and len(self.global_protos) > 0:
            pass

    # 边缘从所属客户端接收原型
    def receive_protos(self):
        for client in self.ends_registration:
            self.uploaded_ids.append(client.index)
            self.uploaded_protos.append(client.sender_knowledges)
            # 把客户端的原型加入边缘的原型队列
            # self.async_queue.put(client.sender_knowledges)

    def async_receive_protos(self):
        self.uploaded_protos = []
        self.uploaded_ids = []
        for client in self.ends_registration:
            self.uploaded_ids.append(client.index)
            self.uploaded_protos.append(client.sender_knowledges)

    # def edge_start(self):
    #     with Manager() as manager:
    #         print("Manager ready!!")
    #         client_pool = multiprocessing.Pool(len(self.ends_registration))
    #         # client_info = manager.Queue()  # 存储客户端信息
    #         # client_proto_list = manager.Queue()  # 原型列表
    #         info_queue = manager.Queue()    # 存储各个client的index与proto
    #         print("client_pool ready!!")
    #
    #         schedule.every(5).seconds.do(self.update, client_proto_list, client_info)  # 每5秒钟检查更新
    #         threading.Thread(target=self.check_update, daemon=True).start()
    #         print("updating mission assigned!")
    #
    #
    #         with client_pool:
    #             while self.current_epoch < self.edge_epoch:  # 规定边缘-端层级的循环次数
    #                 stmp_args = []
    #                 for end in self.ends_registration:
    #                     print("end index: " + str(end.index))
    #                     # 创建参数列表
    #                     tmp_arg = [end, client_proto_list, client_info]
    #                     stmp_args.append(tmp_arg)
    #                 client_pool.starmap_async(EndProAggAsync.train, stmp_args)
    #                 self.current_epoch += 1
    #                 print("async training epoch: " + str(self.current_epoch))
    #             input("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    #             # 到这里，完成了一个边缘轮次的训练，并将训练结果存储在共享变量中

    def edgeStart(self):
        with Manager() as manager:
            print("Multiprocessing manager ready!!!")
            torch.cuda.init()
            client_pool = multiprocessing.Pool(len(self.ends_registration))  # 创建进程池，最大的进程数量就是注册的端设备数量
            print("Client pool ready!!!")

            # ends_state记录每个end当前update的次数，以此计算staleness，这里需要字典来实现，因为end的index不是从0开始的
            # 将ends_state设置为成员变量，方便访问
            cur_epoch = 0  # 记录当前的边缘轮，后面每次update就+=1
            print("End state:")
            print(self.ends_state)

            info_queue = manager.Queue()    # info_queue 是存储updateInfo的共享队列
            print("Shared queue for client pool ready!!!")

            client_inst_list = manager.list()
            client_inst_list.extend(self.ends_registration)  # 将每个端设备添加到共享列表中
            for end in client_inst_list:
                end.train_loader.dataset.share_memory_()
                end.test_loader.dataset.share_memory_()
                print("123123321321123321")
            with client_pool:
                print("Cuda state: " + str(torch.cuda.is_available()))
                while cur_epoch < self.edge_epoch:  # 循环指定的边缘轮次
                    starmap_args = []  # 准备参数列表，方便使用starmap进行并行操作
                    for end in client_inst_list:
                        print("The index of the end in shared list is: " + str(end.index))
                        # 如果state值为-1，说明在上一轮上传了更新。则参与下一轮训练，同时将state置为cur_epoch。
                        if self.ends_state[end.index] == -1:
                            # 只允许上传过的end参与下一轮训练，只将符合要求的参数
                            tmp_args = [end, info_queue]
                            starmap_args.append(tmp_args)
                            self.ends_state[end.index] = cur_epoch
                            print("Training mission for end " + str(end.index) + " has been assigned!!")
                            print(datetime.fromtimestamp(time.time()))
                    # 启动异步训练
                    client_pool.starmap_async(EndProAggAsync.train, starmap_args)
                    cur_epoch += 1
                    time.sleep(self.edge_T)  # 睡一下，这里模拟的是更新周期
                    # todo 这里开始update
                    self.update(info_queue)
                    # 更新后，本轮上传过原型的端设备直接获得边缘原型，进一步训练。
                    for end in self.ends_registration:
                        if self.isUpdated == 1:
                            if self.ends_state[end.index] == -1:  # update后，state被置为-1
                                end.receive_protos(self.edge_protos)
                                print("End " + str(end.index) + " received an edge proto!!")


                # todo: client_pool 训练结束后关闭的问题需要进一步设计

    # todo 端设备update后，边缘怎么将自己原有的proto跟它们聚合？
    # todo 每次下发的proto是什么？是先聚合后下发，还是直接下发之前的proto？
    def update(self, info_queue):  # info_queue 是edgeStart里面创建的，不是
        print("async updating! current length of proto queue: ")
        if info_queue.empty():  # 检查infoqueue是否空，如果空，打印信息
            # input("Nothing in the shared QUEUE!")
            self.isUpdated = 0
        else:
            self.isUpdated = 1
        # proto_list = []  # 准备一个原型列表，方便后面直接聚合 ---不用了
        updated_info_list = []  # 不用原型列表了，用info对象的列表，存放了index和proto

        while not info_queue.empty():
            info = info_queue.get()  # 出队一个info对象
            proto = self.infoMemoryView2Tensor(info.proto)  # 根据这个info对象的proto创建一个proto
            new_info = UpdateInfo(proto, info.index)  # 创建一个新的info对象
            updated_info_list.append(new_info)
            self.ends_state[info.index] = -1  # 别忘了给state置-1
            print("Info object has been moved to 'updated_info_list'!!!!")
            info_queue.task_done()

        # while not info_queue.empty():
        #     info = info_queue.get()
        #     proto = info.proto
        #     index = info.index
        #     proto_list.append(proto)
        #     print("Getting proto of end with index " + str(index) + "!!")
        #     print("The type of the proto is " + str(type(proto)))
        #     for value in proto.values():
        #         print(type(value))
        #     # print(index)
        #     self.ends_state[index] = -1  # 上传完毕的end，获得-1的state
        #     info_queue.task_done()

        # proto_list = self.listMemoryView2Numpy(proto_list)  # 将proto list转换成tensor
        # print("The length of proto list is: " + str(len(proto_list)))
        # self.edge_protos = proto_aggregation(proto_list)    # 已更新，不用

        if self.isUpdated == 1:
            self.edge_protos = self.weightedProtoAggeregation(updated_info_list)
        print("Now printing the aggregated protos!!!!!=================================")
        print(self.edge_protos)
        # input("Proto has been aggregated!!!")
        # print(self.ends_state)  # 打印当前的ends状态
    # todo: 没有为下发设计机制，应该设计到update里面。
    # todo 下一步还要设计

    def listMemoryView2Numpy(self, list):
        proto_tensor_list = []
        for item in list:
            proto_dict_tensor = {}
            for key, value in item.items():
                print("Now transforming!!!")
                proto_array = np.frombuffer(value, dtype=np.float32)
                proto_tensor = torch.from_numpy(proto_array)
                proto_dict_tensor[key] = proto_tensor
            proto_tensor_list.append(proto_dict_tensor)
        print("Transforming from numpy to tensor!!!")
        # print(proto_tensor_list)
        # print(len(proto_tensor_list))
        return proto_tensor_list

    def infoMemoryView2Tensor(self, updated_info):
        proto_tensor_dict = {}
        for key, value in updated_info.items():
            # print("Transforming from numpy to tensor!!!")
            proto_array = np.frombuffer(value, dtype=np.float32)
            proto_tensor = torch.from_numpy(proto_array)
            proto_tensor_dict[key] = proto_tensor
        # print(proto_tensor_dict)
        return proto_tensor_dict

    # 别忘了还要跟边缘模型进行聚合！！！
    # updated_info_list里面存放的是info对象,是一个列表
    # 返回值是一个defaultdict(list)，跟enze的代码保持一致
    def weightedProtoAggeregation(self, updated_info_list):
        weight_dict = {}  # 权重dict
        aggregated_proto = defaultdict(list)  # 准备好聚合后的原型dict
        for info in updated_info_list:  # 遍历list中的info对象
            # 通过staleness计算权重，如果state为-1则置权重为1，否则按照FedAsync中的函数来计算
            if self.ends_state[info.index] == -1:  # 按照info中的index属性去查看state
                weight_dict[info.index] = 1  # 这里是无staleness的权重
            else:
                # 这里用函数计算权重
                staleness = self.edge_epoch - self.ends_state[info.index]
                print("The staleness of end " + str(info.index) + "is: " + str(staleness))
                weight_dict[info.index] = (staleness + 1) ** -0.5   # 使用FedAsync的加权函数
        print(weight_dict)
        w_sum = 0
        for w in weight_dict.values():
            w_sum += w
        print(w_sum)
        # weight_dict已经准备好，下面先聚合update上来的proto
        # 准备好原型dict，一个dict中存放了所有的原型，每个标签对应一个列表，列表中是来自所有端的原型
        for info in updated_info_list:
            for label, proto in info.proto.items():
                aggregated_proto[label].append(proto * weight_dict[info.index])
        for label, proto_list in aggregated_proto.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                aggregated_proto[label] = proto / w_sum
            else:
                aggregated_proto[label] = proto_list[0].data
        # aggregated_proto准备好了，里面存放着加权聚合后的端原型
        print("The type of self.edge_protos is: " + str(type(self.edge_protos)))
        if self.edge_protos is not None:
            print("The type of self.edge_protos is: " + str(type(self.edge_protos)))
            # print("The type of self.edge_protos is: " + str(type(self.edge_protos.get(label))))
            keys = self.edge_protos.keys()
            print("length of the keys: " + str(len(keys)))
            print(keys)
            for label, proto in aggregated_proto.items():
                aggregated_proto[label] = proto * 0.9 + self.edge_protos[label].data * 0.1
        print(aggregated_proto)
        return aggregated_proto

    # def clientPoolInit(self):



