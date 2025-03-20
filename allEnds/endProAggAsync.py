from datetime import datetime

import torch.nn as nn
import time, copy, torch
from collections import defaultdict
import torch.nn.functional as F
from numpy import dtype

from average import agg_func
from allEnds.endBase import EndBase
from allModel.model import FedAvgCNN, BaseHeadSplit
from allEnds.updateInfo import UpdateInfo


class EndProAggAsync(EndBase):
    def __init__(self, index, args, model, delay):
        super().__init__(index, args, model)
        self.local_protos = None
        self.global_protos = None
        self.loss_mse = nn.MSELoss()
        self.device = args.device
        self.lamda = 2
        self.num_classes = args.num_classes
        self.delay = delay
        self.info = None

        self.txt = ""

        # 如果有则进行转换
        if hasattr(self.model, 'fc'):
            fc_backup = copy.deepcopy(self.model.fc)
            self.model.fc = nn.Identity()
            self.model = BaseHeadSplit(self.model, fc_backup)

    def trainAsync(self, info_queue, sig):
        if sig == 1:
            while True:
                self.train(info_queue)
        else:
            return

    def train(self, info_queue):    #重构，将之前的两个queue换成一个info-queue

        torch.cuda.init()
        start_time = time.time()
        print("End " + str(self.index) + " start training!!!")
        print(datetime.fromtimestamp(time.time()))


        self.model.train()
        # 局部原型
        local_protos = defaultdict(list)

        for epoch in range(self.local_epoch):
            for i, (img, label) in enumerate(self.train_loader):
                img = img.to(self.device)
                label = label.to(self.device)

                rep = self.model.base(img)
                # head是原始的fc层
                output = self.model.head(rep)
                loss = self.loss(output, label)

                # 全局原型和本地原型的损失
                if self.global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, y in enumerate(label):
                        y_c = y.item()
                        if type(self.global_protos[y_c]) != type([]):
                            proto_new[i, :] = self.global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda
                # 添加局部原型
                for i, y in enumerate(label):
                    y_c = y.item()
                    local_protos[y_c].append(rep[i, :].detach().data)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
        # 获得自己的局部原型
        self.local_protos = agg_func(local_protos)
        # 到自己的sender_knowledge里
        self.send_protos(self.local_protos)
        # print(self.local_protos.shape)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        self.txt = "clientID: " + str(self.index) + ", uploaded after comm delay_comm of " + str(self.delay)
        # print(self.txt, flush=True)
        time.sleep(self.delay)

        # todo 现在就要看这个average函数，能否接受numpy数组？

        # 将当前的信息（proto+index）放入info_queue
        self.asyncEndUpdate(info_queue)

        torch.cuda.empty_cache()

    def end_id(self):
        return self.index

    def end_proto_len(self):
        return len(self.local_protos)

    # 测试端侧模型效果（结合）
    # def test_metrics(self):
    #     self.model.eval()

    #     test_acc = 0
    #     test_num = 0
    #     y_prob = []
    #     y_true = []

    #     with torch.no_grad():
    #         for x, y in self.test_loader:
    #             x = x.to(self.device)
    #             y = y.to(self.device)
    #             output = self.model(x)
    #             # 总样本数
    #             test_num += y.shape[0]
    #             # 用模型推理，结果是output，10*10向量
    #             probabilities = F.softmax(output, dim=1)

    #             test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()

    #             # 用原型
    #             rep = self.model.base(x)
    #             output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
    #             # 遍历每个样本对应的原型
    #             for i, r in enumerate(rep):
    #                 # 比较样本对应原型 和 全局原型的距离
    #                 for j, pro in self.global_protos.items():
    #                     if type(pro) != type([]):
    #                         output[i, j] = self.loss_mse(r, pro)

    #             test_acc += (torch.sum(torch.argmin(output, dim=1) == y)).item()
    #     return test_acc, test_num, 0

    def test_metrics(self):
        self.model.eval()

        test_acc = 0
        test_num = 0

        if self.global_protos is not None:
            with torch.no_grad():
                for x, y in self.test_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    # 获得原型
                    rep = self.model.base(x)
                    output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
                    # 遍历每个样本对应的原型
                    for i, r in enumerate(rep):
                        # 比较样本对应原型r 和 全局原型的距离
                        for j, pro in self.global_protos.items():
                            if type(pro) != type([]):
                                output[i, j] = self.loss_mse(r, pro)

                    test_acc += (torch.sum(torch.argmin(output, dim=1) == y)).item()
                    test_num += y.shape[0]

            return test_acc, test_num, 0
        else:
            return 0, 1e-5, 0

    def train_metrics(self):
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                if self.global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.global_protos[y_c]) != type([]):
                            proto_new[i, :] = self.global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        return losses, train_num

    # 收到全局原型
    def receive_protos(self, global_protos):
        self.receiver_knowledges = global_protos
        self.global_protos = self.receiver_knowledges

    # 收到局部原型
    def send_protos(self, local_protos):
        self.sender_knowledges = local_protos

    # 注意，每次训练结束，localproto都会被存放在self.local_protos里面
    # saveProto是一个信息类，将localproto和index存放在一个UpdateInfo类中

    # info_queue是clientPool中存储客户端信息（proto+index）的共享队列
    def asyncEndUpdate(self, info_queue):
        info_npy = {}
        for key, proto in self.local_protos.items():  # 遍历proto，将每个tensor都转为numpy数组
            proto_npy = proto.clone().detach().cpu().numpy()
            info_npy[key] = proto_npy

        info = UpdateInfo(info_npy, self.index)
        print("End " + str(self.index) + " updating after delay_comm of " + str(self.delay) + "!!")
        print("info index: " + str(info.index))
        info_queue.put(info)
        print("put info into info_queue, end index: " + str(info.index))
