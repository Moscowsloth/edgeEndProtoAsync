import torch.nn as nn
import time, copy, torch
from collections import defaultdict
import torch.nn.functional as F

from average import agg_func
from allEnds.endBase import EndBase
from allModel.model import FedAvgCNN, BaseHeadSplit


class EndProAggAsyncSimple(EndBase):
    def __init__(self, index, args, model, delay_comm):
        super().__init__(index, args, model)
        self.local_protos = None
        self.global_protos = None
        self.loss_mse = nn.MSELoss()
        self.device = args.device
        self.lamda = 2
        self.num_classes = args.num_classes

        # 时延
        self.delay_comm = delay_comm  # 通信时延
        self.delay_comp = 0  # 计算时延

        # 测试代码
        self.delay_comp_total = 0

        self.txt = ""

        # 如果有则进行转换
        if hasattr(self.model, 'fc'):
            fc_backup = copy.deepcopy(self.model.fc)
            self.model.fc = nn.Identity()
            self.model = BaseHeadSplit(self.model, fc_backup)

    def train(self):
        start_time = time.time()

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
                    local_protos[y_c].append(rep[i, :].detach().data)   # 注意，proto对数据类型就是tensor

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
        # 获得自己的局部原型
        self.local_protos = agg_func(local_protos)
        # 到自己的sender_knowledge里
        self.send_protos(self.local_protos)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        self.delay_comp = time.time() - start_time  # 记录计算时延 by Tim
        print("Computation delay of client " + str(self.index) + " is " + str(self.delay_comp))  # 打印计算时延

        # warm-up测试使用：
        self.delay_comp_total += self.delay_comp
        # print(self.delay_comp_total)

    def trainForClustering(self):   # 为了分簇做预训练
        self.model.train()
        # 局部原型
        local_protos = defaultdict(list)
        # local_protos_for_clustering = defaultdict(list)

        for epoch in range(self.local_epoch):
            for i, (img, label) in enumerate(self.train_loader):
                img = img.to(self.device)
                label = label.to(self.device)

                rep = self.model.base(img)
                # head是原始的fc层
                output = self.model.head(rep)
                loss = self.loss(output, label)

                # 全局原型和本地原型的损失
                # 对“分簇”阶段来说，这里的global原型不知道有没有影响？该以什么准则去计算loss？
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
        local_protos_for_clustering = defaultdict(list)
        local_protos_for_clustering = agg_func(local_protos)
        return local_protos_for_clustering  # 这个返回的proto是一个list，是很多个标签的proto


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

    def text2ctrl(self):
        return self.txt

