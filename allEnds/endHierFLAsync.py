import torch.nn as nn
import time, copy, torch
from collections import defaultdict

from average import agg_func
from allEnds.endBase import EndBase
from allModel.model import FedAvgCNN


class EndHierFLAsync(EndBase):
    def __init__(self, index, args, model, delay_comm):
        super().__init__(index, args, model)
        self.device = args.device
        self.lamda = 1
        self.loss = nn.CrossEntropyLoss()
        self.num_classes = args.num_classes

        # 时延
        self.delay_comm = delay_comm  # 通信时延
        self.delay_comp = 0

    def train(self):
        self.model.train()

        start_time = time.time()
        max_local_epochs = self.local_epoch
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        self.delay_comp = time.time() - start_time


