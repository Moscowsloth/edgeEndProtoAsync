from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
from data.data_utils import read_end_data
from sklearn.preprocessing import label_binarize
from torch.optim import lr_scheduler
import numpy as np
from sklearn import metrics

class EndBase(object):
    def __init__(self, index, args, model):
        self.index = index
        self.train_loader = self.load_train_data(args.dataset, self.index, args.batch_size)
        self.test_loader = self.load_test_data(args.dataset, self.index, args.batch_size)
        self.model = model
        self.batch_size = args.batch_size
        self.train_samples = len(self.train_loader) * self.batch_size
        self.loss = nn.CrossEntropyLoss()
        self.local_epoch = args.num_local_training
        self.learning_rate = args.lr
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # 学习率衰减
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[20,40,60,80,100], gamma=0.9)
        self.parent = None  # 父节点初始化为空
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        # 接收的知识 和 要发送的知识
        self.receiver_knowledges = {}
        self.sender_knowledges = {}

    def receive_from_edge(self, global_knowledge):
        self.receiver_knowledges = global_knowledge
        return None
    
    def receive_model_from_edge(self):
        self.model = self.receiver_knowledges

    def send_to_edge(self):
        self.parent.receiver_knowledges[self.index] = self.sender_knowledges
        return None
    
    # 加载数据，read_client_data在data_utils文件里
    def load_train_data(self, dataset, index, batch_size=None):
        train_data = read_end_data(dataset, index, is_train=True)

        # =============================================验证实验使用
        # 对数据进行20%采样，以此来尝试确定warm-up的时间开销
        # num_samples = int(len(train_data) * 0.2)
        # indices = torch.randperm(len(train_data))[:num_samples]
        # subset_train_data = Subset(train_data, indices)
        # =============================================验证实验使用

        # return DataLoader(subset_train_data, batch_size, drop_last=True, shuffle=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)
    
    def load_test_data(self, dataset, index, batch_size=None):
        test_data = read_end_data(dataset, index, is_train=False)

        # =============================================验证实验使用
        # 对数据进行20%采样，以此来尝试确定warm-up的时间开销
        # num_samples = int(len(test_data) * 0.2)
        # indices = torch.randperm(len(test_data))[:num_samples]
        # subset_test_data = Subset(test_data, indices)
        # =============================================验证实验使用

        # return DataLoader(subset_test_data, batch_size, drop_last=False, shuffle=True)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)


    def test_metrics(self):
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        return test_acc, test_num, auc

    def train_metrics(self):
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
