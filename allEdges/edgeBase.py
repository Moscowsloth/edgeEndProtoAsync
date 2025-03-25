from average import average_weights
import torch, copy
import numpy as np

class EdgeBase(object):
    def __init__(self, index, dids, args, model):
        self.index = index  # 边缘的id
        self.dids = dids    # 范围内端的id----deviceID的意思
        self.ends_registration = []  # 当前轮次参与的客户端对象
        self.model = model  # 边缘模型 
        self.end_global_model = None  # 聚合后的端模型
        self.batch_size = args.batch_size
        self.local_epoch = args.num_local_training
        self.learning_rate = args.lr
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.neigh_registration = []  # 邻居边缘节点初始化为空
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.generated_model = None     # 生成模型
        self.number_samples_edges = {}  # 聚合边缘记录所有边缘拥有的数据
        self.number_samples_ends = 0 # 当前边缘记录自己所属客户端的数据
        # 传输的知识
        self.receiver_knowledges = {}
        self.sender_knowledges = None
        self.sender_models = {}
        # 接收模型的参数， 记录端的id 权重 模型
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.global_model = None

        self.uploaded_edge_ids = []
        self.uploaded_edge_weights = []
        self.uploaded_edge_models = []

    # 注册当前边缘下所有端
    def end_register(self, end):
        self.ends_registration.append(end)
    
    # # 聚合边缘服务器调用，从其他边缘那获得模型
    # def receive_from_other_edges(self, edge_id, shared_state_dict):
    #     self.receiver_models[edge_id] = shared_state_dict
    #     return None

    # # 聚合边缘服务器调用，模型聚合
    # def aggregate_model(self):
    #     received_dict = [dict for dict in self.receiver_models.values()]
    #     for edge in range(self.neigh_registration):
    #         self.number_samples_edges[edge.index] = sum(self.samples_end.values())
    #     self.model = average_weights(w = received_dict, s_num = self.number_samples_edges)
    
    # 聚合后的模型发给其他边缘
    def send_to_other_edges(self, model):
        # self.model是边缘自己的模型，model是收到的模型，用model更新self.model
        for param, new_param in zip(self.model.parameters(), model.parameters()):
            param.data = new_param.data.clone()

    # 边缘发给所属的端 自己的聚合知识（如全局原型）
    def send_to_ends(self):
        for end in self.ends_registration:
            end.receive_from_edge(self.sender_knowledges)

    # 边缘发给所属的端 自己的模型
    def send_to_ends_model(self):
        for end in self.ends_registration:
            global_state_dict = self.sender_knowledges.state_dict()
            end.model.load_state_dict(global_state_dict) 

    # 边缘收到端传递的知识（如局部原型）
    def receive_from_ends(self, end):
        for end in range(self.ends_registration):
            end.send_to_edge()

    # 从端侧接收模型
    def receive_models(self):
        # 总样本数的记录
        tot_samples = 0
        for end in self.ends_registration:
            tot_samples += end.train_samples
            self.uploaded_ids.append(end.index)
            self.uploaded_weights.append(end.train_samples)
            self.uploaded_models.append(end.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    # 从边侧接收模型，为聚合做准备
    def receive_models_from_edges(self):
        tot_samples = 0
        self.uploaded_edge_ids = []
        self.uploaded_edge_weights = []
        self.uploaded_edge_models = []
        tot_samples += self.number_samples_ends
        self.uploaded_edge_ids.append(self.index)
        self.uploaded_edge_weights.append(self.number_samples_ends)
        self.uploaded_edge_models.append(self.global_model)
        for edge in self.neigh_registration:
            tot_samples += edge.number_samples_ends
            self.uploaded_edge_ids.append(edge.index)
            self.uploaded_edge_weights.append(edge.number_samples_ends)
            self.uploaded_edge_models.append(edge.global_model)
        for i, w in enumerate(self.uploaded_edge_weights):
            self.uploaded_edge_weights[i] = w / tot_samples
    # 聚合端模型
    def aggregate_parameters(self):
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)
    # 聚合边缘模型
    def aggregate_parameters_edge(self):
        # self.end_global_model = average_weights(self.uploaded_edge_models, self.uploaded_edge_weights)
        self.end_global_model = copy.deepcopy(self.uploaded_edge_models[0])
        for param in self.end_global_model.parameters():
            param.data.zero_()
        for w, client_model in zip(self.uploaded_edge_weights, self.uploaded_edge_models):
            self.add_parameters_edge(w, client_model)
    # 遍历 服务器 和 客户端参数
    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w
    # 遍历
    def add_parameters_edge(self, w, client_model):
        for server_param, client_param in zip(self.end_global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    # 测试度量
    def test_metrics(self):        
        num_samples = []
        tot_correct = []
        tot_auc = []
        # 返回每个端的正确数目、总数、以及ruc曲线的内容
        for c in self.ends_registration:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.index for c in self.ends_registration]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):        
        num_samples = []
        losses = []
        for c in self.ends_registration:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.index for c in self.ends_registration]

        return ids, num_samples, losses
    

    # 单个边缘的评估函数
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        return train_loss, test_acc, np.std(accs)

    def all_evaluate(self,index):
        print(f"\nEvaluate ends models in edge %d" % self.index)
        
        # 边缘索引和对应结果
        indexs = []
        train_loss_list = []
        test_acc_list = []
        std_accs_list = []

        train_loss, test_acc, std_accs = self.evaluate()
        indexs.append(self.index)
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
        std_accs_list.append(std_accs)

        # for neigh in self.neigh_registration:
        #     print(f"\nEvaluate ends models in Edge %d" % neigh.index)
        #     train_loss, test_acc, std_accs = neigh.evaluate()
        #     indexs.append(self.index)
        #     train_loss_list.append(train_loss)
        #     test_acc_list.append(test_acc)
        #     std_accs_list.append(std_accs)
        print(f"\n-----------------------------------------------")
        return indexs, train_loss_list, test_acc_list, std_accs_list