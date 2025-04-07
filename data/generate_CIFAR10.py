import torchvision.transforms as transforms
import torchvision
import numpy as np
import random, sys, os
import torch
from sklearn.model_selection import train_test_split

from itertools import combinations

random.seed(1)
np.random.seed(1)
num_clients = 80  # 客户端数量
dir_path = "cifar10/"
batch_size = 10
train_ratio = 0.75 # merge original training set and test set, then split it manually. 
alpha = 0.1 # for Dirichlet distribution. 100 for exdir

# 参数：数据目录、端数目、数据划分方式、是否平衡、partition方法
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    # Setup directory for train/test data
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 获得数据集，用trainloader和testloader加载
    trainset = torchvision.datasets.CIFAR10(root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)

    # 将train_data和test_data拆分为data和targets
    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    # 汇总训练集 和 测试集 数据图片
    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    # 汇总训练集 和 测试集 数据标签
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # seperate数据
    # X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    # niid, balance, partition, class_per_client=2)

    '''==========================================='''
    '''这里是新的数据生成方式'''
    distributions = [
        [0, 1, 3],  # 分布1
        [2, 4, 5],  # 分布2
        [6, 9],  # 分布3
        [7, 8]  # 分布4
    ]

    # 随机分配每个客户端一种分布
    client_distribution_map = {client: np.random.randint(len(distributions)) for client in range(80)}
    X, y, statistic = separate_data_new(
        (dataset_image, dataset_label),
        num_clients=80,
        num_classes=10,
        niid=True,
        balance=False,
        partition='pat',
        class_per_client=3,
        distributions=distributions,
        client_distribution_map=client_distribution_map
    )

    for client in range(num_clients):
        print(
            f"Client {client}\t Distribution: {client_distribution_map[client]}\t Size of data: {len(X[client])}\t Labels: ",
            np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    # X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
    #                                 niid, balance, partition, class_per_client=2)
    # 分割数据
    train_data, test_data = split_data(X, y)
    # 保存文件
    save_file(train_path, test_path, train_data, test_data, num_clients, num_classes,
        statistic, niid, balance, partition)

# 划分数据集到各个端
def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=None):
    # 用于保存每个客户端的X、y和statistic
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]
    # 数据集内容、数据集标签
    dataset_content, dataset_label = data
    # guarantee that each client must have at least one batch of data for testing. 
    # 记录int(10/0.25, 60000/20/2)
    least_samples = int(min(batch_size / (1-train_ratio), len(dataset_label) / num_clients / 2))

    # 记录客户端的dataidx
    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            selected_clients = selected_clients[:int(np.ceil((num_clients/num_classes)*class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
            else:
                num_samples = np.random.randint(max(num_per/10, least_samples/num_classes), num_per, num_selected_clients-1).tolist()
            num_samples.append(num_all_samples-sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1
    # dir划分方法
    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        # min_size是每个客户端分配到的最小样本数，K是类别数目，N是训练和测试数据集总和
        min_size = 0
        K = num_classes
        N = len(dataset_label)
        # 跟踪分配尝试的次数，初始值设为1
        try_cnt = 1
        while min_size < least_samples:
            # 当前客户端数据的大小不满足最小要求，提示正在进行第几次重新分配
            if try_cnt > 1:
                print(f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')
            # 二维列表idx_batch，存储每个客户端被分配到的样本的索引
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                # 获得标签为k的样本索引idx_k并打乱
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                # 生成一个Dirichlet分布，用于确定每个客户端所占比例的随机向量
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                # 根据每个客户端当前已分配到的样本数，调整Dirichlet分布向量中的值，确保每个客户端接收到的样本数量接近平均值
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
                # 归一化，根据比例计算每个客户端应该分配的样本数目，转换为整数类型
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                # 根据计算的样本数目，将样本索引分配给每个客户端
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                # 更新当前客户端分配的最小样本数
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1
        # 记录客户端对应的dataindex
        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        # 数据索引idxs，获得对应的内容和标签，存入X和y里
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]
        # 遍历所有标签，索引为i，记录标签i对应的样本数目到statistic里
        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))
            

    del data

    # 输出端的样本大小、标签列表
    # 每个标签的样本数目
    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic


import numpy as np


def separate_data_new(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=None,
                  distributions=None, client_distribution_map=None):
    """
    分配数据给客户端，支持指定每个客户端的类别分配。

    参数:
    - data: 数据集内容和标签
    - num_clients: 客户端数量
    - num_classes: 类别数量
    - niid: 是否采用非独立同分布（NIID）分配方式
    - balance: 是否平衡分配数据
    - partition: 数据分配方式（'pat' 或 'dir'）
    - class_per_client: 每个客户端分配的类别数量（适用于 'pat' 分区方式）
    - distributions: 定义的分布种类（列表，每个元素是一个类别列表）
    - client_distribution_map: 指定每个客户端的分布索引（字典，键为客户端索引，值为分布索引）

    返回:
    - X: 每个客户端的数据
    - y: 每个客户端的标签
    - statistic: 每个客户端的标签统计信息
    """
    # 用于保存每个客户端的X、y和statistic
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]
    # 数据集内容、数据集标签
    dataset_content, dataset_label = data
    # guarantee that each client must have at least one batch of data for testing.
    # 记录int(10/0.25, 60000/20/2)
    least_samples = int(min(batch_size / (1 - train_ratio), len(dataset_label) / num_clients / 2))

    # 记录客户端的dataidx
    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        # 初始化每个客户端的类别计数
        class_count_per_client = [0] * num_clients

        for i in range(num_classes):
            # 获取当前类别的分布
            current_distribution_idx = None
            for dist_idx, dist in enumerate(distributions):
                if i in dist:
                    current_distribution_idx = dist_idx
                    break

            # 选择属于当前分布的客户端
            selected_clients = []
            for client in range(num_clients):
                if client_distribution_map[client] == current_distribution_idx:
                    selected_clients.append(client)

            if not selected_clients:
                continue  # 如果没有客户端需要这个类别，跳过

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients)]
            else:
                # 确保每个客户端至少有最少样本数
                min_samples_per_client = max(int(num_per / 10), least_samples // num_classes)
                num_samples = np.random.randint(min_samples_per_client, num_per, num_selected_clients - 1).tolist()
                num_samples.append(num_all_samples - sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map:
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample],
                                                    axis=0)
                idx += num_sample
                class_count_per_client[client] += 1  # 更新类别计数

    # dir划分方法
    elif partition == "dir":
        # <url id="cvmqa5b3jih919hkgh90" type="url" status="parsed" title="probabilistic-federated-neural-matching/experiment.py at master · IBM/probabilistic-federated-neural-matching" wc="98">https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py</url>
        # min_size是每个客户端分配到的最小样本数，K是类别数目，N是训练和测试数据集总和
        min_size = 0
        K = num_classes
        N = len(dataset_label)
        # 跟踪分配尝试的次数，初始值设为1
        try_cnt = 1
        while min_size < least_samples:
            # 当前客户端数据的大小不满足最小要求，提示正在进行第几次重新分配
            if try_cnt > 1:
                print(
                    f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')
            # 二维列表idx_batch，存储每个客户端被分配到的样本的索引
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                # 获得标签为k的样本索引idx_k并打乱
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                # 生成一个Dirichlet分布，用于确定每个客户端所占比例的随机向量
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                # 根据每个客户端当前已分配到的样本数，调整Dirichlet分布向量中的值，确保每个客户端接收到的样本数量接近平均值
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                # 归一化，根据比例计算每个客户端应该分配的样本数目，转换为整数类型
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # 根据计算的样本数目，将样本索引分配给每个客户端
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                # 更新当前客户端分配的最小样本数
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1
        # 记录客户端对应的dataindex
        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # 确保所有客户端都被分配到数据
    for client in range(num_clients):
        if client not in dataidx_map:
            dataidx_map[client] = np.array([])

    # assign data
    for client in range(num_clients):
        # 数据索引idxs，获得对应的内容和标签，存入X和y里
        idxs = dataidx_map[client]
        idxs = idxs.astype(int)  # 确保idxs是整数类型
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]
        # 遍历所有标签，索引为i，记录标签i对应的样本数目到statistic里
        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))

    # 检查client_distribution_map是否有效
    if client_distribution_map is not None:
        for client in range(num_clients):
            if client not in client_distribution_map:
                print(f"警告：客户端 {client} 没有分配分布，将随机分配一个分布。")
                client_distribution_map[client] = np.random.randint(len(distributions))
            # 确保每个客户端的分布索引有效
            if client_distribution_map[client] < 0 or client_distribution_map[client] >= len(distributions):
                print(f"警告：客户端 {client} 分配的分布索引无效，将随机分配一个分布。")
                client_distribution_map[client] = np.random.randint(len(distributions))

    # 输出端的样本大小、标签列表
    # 每个标签的样本数目
    for client in range(num_clients):
        print(
            f"Client {client}\t Distribution: {client_distribution_map[client]}\t Size of data: {len(X[client])}\t Labels: ",
            np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic


# 划分训练集 和 测试集
def split_data(X, y):
    # 记录每个客户端的训练数据、每个客户端的测试数据
    train_data, test_data = [], []
    num_samples = {'train':[], 'test':[]}

    # i是第i个客户端对应的标签列表，train_ratio是总数据划分训练和测试的比例
    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_ratio, shuffle=True)

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    del X, y

    return train_data, test_data

def save_file(train_path, test_path, train_data, test_data, num_clients, 
                num_classes, statistic, niid=False, balance=True, partition=None):
    config = {
        'num_clients': num_clients, 
        'num_classes': num_classes, 
        'non_iid': niid, 
        'balance': balance, 
        'partition': partition, 
        'Size of samples for labels in clients': statistic, 
        'alpha': alpha, 
        'batch_size': batch_size, 
    }
    print("Saving to disk.\n")

    # 保存train_dict和test_dict
    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)

    print("Finish generating dataset.\n")


if __name__ == "__main__":
    # 参数从0到3依次为 generate_CIFAR10.py、noniid、-、dir
    # 非iid、是否平衡
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    # dis = generate_distributions(10, 4, 3, 5)
    # print("生成的分布:")
    # for i, dist in enumerate(dis):
    #     print(f"分布 {i + 1}: {dist}")

    generate_dataset(dir_path, num_clients, niid, balance, partition)