import copy
import torch
from collections import defaultdict
from torch import nn

# 模型聚合
def average_weights(w, s_num):
    #copy the first client's weights
    total_sample_num = sum(s_num)
    temp_sample_num = s_num[0]
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():  #the nn layer loop
        for i in range(1, len(w)):   #the client loop
            w_avg[k] += torch.mul(w[i][k], s_num[i]/temp_sample_num)
        w_avg[k] = torch.mul(w_avg[k], temp_sample_num/total_sample_num)
    return w_avg

# 原型聚合
def agg_func(protos):
    """
    Returns the average of the weights.
    """
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]
    return protos

# def agg_func(protos, weights):
    '''根据权重聚合，protos是原型列表，weights是权重列表'''


# 局部原型到边缘原型
def proto_aggregation(local_protos_list):   # 存放了多个proto的list
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label

# 边缘原型到全局原型
def edge_proto_aggregation(edge_protos_all):
    # print(len(edge_protos_all))
    agg_protos_label = defaultdict(list)
    for edge_protos in edge_protos_all:
        if edge_protos is not None:
            for label in edge_protos.keys():
                agg_protos_label[label].append(edge_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data
    return agg_protos_label