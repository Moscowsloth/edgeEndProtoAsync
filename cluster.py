from collections import defaultdict

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

def cosine_similarity(a, b):    # 计算余弦相似度
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
def protos2featureVector(protos):    # 将存放proto的defaultlist转化为numpy数组方便分簇
    feature_vector = torch.cat(protos)
    return feature_vector.cpu().numpy()   # 转化为numpy后返回
def vecPadding(protos): # 填充protos缺失的label
    tensor_shape = None
    device = None
    for label in protos.keys():
        tensor_shape = protos[label].shape
        device = protos[label].device
        break   # 获取了tensor的shape和device
    for label in range(10): # 补全没有标签的label
        if label not in protos:
            protos[label] = torch.zeros(tensor_shape, device=device)   # 通过上面获取的tensorshape来补全没有标签的原型
    # sorted_protos = sorted(protos.keys())   # 对proto进行排序，避免错乱
    sorted_protos = []
    for i in range(10):
        sorted_protos.append(protos[i])
    return sorted_protos    # 返回的直接就是list


def createClusters(ends, n_clusters):     # 需要先训练一下得出一个proto才能分簇
    protos_for_clustering = []               # 用来存放【用来分簇的protos】
    for end in ends:
        print("Client training!")
        protos_tmp = end.trainForClustering()    # 训练得到的protos，是很多个proto，不同标签的，是个列表
        # protos_tmp_padded = vecPadding(protos_tmp)
        # tensors_to_cat = list(protos_tmp_padded.values())  # 先把defaultdist转成普通list，然后才能torch.cat()
        tensors_to_cat = vecPadding(protos_tmp) # 这里直接进行了padding和转换为list
        protos_numpy_single = protos2featureVector(tensors_to_cat)    # 单个客户端的protos转化为numpy数组
        protos_for_clustering.append(protos_numpy_single) # 将用于分簇的单个proto填入

    protos_for_clustering_normalized = normalize(protos_for_clustering) # 首先需要正则化
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(protos_for_clustering_normalized) # clusters数据结构是array，按顺序存放不同的cluster索引

    # # test-查看不同簇数目的影响
    # K_range = range(2, 11)
    # sse = []
    # for k in K_range:
    #     kmeans = KMeans(n_clusters=k, random_state=0)
    #     kmeans.fit(protos_for_clustering_normalized)
    #     sse.append(kmeans.inertia_)
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(K_range, sse, marker='o')
    # plt.title('Elbow Method for K Selection', fontsize=16)
    # plt.xlabel('Number of Clusters (K)', fontsize=14)
    # plt.ylabel('Sum of Squared Errors (SSE)', fontsize=14)
    # plt.grid()
    # plt.show()

    for i, cluster in enumerate(clusters):
        print(f"客户端 {i+1} 属于类别 {cluster}")

    return clusters


