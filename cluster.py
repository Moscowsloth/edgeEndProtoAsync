from collections import defaultdict

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

def cosine_similarity(a, b):    # 计算余弦相似度
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
def protos2featureVector(protos):    # 将存放proto的defaultlist转化为numpy数组方便分簇
    feature_vector = torch.cat(protos)
    return feature_vector.numpy()   # 转化为numpy后返回

def createClusters(ends, protos, n_clusters):     # 需要先训练一下得出一个proto才能分簇
    protos_for_clustering = []               # 用来存放【用来分簇的protos】
    for end in ends:
        protos_tmp = end.trainForClustering()    # 训练得到的protos，是很多个proto，不同标签的，是个列表
        protos_numpy_single = protos2featureVector(protos_tmp)    # 单个客户端的protos转化为numpy数组
        protos_for_clustering.append(protos_numpy_single) # 将用于分簇的单个proto填入

    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(protos_for_clustering)

    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1}: {cluster}")


