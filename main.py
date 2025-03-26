import argparse, time, copy
import multiprocessing

import torch, torchvision
import numpy as np
import torch.nn as nn

from cluster import createClusters
from options import args_parser
from allModel.model import FedAvgCNN, BaseHeadSplit, Net
from allModel.resnet import resnet8, resnet10, resnet18, resnet34, resnet50

import random

from allEdges.edgeProAgg import EdgeProAgg
from allEnds.endProAgg import EndProAgg
from allEdges.edgeHierFL import EdgeHierFL
from allEnds.endHierFL import EndHierFL
from allEdges.edgeGKT import EdgeGKT
from allEnds.endGKT import EndGKT
from allEdges.edgeAgg import EdgeAgg
from allEnds.endAgg import EndAgg
from allEdges.edgeProAggAsync import EdgeProAggAsync
from allEnds.endProAggAsync import EndProAggAsync
from allEdges.edgeHierFLAsync import EdgeHierFLAsync
from allEnds.endHierFLAsync import EndHierFLAsync
from allEdges.edgeAsyncFL_AFO import EdgeAsyncFLAFO
from allEnds.endAsyncFL_AFO import EndAsyncFLAFO

from allEdges.edgeProAggAsyncSimple import EdgeProAggAsyncSimple
from allEnds.endProAggAsyncSimple import EndProAggAsyncSimple

from multiprocessing import Pool, Manager
import sh


def run(args):
    # 记录运行时间
    time_list = []
    
    end_model_str = args.end_model
    edge_model_str = args.edge_model

    edgeepoch = 6

    # 端侧模型赋值
    if end_model_str == "cnn":
        if "MNIST" in args.dataset:
            # args.end_model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            args.end_model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1024).to(args.device)
        # elif args.algorithm == "FedAgg":
        #     args.end_model = Net()
        else:    
            args.end_model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            
    elif end_model_str == "resnet8":
        args.end_model = resnet8(num_classes=args.num_classes).to(args.device)
    # 边缘模型赋值
    if edge_model_str == "cnn":
        args.edge_model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
    elif edge_model_str == "resnet10":
        args.edge_model = resnet10(num_classes=args.num_classes).to(args.device)
    elif edge_model_str == "resnet8":
        args.edge_model = resnet8(num_classes=args.num_classes).to(args.device)
    elif edge_model_str == "resnet50":
        args.edge_model = resnet50(num_classes=args.num_classes).to(args.device)
    elif edge_model_str == "resnet18":
        args.edge_model = resnet18(num_classes=args.num_classes).to(args.device)
    elif edge_model_str == "resnet34":
        args.edge_model = resnet34(num_classes=args.num_classes).to(args.device)            
        # args.edge_model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)
    
    backup = copy.deepcopy(args.end_model)
    # ends存储所有的端，i是客户端
    ends = []
    # 初始化并存储客户端
    for i in range(args.num_ends):
        if args.algorithm == "FedProAgg":
            ends.append(EndProAgg(i, args, copy.deepcopy(args.end_model), 0))
        elif args.algorithm == "HierFL":
            args.head = copy.deepcopy(args.end_model.fc)
            args.end_model.fc = nn.Identity()
            args.end_model = BaseHeadSplit(args.end_model, args.head)
            ends.append(EndHierFL(i, args, copy.deepcopy(args.end_model)))
            args.end_model = copy.deepcopy(backup)
        elif args.algorithm == "FedGKT":
            ends.append(EndGKT(i, args, copy.deepcopy(args.end_model)))
        elif args.algorithm == "FedAgg":
            ends.append(EndAgg(i, args, copy.deepcopy(args.end_model)))
        elif args.algorithm == "AsyncProAgg":
            ends.append(EndProAggAsync(i, args, copy.deepcopy(args.end_model), random.randint(0, 4)))
            # 最后一个参数是delay，可设置。
        elif args.algorithm == "AsyncProSimple":
            ends.append(EndProAggAsyncSimple(i, args, copy.deepcopy(args.end_model), random.randint(0, 4)))
            # 最后给出了delay的接口
            # todo delay应通过数据量决定
        elif args.algorithm == "HierFLAsync":
            ends.append(EndHierFLAsync(i, args, copy.deepcopy(args.end_model), random.randint(0, 4)))
        elif args.algorithm == "AFO":
            ends.append(EndAsyncFLAFO(i, args, copy.deepcopy(args.end_model), random.randint(0, 4)))
        else:
            raise ValueError(f"Algorithm {args.algorithm} not implemented.")

    if args.cluster:    # 测试，看看分簇能不能跑起来
        createClusters(ends, args.num_edges)
        return

    # edges初始化所有边缘，e是边缘
    edges = []
    dids = np.arange(args.num_ends)
    clients_per_edge = int(args.num_ends/args.num_edges)
    edge_dids = [dids[i:i+clients_per_edge] for i in range(0,len(dids),clients_per_edge)]

    if args.algorithm == "FedProAgg":
        # 设定边缘和端的所属关系
        for i in range(args.num_edges):
            edges.append(EdgeProAgg(i, edge_dids[i], args, copy.deepcopy(args.edge_model)))
            # 把端加入到每个边的self_ends里，指定端对应的边
            for j in edges[i].dids:
                edges[i].end_register(ends[j])
                ends[j].parent = edges[i]
        # 设定边缘的邻居关系
        for i in range(args.num_edges):
            edges[i].neigh_registration = edges[0:i] + edges[i+1:]      
        # 设定聚合边缘
        edges[len(edges)-1].aggregated_edge = True
        file = open(args.dataset + "_" + args.algorithm + "_results.txt", "w")
        # 通信num_comm次
        for i in range(args.num_comm):
            # 每个边缘下的所有端做训练
            print(f"\n-------------Round number: {i}-------------")
            for edge in edges:
                edge.train(i, file)
        file.close()
        
    elif args.algorithm == "HierFL":
        for i in range(args.num_edges):
            edges.append(EdgeHierFL(i, edge_dids[i], args, copy.deepcopy(args.edge_model)))
            # 把端加入到每个边的self_ends里，指定端对应的边
            for j in edges[i].dids:
                edges[i].end_register(ends[j])
                ends[j].parent = edges[i]
                edges[i].number_samples_ends += ends[j].train_samples
            # print("edge %d has \n" % edges[i].number_samples_ends)
        # 设定边缘的邻居关系
        for i in range(args.num_edges):
            edges[i].neigh_registration = edges[0:i] + edges[i+1:]
        # 设定聚合边缘
        edges[len(edges)-1].aggregated_edge = True
        file = open(args.dataset + "_" + args.algorithm + "_results.txt", "w")

        # 通信num_comm次
        for i in range(args.num_comm):
            # 每个边缘做训练
            print(f"\n-------------Round number: {i}-------------")
            for edge in edges:
                edge.train(i, file)
        file.close()

    elif args.algorithm == "HierFLAsync":
        for i in range(args.num_edges):
            edges.append(EdgeHierFLAsync(i, edge_dids[i], args, copy.deepcopy(args.edge_model), 6, 2))
            # 把端加入到每个边的self_ends里，指定端对应的边
            for j in edges[i].dids:
                edges[i].end_register(ends[j])
                ends[j].parent = edges[i]
                edges[i].number_samples_ends += ends[j].train_samples
            # print("edge %d has \n" % edges[i].number_samples_ends)
        # 设定边缘的邻居关系
        for i in range(args.num_edges):
            edges[i].neigh_registration = edges[0:i] + edges[i + 1:]
        # 设定聚合边缘
        edges[len(edges) - 1].aggregated_edge = True
        file = open(args.dataset + "_" + args.algorithm + "_results.txt", "w")

        # 通信num_comm次
        for i in range(args.num_comm):
            # 每个边缘做训练
            print(f"\n-------------Round number: {i}-------------")
            for edge in edges:
                edge.train(i, file)
        file.close()
    elif args.algorithm == "AFO":
        for i in range(args.num_edges):
            edges.append(EdgeAsyncFLAFO(i, edge_dids[i], args, copy.deepcopy(args.edge_model), 6, 2))
            # 把端加入到每个边的self_ends里，指定端对应的边
            for j in edges[i].dids:
                edges[i].end_register(ends[j])
                ends[j].parent = edges[i]
                edges[i].number_samples_ends += ends[j].train_samples
        for i in range(args.num_edges):
            edges[i].neigh_registration = edges[0:i] + edges[i + 1:]
            # 设定聚合边缘
        edges[len(edges) - 1].aggregated_edge = True
        file = open(args.dataset + "_" + args.algorithm + "_results.txt", "w")

        # 通信num_comm次
        for i in range(args.num_comm):
                # 每个边缘做训练
            print(f"\n-------------Round number: {i}-------------")
            for edge in edges:
                edge.train(i, file)
        file.close()

    # 设置edge_model为同构和异构两个场景
    elif args.algorithm == "FedGKT":
        for i in range(args.num_edges):
            edges.append(EdgeGKT(i, edge_dids[i], args, copy.deepcopy(args.edge_model)))
            # 把端加入到每个边的self_ends里，指定端对应的边
            for j in edges[i].dids:
                edges[i].end_register(ends[j])
                ends[j].parent = edges[i]
                edges[i].number_samples_ends += ends[j].train_samples
        # 设定边缘的邻居关系
        for i in range(args.num_edges):
            edges[i].neigh_registration = edges[0:i] + edges[i+1:]
        # 设定聚合边缘
        edges[len(edges)-1].aggregated_edge = True
        file = open(args.dataset + "_" + args.algorithm + "_results.txt", "w")

        # 通信num_comm次
        for i in range(args.num_comm):
            # 每个边缘做训练
            print(f"\n-------------Round number: {i}-------------")
            for edge in edges:
                edge.train(i, file)
        file.close()

    elif args.algorithm == "FedAgg":
        for i in range(args.num_edges):
            edges.append(EdgeAgg(i, edge_dids[i], args, copy.deepcopy(args.edge_model)))
            # 把端加入到每个边的self_ends里，指定端对应的边
            for j in edges[i].dids:
                edges[i].end_register(ends[j])
                ends[j].parent = edges[i]
                edges[i].number_samples_ends += ends[j].train_samples
                ends[j].get_noises_labels()
        # 设定边缘的邻居关系
        for i in range(args.num_edges):
            edges[i].neigh_registration = edges[0:i] + edges[i+1:]
        # 设定聚合边缘
        edges[len(edges)-1].aggregated_edge = True
        file = open(args.dataset + "_" + args.algorithm + "_results.txt", "w")

        # 通信num_comm次
        for i in range(args.num_comm):
            # 每个边缘做训练
            print(f"\n-------------Round number: {i}-------------")
            for edge in edges:
                edge.train(i, file, args)
        file.close()
    elif args.algorithm == "AsyncProSimple":
        for i in range(args.num_edges):
            edges.append(EdgeProAggAsyncSimple(i, edge_dids[i], args, copy.deepcopy(args.edge_model), 12, 2))
            # todo: 2可以改为别的周期
            # 把端加入到每个边的self_ends里，指定端对应的边
            for j in edges[i].dids:
                edges[i].end_register(ends[j])
                ends[j].parent = edges[i]
        # 设定边缘的邻居关系
        for i in range(args.num_edges):
            edges[i].neigh_registration = edges[0:i] + edges[i+1:]
        # 设定聚合边缘
        edges[len(edges)-1].aggregated_edge = True
        file = open(args.dataset + "_" + args.algorithm + "_results.txt", "w")

        # 训练总时间
        training_time_total = 0
        # 通信num_comm次
        for i in range(args.num_comm):
            # 每个边缘做训练
            print(f"\n-------------Round number: {i}-------------")
            for edge in edges:
                edge.asyncTrain(i, file, training_time_total)
                # todo 移除了args，添加了训练时间，不知道这样行不行
            print("Total training time is: " + str(training_time_total) + "!!!")

        file.close()

    elif args.algorithm == "AsyncProAgg":
        for i in range(args.num_edges):
            edges.append(EdgeProAggAsync(i, edge_dids[i], args, copy.deepcopy(args.edge_model), edgeepoch))
            # 把端加入到每个边的self_ends里，指定端对应的边
            for j in edges[i].dids:
                edges[i].end_register(ends[j])
                ends[j].parent = edges[i]
                edges[i].ends_state[j] = -1
            print("Client state test from main function:")
            print(edges[i].ends_state)
        # 设定边缘的邻居关系
        for i in range(args.num_edges):
            edges[i].neigh_registration = edges[0:i] + edges[i+1:]
        # 设定聚合边缘
        edges[len(edges)-1].aggregated_edge = True
        file = open(args.dataset + "_" + args.algorithm + "_results.txt", "w")
        # 通信num_comm次

        for i in range(args.num_comm):
            # 每个边缘下的所有端做训练
            print(f"\n-------------Round number: {i}-------------")
            for edge in edges:
                edge.asyn_train(i, file)   # 使用公用的线程池
        file.close()

    with open('warm_up_time.txt', 'w', encoding='utf-8') as file:
        for end in ends:
            print("Training time of end " + str(end.index) + ": " + str(end.delay_comp_total))
            text = "Training time of end " + str(end.index) + ": " + str(end.delay_comp_total)
            file.write(text)

    del args.end_model, args.edge_model, dids
            
    # i是运行几次
    for i in range(args.num_runtime):
        start_time = time.time()
        time_list.append(time.time()-start_time)
    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = args_parser(parser)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    time_start = time.time()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    # multiprocessing.set_start_method('forkserver')

    print("="*50)
    print("Data directory: {}".format(args.data_dir))
    print("Dataset: {}".format(args.dataset))
    print("End model: {}".format(args.end_model))
    print("Edge model: {}".format(args.edge_model))
    print("Batch size: {}".format(args.batch_size))
    print("Number of communication: {}".format(args.num_comm))
    print("Number of local training: {}".format(args.num_local_training))
    print("Learning rate: {}".format(args.lr))
    print("Device: {}".format(args.device))
    print("Algorithm: {}".format(args.algorithm))
    print("Number of running: {}".format(args.num_runtime))
    print("Number of classes: {}".format(args.num_classes))
    print("Number of evaluation term: {}".format(args.eval_term))

    print("Fraction: {}".format(args.frac))
    print("Number of ends: {}".format(args.num_ends))
    print("Number of edges: {}".format(args.num_edges))
    print("Seed: {}".format(args.seed))
    print("T_agg: {}".format(args.T_agg))

    print("Clustering: {}".format(args.cluster))

    print("="*50)

    run(args)