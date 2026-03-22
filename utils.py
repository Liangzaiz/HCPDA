import dgl
import gc
import networkx as nx
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import auc as auc3
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score, precision_recall_curve
from CLaugment import *

default_configure = {
    'batch_size': 20
}
heter_configure = {
    "lr": 0.0001,
    "dropout": 0,
    "cl_loss_co": 0.5,
    "reg_co": 0.0003,
    "in_size": 512,
    "hidden_size": 256,
    "out_size": 128,
    "weight_decay": 1e-10

}
Es_configure = {
    "lr": 0.0001,
    "dropout": 0,
    "cl_loss_co": 0.5,
    "reg_co": 0.0003,
    "in_size": 512,
    "hidden_size": 256,
    "out_size": 128,
    "weight_decay": 1e-10

}


def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def setup(args, seed):
    args.update(default_configure)
    set_random_seed(seed)
    return args


def comp_jaccard(M):
    matV = np.mat(M)
    x = dist.pdist(matV, 'jaccard')

    k = np.eye(matV.shape[0])
    count = 0
    for i in range(k.shape[0]):
        for j in range(i + 1, k.shape[1]):
            k[i][j] = x[count]
            k[j][i] = x[count]
            count += 1
    return k


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


def load_homo(dataset, save_path):
    network_path = f"./data/{dataset}/"
    # network_path = "D:\\wyl\\programs\\python\\CL-PDA\\data\\pirpheno2.0(102-462)\\"
    # network_path = "D:\\wyl\\programs\\python\\CL-PDA\\data\\piRDisease(21-4350)\\"
    # network_path = "D:\\wyl\\programs\\python\\CL-PDA\\data\\MNDR4.0(15-8178)\\"

    # drug_protein -> pirna_disease 形状：(8178, 15)
    pirna_disease = np.genfromtxt(network_path + 'adj1.csv', delimiter=',')  # p-d

    # protein_drug -> disease_pirna 形状：(15, 8178)
    disease_pirna = pirna_disease.T

    # drug_drug -> pirna_pirna 形状：(8178, 8178)
    pirna_pirna = np.genfromtxt(network_path + "p2p_smith.csv", delimiter=',')  # d2d_do.csv  p2p_needleman.csv
    # pirna_pirna = np.genfromtxt(network_path + "p_p_f.csv", delimiter=',')  # d2d_do.csv  p2p_needleman.csv

    # protein_protein -> disease_disease 形状：(15, 15)
    disease_disease = np.genfromtxt(network_path + "d2d_do.csv", delimiter=',')
    # disease_disease = np.genfromtxt(network_path + "d_d_f.csv", delimiter=',')

    # dti_o -> pda_o 形状：(8178, 15)
    pda_o = np.genfromtxt(network_path + 'adj1.csv', delimiter=',')

    # 将边转换为无向边
    pirna_pirna_edges = sparse.coo_matrix(pirna_pirna).nonzero()  # 起点为RNA，终点为RNA的边
    disease_disease_edges = sparse.coo_matrix(disease_disease).nonzero()  # 起点为disease，终点为disease的边
    pirna_disease_edges = sparse.coo_matrix(pirna_disease).nonzero()  # 起点为RNA，终点为disease的边
    disease_pirna_edges = sparse.coo_matrix(disease_pirna).nonzero()  # 起点为disease，终点为RNA的边

    # 创建异构图的边列表，添加反向边
    # graph_data 是一个字典，
    # 键是 边类型三元组 (源节点类型, 边类型, 目标节点类型)，
    # 值是对应的 边索引元组 (源节点索引数组, 目标节点索引数组)
    graph_data = {('pirna', 'similarity', 'pirna'): (pirna_pirna_edges[0], pirna_pirna_edges[1]),
                  ('disease', 'similarity', 'disease'): (disease_disease_edges[0], disease_disease_edges[1]),
                  ('pirna', 'pd', 'disease'): (pirna_disease_edges[0], pirna_disease_edges[1]),
                  ('disease', 'dp', 'pirna'): (disease_pirna_edges[0], disease_pirna_edges[1]),
                  # 添加无向边：即将 pirna_disease_edges 的反向边添加到图中
                  ('disease', 'pd', 'pirna'): (pirna_disease_edges[1], pirna_disease_edges[0])}

    # 创建异构图
    graph = dgl.heterograph(graph_data)  # 完整的异构图

    # 获取节点数
    num_pirna = graph.num_nodes('pirna')
    num_disease = graph.num_nodes('disease')
    # print("num_pirna ", num_pirna)
    # print("num_disease ", num_disease)
    # print("Pirna-Pirna shape:", pirna_pirna.shape)
    # print("Disease-Disease shape:", disease_disease.shape)
    # print("Pirna-Disease shape:", pirna_disease.shape)
    unique_pirnas = np.unique(pirna_disease)  # Unique pirnas: [0. 1.] 只有0和1两种值
    # print("Unique pirnas:", len(unique_pirnas))

    # 明确指定子图中的边类型
    # piRNA为视角的子图
    pg = graph.edge_type_subgraph(
        [('pirna', 'similarity', 'pirna'), ('pirna', 'pd', 'disease'), ('disease', 'dp', 'pirna')])
    # disease为视角的子图
    dg = graph.edge_type_subgraph(
        [('disease', 'similarity', 'disease'), ('disease', 'dp', 'pirna'), ('pirna', 'pd', 'disease')])

    # 保存为 graph 列表
    graph = [pg, dg]  # 异构图存储为两个不同视角的子图

    whole_positive_index = []
    whole_negative_index = []

    # 假设pda_o是一个二维数组，其中的1代表正样本，0代表负样本
    for i in range(np.shape(pda_o)[0]):
        for j in range(np.shape(pda_o)[1]):
            if int(pda_o[i][j]) == 1:
                whole_positive_index.append([i, j])
            elif int(pda_o[i][j]) == 0:
                whole_negative_index.append([i, j])

    # 计算正负样本数量
    positive_sample_size = len(whole_positive_index)
    # print(positive_sample_size)
    negative_sample_size = positive_sample_size * 1  # 负样本数量是正样本的5倍

    # 随机打乱正样本
    positive_shuffle_index = np.random.choice(np.arange(positive_sample_size),
                                              size=positive_sample_size, replace=False)
    whole_positive_index = np.array(whole_positive_index)
    whole_positive_index = whole_positive_index[positive_shuffle_index]  # 全部正样本的索引
    # print(whole_positive_index)

    # 随机抽取负样本
    negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                             size=negative_sample_size, replace=False)

    # 创建数据集
    data_set = np.zeros((positive_sample_size + negative_sample_size, 3), dtype=int)
    # print(data_set)

    # 填充正样本
    count = 0
    for ind, i in enumerate(whole_positive_index):
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1  # 正样本标签为1
        count += 1

    # 生成pda_cledge.txt文件
    with open(save_path + "pda_cledge.txt", "w", encoding="utf-8") as f:
        for i in range(count):
            for j in range(count):
                if data_set[i][0] == data_set[j][0] or data_set[i][1] == data_set[j][1]:
                    f.write(f"{i}\t{j}\n")

    # 填充负样本
    for ind, i in enumerate(negative_sample_index):
        data_set[count][0] = whole_negative_index[i][0]
        data_set[count][1] = whole_negative_index[i][1]
        data_set[count][2] = 0  # 负样本标签为0
        count += 1

    # 生成pda_index.txt文件
    with open(save_path + "pda_index.txt", "w", encoding="utf-8") as f:
        for i in data_set:
            f.write(f"{i[0]}\t{i[1]}\t{i[2]}\n")

    dateset = data_set  # 全部正负样本
    # print(data_set)
    # pdaedge.txt文件
    f = open(save_path + "pdaedge.txt", "w", encoding="utf-8")
    for i in range(dateset.shape[0]):
        for j in range(i, dateset.shape[0]):

            # if dateset[i][0] == dateset[j][0] or dateset[i][1] == dateset[j][1]:
            if i == j:
                f.write(f"{i}\t{j}\n")
    f.close()
    node_num = [num_pirna, num_disease]

    # 元路径（Meta - path）的定义：
    # 分别以piRNA的视角[['similarity'], ['pd', 'dp']]和疾病的视角[['similarity'], ['dp', 'pd']]定义元路径（每个视角两种）
    # piRNA视角：
    # ['similarity']：
    # 路径：piRNA --similarity--> piRNA
    # 语义：直接利用 piRNA 之间的相似性关系。
    # ['pd', 'dp']：
    # 路径：piRNA --pd--> disease --dp--> piRNA
    # 语义：piRNA 通过关联的疾病间接连接到其他 piRNA。
    all_meta_paths = [[['similarity'], ['pd', 'dp']],
                      [['similarity'], ['dp', 'pd']]]
    # 返回值：
    # data_set 是piRNA-disease对儿的标签信息（1正样本，0负样本），是N(正负样本总数)x3形状的 ndarray
    # graph 是异构图，由两个异构图子图构成：graph = [pg, dg]，分别是piRNA为视角的异构图和disease为视角的异构图
    # node_num 是piRNA和disease节点数量，是含有2个元素的列表List
    # all_meta_paths 是元路径的定义
    return data_set, graph, node_num, all_meta_paths


def load_graph(feature_edges, n):
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sparse.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(n, n),
                             dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    nfadj = normalize(fadj + sparse.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)

    return nfadj


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def construct_fgraph(features, topk):
    ##### Kernel
    # dist = -0.5 * pair(features) ** 2
    # dist = np.exp(dist)

    #### Cosine
    dist = cos(features)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)
    edge = []
    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                edge.append([i, vv])
    return edge


def generate_knn(data):
    topk = 1

    edge = construct_fgraph(data, topk)
    res = []

    for line in edge:
        start, end = line[0], line[1]
        if int(start) < int(end):
            res.append([start, end])
    return res


# 生成 k-近邻图
def generate_knn(data, topk=1):  # 默认 topk 设置为 5
    edge = construct_fgraph(data, topk)
    res = []

    # 只保留一个方向的边，避免重复边
    for line in edge:
        start, end = line[0], line[1]
        if start < end:
            res.append([start, end])

    # 强制垃圾回收，释放内存
    gc.collect()

    return res


def visualize_graph(dateset, h1, h2, edge, feature):
    """
    可视化构建的图，节点根据特征显示，边表示节点之间的关系，
    并且区分不同类型的节点（h1 和 h2），显示所有节点，包括无关联的节点。
    """
    # print(" feature12", feature)
    # 如果 edge 是稀疏张量，合并并提取索引
    if isinstance(edge, torch.sparse.Tensor):
        edge = edge.coalesce().indices().T.cpu().detach().numpy()

    # 创建一个无向图对象
    G = nx.Graph()

    # 将边添加到图中，去除自环
    edge = [e for e in edge if e[0] != e[1]]
    G.add_edges_from(edge)

    # 计算节点的位置（使用 spring 布局）
    pos = nx.spring_layout(G, seed=42)

    # 为节点设置颜色（根据节点类型）
    node_colors = []
    feature_values = feature[:, :256].mean(dim=1).cpu().detach().numpy()  # 使用特征的均值

    # 为 h1 和 h2 分配不同的颜色，且根据特征值深浅调整颜色
    for i in range(dateset.shape[0]):
        if i < len(h1):  # h1 节点
            # 蓝色渐变：深蓝表示特征值大，浅蓝表示特征值小
            color = plt.cm.Blues(feature_values[i] / max(feature_values))
        else:  # h2 节点
            # 红色渐变：深红表示特征值大，浅红表示特征值小
            color = plt.cm.Reds(feature_values[i] / max(feature_values))
        node_colors.append(color)

    # 确保所有节点都绘制出来，甚至是无连接的孤立节点
    all_nodes = set(range(dateset.shape[0]))
    missing_nodes = all_nodes - set([e[0] for e in edge]) - set([e[1] for e in edge])
    G.add_nodes_from(missing_nodes)

    # 绘制节点和边
    plt.figure(figsize=(6, 4))
    nodes = nx.draw_networkx_nodes(G, pos, node_size=80, node_color=node_colors, alpha=0.8)
    edges = nx.draw_networkx_edges(G, pos, width=0.8, alpha=0.5, edge_color='gray')

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=min(feature_values), vmax=max(feature_values)))
    sm.set_array([])
    cbar = plt.colorbar(sm, label='piRNA(Color intensity represents node value)', orientation='horizontal', shrink=0.1,
                        pad=0.0001, anchor=(0, 0))
    cbar.set_ticks([])
    sm2 = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(vmin=min(feature_values), vmax=max(feature_values)))
    sm2.set_array([])
    cbar2 = plt.colorbar(sm2, label='Disease(Color intensity represents node value)', orientation='horizontal',
                         shrink=0.1, pad=0.0001, anchor=(0, 0))
    cbar2.set_ticks([])

    # 调整图形边距和标题位置
    plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98)
    plt.title("Graph Visualization", fontsize=14, y=0.95)

    plt.axis('off')
    plt.show()


def constructure_graph(dateset, h1, h2, save_path, aug=False):
    # 用 PyTorch 和 NumPy 高效操作
    # print("dateset", dateset.shape)
    # print("h1", h1.shape)
    # print("h2", h2.shape)

    # 通过切片操作获取特征
    feature = torch.cat((h1[dateset[:, 0]], h2[dateset[:, 1]]), dim=1)
    # print("feature ", feature.shape)

    feature = feature.squeeze(1)
    # print(" feature", feature.shape)

    # 只读一次文件
    if not hasattr(constructure_graph, 'edge_cache'):  # 缓存文件内容
        edge = np.loadtxt(save_path + f"pdaedge.txt", dtype=int)
        constructure_graph.edge_cache = edge  # 将数据缓存起来
    else:
        edge = constructure_graph.edge_cache
    # print("edge shape", edge.shape)

    if aug:
        edge_aug = aug_random_edge(np.array(edge))
        edge_aug = load_graph(np.array(edge_aug), dateset.shape[0])
        edge = load_graph(np.array(edge), dateset.shape[0])

        feature_aug = aug_random_mask(feature)
        return edge, feature, edge_aug, feature_aug

    edge = load_graph(np.array(edge), dateset.shape[0])

    return edge, feature


def visualize_graph1(dateset, h1, h2, edge, feature):
    """
    可视化构建的图，节点根据特征显示，边表示节点之间的关系，
    并且区分不同类型的节点（h1 和 h2），显示所有节点，包括无关联的节点。
    """
    print(" feature12", feature)
    # 如果 edge 是稀疏张量，合并并提取索引
    if isinstance(edge, torch.sparse.Tensor):
        edge = edge.coalesce().indices().T.cpu().detach().numpy()

    # 创建一个无向图对象
    G = nx.Graph()

    # 将边添加到图中，去除自环
    edge = [e for e in edge if e[0] != e[1]]
    G.add_edges_from(edge)

    # 计算节点的位置（使用 spring 布局）
    pos = nx.spring_layout(G, seed=42)

    # 确保所有节点都有位置
    all_nodes = set(range(dateset.shape[0]))
    missing_nodes = all_nodes - set(pos.keys())

    # 给缺失的节点手动指定位置，通常可以放置在图的边缘位置
    for node in missing_nodes:
        pos[node] = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))  # 随机生成一个位置

    # 为节点设置颜色（根据节点类型）
    node_colors = []
    feature_values = feature[:, :256].mean(dim=1).cpu().detach().numpy()  # 使用特征的均值

    # 为 h1 和 h2 分配不同的颜色，且根据特征值深浅调整颜色
    for i in range(dateset.shape[0]):
        if i < len(h1):  # h1 节点
            # 蓝色渐变：深蓝表示特征值大，浅蓝表示特征值小
            color = plt.cm.Blues(feature_values[i] / max(feature_values))
        else:  # h2 节点
            # 红色渐变：深红表示特征值大，浅红表示特征值小
            color = plt.cm.Reds(feature_values[i] / max(feature_values))
        node_colors.append(color)

    # 确保所有节点都绘制出来，甚至是无连接的孤立节点
    missing_nodes = all_nodes - set([e[0] for e in edge]) - set([e[1] for e in edge])
    G.add_nodes_from(missing_nodes)

    # 绘制节点和边
    plt.figure(figsize=(8, 6))

    # 绘制节点
    nodes = nx.draw_networkx_nodes(
        G, pos, node_size=80, node_color=node_colors, alpha=1.0
    )

    # 调整子图和图例位置
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

    # 显示图形
    plt.title("Graph Visualization")
    plt.axis('off')
    plt.show()


def constructure_knngraph(dateset, h1, h2, aug=False):
    feature = torch.cat((h1[dateset[:, :1]], h2[dateset[:, 1:2]]), dim=2)

    feature = feature.squeeze(1)

    fedge = np.array(generate_knn(feature.cpu().detach().numpy()))

    if aug:
        fedge_aug = aug_random_edge(np.array(fedge))
        feature_aug = aug_random_mask(feature)
        fedge_aug = load_graph(np.array(fedge_aug), dateset.shape[0])
        fedge = load_graph(np.array(fedge), dateset.shape[0])

        return fedge, feature, fedge_aug, feature_aug
    else:
        fedge = load_graph(np.array(fedge), dateset.shape[0])
        # visualize_graph1(dateset, h1, h2, fedge, feature)

        return fedge, feature


def get_clGraph(data, task, save_path):
    cledge = np.loadtxt(save_path + f"{task}_cledge.txt", dtype=int)  # 加载边列表文件
    # cl: 形状为[n, n]的张量，n为正负样本总数，表示两个样本之间是否存在关联
    cl = torch.eye(len(data))  # 初始化单位矩阵（对角线为1，其余为0）
    for i in cledge:
        cl[i[0]][i[1]] = 1
    return cl


def get_set(data, split=5):
    """
    :param data: dataset and label
    :return:
    testset index and trainset index
    """
    set1 = []
    set2 = []
    skf = StratifiedKFold(n_splits=split, shuffle=True)
    for train_index, test_index in skf.split(data[:, :2], data[:, 2:3]):
        set1.append(train_index)
        set2.append(test_index)
    return set1[0].reshape(-1), set2[0].reshape(-1)


# 对数据集进行分层 K 折交叉验证（Stratified K-Fold Cross-Validation），生成训练集和测试集的索引
def get_cross(data, split=5):
    """
    :param data: dataset and label
    :param split: number of folds
    :return:
    trainset index and testset index
    """
    set1 = []
    set2 = []
    skf = StratifiedKFold(n_splits=split, shuffle=True)
    for train_index, test_index in skf.split(data[:, :2], data[:, 2:3]):  # 分别是前两列和第三列
        set1.append(train_index)
        set2.append(test_index)
    return set1, set2


def get_roc(out, label):
    return roc_auc_score(label.cpu(), out[:, 1:].cpu().detach().numpy())


def get_pr(out, label):
    precision, recall, thresholds = precision_recall_curve(label.cpu(), out[:, 1:].cpu().detach().numpy())
    return auc3(recall, precision)


def get_f1score(out, label):
    return f1_score(label.cpu(), out.argmax(dim=1).cpu().detach().numpy())


def get_precision(out, label):
    preds = out.argmax(dim=1).cpu().detach().numpy()
    return precision_score(label.cpu().detach().numpy(), preds, average='binary')


def get_recall(out, label):
    preds = out.argmax(dim=1).cpu().detach().numpy()
    return recall_score(label.cpu().detach().numpy(), preds, average='binary')


def get_L2reg(parameters):
    reg = 0
    for param in parameters:
        reg += 0.5 * (param ** 2).sum()
    return reg


def load_dataset(dataset, save_path):
    return load_homo(dataset, save_path)
