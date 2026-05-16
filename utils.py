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


def load_dataset(dataset, save_path):
    network_path = f"./data/{dataset}/"

    pirna_disease = np.genfromtxt(network_path + 'adj1.csv', delimiter=',')  # p-d

    disease_pirna = pirna_disease.T

    pirna_pirna = np.genfromtxt(network_path + "p2p_smith.csv", delimiter=',')

    disease_disease = np.genfromtxt(network_path + "d2d_do.csv", delimiter=',')

    pda_o = np.genfromtxt(network_path + 'adj1.csv', delimiter=',')

    pirna_pirna_edges = sparse.coo_matrix(pirna_pirna).nonzero()
    disease_disease_edges = sparse.coo_matrix(disease_disease).nonzero()
    pirna_disease_edges = sparse.coo_matrix(pirna_disease).nonzero()
    disease_pirna_edges = sparse.coo_matrix(disease_pirna).nonzero()

    graph_data = {('pirna', 'similarity', 'pirna'): (pirna_pirna_edges[0], pirna_pirna_edges[1]),
                  ('disease', 'similarity', 'disease'): (disease_disease_edges[0], disease_disease_edges[1]),
                  ('pirna', 'pd', 'disease'): (pirna_disease_edges[0], pirna_disease_edges[1]),
                  ('disease', 'dp', 'pirna'): (disease_pirna_edges[0], disease_pirna_edges[1]),
                  ('disease', 'pd', 'pirna'): (pirna_disease_edges[1], pirna_disease_edges[0])}

    graph = dgl.heterograph(graph_data)

    num_pirna = graph.num_nodes('pirna')
    num_disease = graph.num_nodes('disease')

    unique_pirnas = np.unique(pirna_disease)

    pg = graph.edge_type_subgraph(
        [('pirna', 'similarity', 'pirna'), ('pirna', 'pd', 'disease'), ('disease', 'dp', 'pirna')])  # (节点类型, 边类型, 节点类型)
    dg = graph.edge_type_subgraph(
        [('disease', 'similarity', 'disease'), ('disease', 'dp', 'pirna'), ('pirna', 'pd', 'disease')])

    graph = [pg, dg]

    whole_positive_index = []
    whole_negative_index = []

    for i in range(np.shape(pda_o)[0]):
        for j in range(np.shape(pda_o)[1]):
            if int(pda_o[i][j]) == 1:
                whole_positive_index.append([i, j])
            elif int(pda_o[i][j]) == 0:
                whole_negative_index.append([i, j])

    positive_sample_size = len(whole_positive_index)
    negative_sample_size = positive_sample_size * 1
    positive_shuffle_index = np.random.choice(np.arange(positive_sample_size),
                                              size=positive_sample_size, replace=False)
    whole_positive_index = np.array(whole_positive_index)
    whole_positive_index = whole_positive_index[positive_shuffle_index]  # 全部正样本的索引

    negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                             size=negative_sample_size, replace=False)

    data_set = np.zeros((positive_sample_size + negative_sample_size, 3), dtype=int)

    count = 0
    for ind, i in enumerate(whole_positive_index):
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1

    with open(save_path + "pda_cledge.txt", "w", encoding="utf-8") as f:
        for i in range(count):
            for j in range(count):
                if data_set[i][0] == data_set[j][0] or data_set[i][1] == data_set[j][1]:
                    f.write(f"{i}\t{j}\n")

    for ind, i in enumerate(negative_sample_index):
        data_set[count][0] = whole_negative_index[i][0]
        data_set[count][1] = whole_negative_index[i][1]
        data_set[count][2] = 0
        count += 1

    with open(save_path + "pda_index.txt", "w", encoding="utf-8") as f:
        for i in data_set:
            f.write(f"{i[0]}\t{i[1]}\t{i[2]}\n")

    dateset = data_set
    f = open(save_path + "pdaedge.txt", "w", encoding="utf-8")
    for i in range(dateset.shape[0]):
        for j in range(i, dateset.shape[0]):

            if dateset[i][0] == dateset[j][0] or dateset[i][1] == dateset[j][1]:
                f.write(f"{i}\t{j}\n")
    f.close()
    node_num = [num_pirna, num_disease]

    all_meta_paths = [[['similarity'], ['pd', 'dp']],
                      [['similarity'], ['dp', 'pd']]]

    return data_set, graph, node_num, all_meta_paths


def constructure_graph(dateset, h1, h2, save_path, aug=False):
    feature = torch.cat((h1[dateset[:, 0]], h2[dateset[:, 1]]), dim=1)

    feature = feature.squeeze(1)

    if not hasattr(constructure_graph, 'edge_cache'):  # 缓存文件内容
        edge = np.loadtxt(save_path + f"pdaedge.txt", dtype=int)
        constructure_graph.edge_cache = edge  # 将数据缓存起来
    else:
        edge = constructure_graph.edge_cache

    if aug:
        edge_aug = aug_random_edge(np.array(edge))
        edge_aug = load_graph(np.array(edge_aug), dateset.shape[0])
        edge = load_graph(np.array(edge), dateset.shape[0])

        feature_aug = aug_random_mask(feature)
        return edge, feature, edge_aug, feature_aug

    edge = load_graph(np.array(edge), dateset.shape[0])

    return edge, feature


def constructure_knngraph(dateset, h1, h2, topk=2, aug=False):
    feature = torch.cat((h1[dateset[:, :1]], h2[dateset[:, 1:2]]), dim=2)

    feature = feature.squeeze(1)

    fedge = np.array(generate_knn(feature.cpu().detach().numpy(), topk))

    if aug:
        fedge_aug = aug_random_edge(np.array(fedge))
        feature_aug = aug_random_mask(feature)
        fedge_aug = load_graph(np.array(fedge_aug), dateset.shape[0])
        fedge = load_graph(np.array(fedge), dateset.shape[0])

        return fedge, feature, fedge_aug, feature_aug
    else:
        fedge = load_graph(np.array(fedge), dateset.shape[0])

        return fedge, feature


def get_clGraph(data, task, save_path):
    cledge = np.loadtxt(save_path + f"{task}_cledge.txt", dtype=int)
    cl = torch.eye(len(data))
    for i in cledge:
        cl[i[0]][i[1]] = 1
    return cl


def generate_knn(data, topk):
    edge = construct_fgraph(data, topk)
    res = []
    for line in edge:
        start, end = line[0], line[1]
        if int(start) < int(end):
            res.append([start, end])
    return res


def construct_fgraph(features, topk):
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


def set_random_seed(seed=0):
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
    for train_index, test_index in skf.split(data[:, :2], data[:, 2:3]):  # 分别是前两列(data[:, :2])和第三列(data[:, 2:3])
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
