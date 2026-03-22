from utils import *
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
from GCNLayer import *
import time

# device = "cpu"
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# 初始化函数
def init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# TODO t
# class TransformerEncoder(nn.Module):
#     def __init__(self, in_size, num_heads=1, dropout=0.1):
#         super(TransformerEncoder, self).__init__()
#         # 创建多头注意力层：
#         # embed_dim: 输入特征维度
#         # num_heads: 注意力头数
#         self.attention = nn.MultiheadAttention(embed_dim=in_size, num_heads=num_heads, dropout=dropout)
#         # 两个LayerNorm层用于残差连接后的归一化
#         self.norm1 = nn.LayerNorm(in_size)
#         self.norm2 = nn.LayerNorm(in_size)
#         # 前馈网络：
#         #   两层线性变换 + ReLU激活
#         #   保持输入输出维度一致
#         self.fc = nn.Sequential(
#             nn.Linear(in_size, in_size),
#             nn.ReLU(),
#             nn.Linear(in_size, in_size)
#         )
#         # Dropout层用于前馈网络输出
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         # Apply Multihead Attention
#         # 自注意力计算(Q=K = V = x) (query, key, value)
#         # 忽略注意力权重(第二个返回值)
#         attn_output, _ = self.attention(x, x, x)
#         # 残差连接+层归一化
#         x = self.norm1(x + attn_output)  # Add & Norm
#
#         # Feed-forward layer
#         # 前馈网络处理
#         ff_output = self.fc(x)
#         # 残差连接+dropout+层归一化
#         return self.norm2(x + self.dropout(ff_output))  # Add & Norm


# Semantic Attention
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(SemanticAttention, self).__init__()
        # 语义注意力投影网络：
        #   第一层线性变换到hidden_size(默认32) + Xavier初始化
        #   Tanh激活函数
        #   第二层线性变换到1维(注意力分数) + Xavier初始化
        #   无偏置项
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size).apply(init),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False).apply(init)
        )

    def forward(self, z):
        # 计算所有节点的注意力分数
        # 在节点维度取平均(得到每个元路径的重要性分数)
        w = self.project(z).mean(0)
        # 对注意力分数做softmax归一化
        beta = torch.softmax(w, dim=0)
        # 扩展注意力权重以匹配输入维度
        beta = beta.expand((z.shape[0],) + beta.shape)
        # 加权求和得到最终表示
        return (beta * z).sum(1)


# FastGTN集成到图卷积层(节点集加语义集)
class FastGTNLayer(nn.Module):
    # layer_num_heads: 注意力头数
    # transformer_heads: Transformer注意力头数(默认2)
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout, transformer_heads=2):
        super(FastGTNLayer, self).__init__()
        self.gat_layers = nn.ModuleList()  # 初始化图注意力层列表
        # 添加一个基础图卷积层
        self.gat_layers.append(GraphConv(in_size, out_size, activation=F.relu, allow_zero_in_degree=True).apply(init))
        # 初始化语义注意力层：输入维度为out_size * layer_num_heads(多头注意力输出的拼接)
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)  # 将元路径转换为元组形式保存
        # 初始化图缓存：
        #   _cached_graph: 缓存原始图结构
        #   _cached_coalesced_graph: 缓存基于不同元路径的子图
        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self.dropout = dropout

        # 使用改进的Transformer编码器  TODO t
        # self.transformer = TransformerEncoder(
        #     out_size * layer_num_heads,
        #     num_heads=transformer_heads,
        #     dropout=dropout
        # )

    # 前向传播接收图结构g和节点特征h
    def forward(self, g, h):
        semantic_embeddings = []  # 初始化语义嵌入列表
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g  # 存储当前图
            self._cached_coalesced_graph.clear()
            # 为每个元路径生成可达子图并缓存
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(g, meta_path)
        # 结点级注意力机制
        #    遍历所有元路径
        #    从缓存获取对应元路径的子图
        #    应用图卷积层计算节点表示
        #    将结果展平后加入语义嵌入列表
        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]

            semantic_embeddings.append(self.gat_layers[0](new_g, h).flatten(1))

        # 堆叠所有元路径的嵌入结果：形成维度为(节点数, 元路径数, 特征维度)的张量
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)

        # 通过改进的Transformer增强语义表示
        #   用Transformer编码器增强语义表示
        #   捕捉不同元路径间的复杂关系
        # semantic_embeddings = self.transformer(semantic_embeddings)  # TODO t

        # 应用语义注意力机制：
        #   自动学习不同元路径的重要性权重
        #   生成最终的节点表示
        return self.semantic_attention(semantic_embeddings)


# Heterogeneous Graph Attention Network（HAN）
# HAN类使用FastGTN思想
class HAN(nn.Module):
    # num_heads: 注意力头数
    # transformer_heads: Transformer头数
    def __init__(self, meta_paths, in_size, hidden_size, out_size, dropout, num_heads=1, transformer_heads=1):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        # 创建一个线性预测层
        # 输入维度：hidden_size * num_heads(因为多头注意力会拼接各头的输出)
        # 输出维度：out_size
        self.predict = nn.Linear(hidden_size * num_heads, out_size, bias=False).apply(init)
        # 添加一个 FastGTNLayer 层
        self.layers.append(FastGTNLayer(meta_paths, in_size, hidden_size, num_heads, dropout, transformer_heads))

    # g: 图结构数据
    # h: 节点特征
    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)  # 更新特征 h 为当前层(FastGTNLayer)的输出
        # 最终的特征表示 h 通过线性预测层，返回预测结果
        return self.predict(h)


# HAN_DTI层结合FastGTN的思想
class HAN_DTI(nn.Module):
    def __init__(self, all_meta_paths, in_size, hidden_size, out_size, dropout, transformer_heads=1):
        super(HAN_DTI, self).__init__()
        self.sum_layers = nn.ModuleList()  # 创建一个 ModuleList 来存储网络层

        # 集成FastGTN思想的每一层
        for i in range(len(all_meta_paths)):  # len(all_meta_paths)=2
            self.sum_layers.append(
                # 为每种类型的节点(p和d)创建一个HAN网络
                # 每个HAN网络对应不同的元路径集合(all_meta_paths[i])
                HAN(all_meta_paths[i], in_size[i], hidden_size[i], out_size[i], dropout,
                    transformer_heads=transformer_heads)
            )

    # s_g: 图结构数据(包含两种节点类型的图)
    # s_h_1: 第一种节点类型(如piRNA)的特征
    # s_h_2: 第二种节点类型(如disease)的特征
    # 返回值h1, h2: 两种节点类型的处理结果
    def forward(self, s_g, s_h_1, s_h_2):
        h1 = self.sum_layers[0](s_g[0], s_h_1)
        h2 = self.sum_layers[1](s_g[1], s_h_2)
        return h1, h2

# Graph Transformer Networks(GTN)
# FastGTN集成的核心网络
class FastGTNCore(nn.Module):
    def __init__(self, all_meta_paths, in_size, hidden_size, out_size, dropout, num_heads=1, transformer_heads=1):
        super(FastGTNCore, self).__init__()
        self.han_dti = HAN_DTI(all_meta_paths, in_size, hidden_size, out_size, dropout, transformer_heads)

    def forward(self, graph, node_features_1, node_features_2):
        return self.han_dti(graph, node_features_1, node_features_2)


class GCN(nn.Module):
    # nfeat: 节点特征的输入维度
    def __init__(self, nfeat, dropout):
        super(GCN, self).__init__()
        # 第一层图卷积：
        #     输入维度nfeat
        #     输出维度256
        self.gc1 = GraphConvolution(nfeat, 256)
        # 第二层图卷积：
        #     输入维度256(与第一层输出匹配)
        #     输出维度128
        self.gc2 = GraphConvolution(256, 128)
        self.dropout = dropout

    # x: 节点特征矩阵(形状为[节点数, 特征维度])
    # adj: 邻接矩阵(稀疏矩阵表示)
    def forward(self, x, adj):
        # 确保数据和模型在相同设备(CPU / GPU)
        x = x.to(device)
        adj = adj.to(device)
        # 第一层图卷积计算
        #   应用ReLU激活函数：
        #   inplace = True节省内存
        x1 = F.relu(self.gc1(x, adj), inplace=True)
        # 应用dropout正则化：
        #   防止过拟合
        #   只在训练阶段生效
        x1 = F.dropout(x1, self.dropout)
        # 第二层图卷积计算：
        #   使用第一层的输出作为输入
        #   不使用激活函数(通常最后一层不激活)
        x2 = self.gc2(x1, adj)
        res = x2
        # 返回最终节点表示
        return res


class CL_GCN(nn.Module):
    # nfeat: 节点特征维度
    # alpha: 对比损失权重系数(默认0.8)
    def __init__(self, nfeat, dropout, alpha=0.8):
        super(CL_GCN, self).__init__()
        # 创建两个相同的GCN网络：
        #   用于处理两个视图(view)的图数据
        #   参数不共享
        self.gcn1 = GCN(nfeat, dropout)
        self.gcn2 = GCN(nfeat, dropout)
        self.tau = 0.5  # 设置温度参数tau(默认0.5)？
        self.alpha = alpha

    # x1, adj1: 视图1的节点特征和邻接矩阵
    # x2, adj2: 视图2的节点特征和邻接矩阵
    # clm: 对比学习掩码(contrastive learning mask)
    def forward(self, x1, adj1, x2, adj2, clm):
        # 分别通过两个GCN获取节点表示z1, z2
        z1 = self.gcn1(x1, adj1)
        z2 = self.gcn2(x2, adj2)

        # 计算对称对比损失：
        #   加权组合两个方向的相似度计算
        #   使用alpha平衡两个方向
        loss = self.alpha * self.sim(z1, z2, clm) + (1 - self.alpha) * self.sim(z2, z1, clm)

        # 返回两个视图的节点表示和对比损失
        return z1, z2, loss

    def sim(self, z1, z2, clm):
        # print("z1",z1.shape)
        # print("z2", z2.shape)
        # 计算节点表示的L2范数：保持维度便于后续矩阵运算
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        # 计算：
        #   分子：节点表示的点积矩阵
        #   分母：范数乘积矩阵(用于归一化)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        # 计算带温度参数的指数化相似度矩阵
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        # 行归一化(softmax)：添加小常数1e - 8防止除零
        sim_matrix = sim_matrix / (torch.sum(sim_matrix, dim=1).view(-1, 1) + 1e-8)
        # 计算对比损失：
        #   使用掩码clm筛选正样本对
        #   取对数负似然的均值
        sim_matrix = sim_matrix.to(device)
        # print("sim_matrix ",sim_matrix.shape)
        loss = -torch.log(sim_matrix.mul(clm).sum(dim=-1)).mean()

        return loss

    # 简单的均方误差计算：可作为额外的正则化项
    def mix2(self, z1, z2):
        loss = ((z1 - z2) ** 2).sum() / z1.shape[0]
        return loss


class MLP(nn.Module):
    def __init__(self, nfeat):
        super(MLP, self).__init__()
        # 创建顺序模型容器(Sequential)
        self.MLP = nn.Sequential(
            # 第一层全连接：
            #   输入维度nfeat
            #   输出维度32
            #   无偏置项(bias=False)
            #   应用Xavier初始化(init函数)
            nn.Linear(nfeat, 32, bias=False).apply(init),
            # ELU激活函数：
            #   指数线性单元
            #   相比ReLU能缓解神经元死亡问题
            nn.ELU(),
            # 第二层全连接：
            #   输入维度32(与上层输出匹配)
            #   输出维度2(二分类)
            #   无偏置项
            nn.Linear(32, 2, bias=False),
            # 对数Softmax：
            #   在维度1(特征维度)计算
            #   输出对数概率(配合NLLLoss使用)
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        output = self.MLP(x)
        return output


class HMTCL(nn.Module):
    # all_meta_paths: 所有元路径信息
    # in_size: 输入特征维度
    # hidden_size: 隐藏层维度
    # out_size: 输出维度
    # dropout: dropout率
    def __init__(self, all_meta_paths, in_size, hidden_size, out_size, dropout):
        super(HMTCL, self).__init__()
        # HAN_DTI: 处理异构图的注意力网络？
        self.HAN_DTI = HAN_DTI(all_meta_paths, in_size, hidden_size, out_size, dropout)
        # CL_GCN: 对比学习的图卷积网络
        self.CL_GCN = CL_GCN(256, dropout)
        self.MLP = MLP(256)

    # graph: 异构图
    # h: 全部节点特征矩阵
    # cl: 表示两个样本是否存在关联的矩阵
    # dateset_index: 样本数据的索引
    # data: 样本数据
    # iftrain: 标记是否是训练
    # p: piRNA特征矩阵
    # d: disease特征矩阵
    def forward(self, graph, h, cl, dateset_index, data, save_path, iftrain=True, p=None, d=None):
        start_time = time.time()
        if iftrain:
            # 通过 HAN_DTI 模块处理异构图，获取piRNA(p)和disease(d)的特征表示
            p, d = self.HAN_DTI(graph, h[0], h[1])
        end_time = time.time()
        # print(f'HAN_DTI execution time: {end_time - start_time:.4f} seconds')

        start_time = time.time()
        # 调用 constructure_graph 函数构建图结构
        # 返回边信息 edge 和节点特征 feature
        edge, feature = constructure_graph(data, p, d, save_path)
        end_time = time.time()
        # print(f'constructure_graph execution time: {end_time - start_time:.4f} seconds')

        start_time = time.time()
        f_edge, f_feature = constructure_knngraph(data, p, d)
        end_time = time.time()
        # print(f'constructure_knngraph execution time: {end_time - start_time:.4f} seconds')

        start_time = time.time()
        feature1, feature2, cl_loss1 = self.CL_GCN(feature, edge, f_feature, f_edge, cl)
        end_time = time.time()
        # print(f'CL_GCN execution time: {end_time - start_time:.4f} seconds')

        start_time = time.time()
        pred1 = self.MLP(torch.cat((feature1, feature2), dim=1)[dateset_index])
        end_time = time.time()
        # print(f'MLP execution time: {end_time - start_time:.4f} seconds')

        if iftrain:
            return pred1, cl_loss1, p, d
        return pred1
