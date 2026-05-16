from utils import *
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
from GCNLayer import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def compute_laplacian_pe(g, k=8):
    device = g.device
    n = g.num_nodes()

    adj = g.adj().to_dense().to(device)

    degree = adj.sum(1)
    degree[degree == 0] = 1.0

    D_inv_sqrt = torch.diag(degree.pow(-0.5))
    L = torch.eye(n, device=device) - D_inv_sqrt @ adj @ D_inv_sqrt

    eigvals, eigvecs = torch.linalg.eigh(L)

    pe = eigvecs[:, 1:k + 1]

    return pe


class TransformerEncoder(nn.Module):
    def __init__(self, in_size, num_heads, dropout):
        super(TransformerEncoder, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=in_size, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(in_size)
        self.norm2 = nn.LayerNorm(in_size)
        self.fc = nn.Sequential(
            nn.Linear(in_size, in_size),
            nn.ReLU(),
            nn.Linear(in_size, in_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, attn_weight = self.attention(x, x, x)

        x = self.norm1(x + attn_output)  # Add & Norm

        ff_output = self.fc(x)
        return self.norm2(x + self.dropout(ff_output))  # Add & Norm


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size).apply(init),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False).apply(init)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)


class FastGTNLayer(nn.Module):
    def __init__(self, meta_paths, in_size, out_size, dropout, transformer_heads):
        super(FastGTNLayer, self).__init__()
        self.gat_layers = nn.ModuleList()

        self.gat_layers.append(GraphConv(in_size, out_size, activation=F.relu, allow_zero_in_degree=True).apply(init))

        self.semantic_attention = SemanticAttention(in_size=out_size)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self.linear = nn.Linear(out_size, 96)

        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self.dropout = dropout

        self.transformer = TransformerEncoder(
            out_size,
            num_heads=transformer_heads,
            dropout=dropout
        )

    def forward(self, g, h):
        semantic_embeddings = []
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                sub_g = dgl.metapath_reachable_graph(g, meta_path)

                pe = compute_laplacian_pe(sub_g, k=16)

                sub_g.ndata['pe'] = pe

                self._cached_coalesced_graph[meta_path] = sub_g

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]

            semantic_embeddings.append(self.gat_layers[0](new_g, h).flatten(1))

        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)

        semantic_embeddings = self.semantic_attention(semantic_embeddings)

        semantic_embeddings = self.linear(semantic_embeddings)
        pe_list = []
        for meta_path in self.meta_paths:
            sub_g = self._cached_coalesced_graph[meta_path]
            pe = sub_g.ndata['pe']
            pe_list.append(pe)

        pe_cat = torch.cat(pe_list, dim=-1)

        semantic_embeddings_with_pe = torch.cat([semantic_embeddings, pe_cat], dim=-1)

        final_embeddings = self.transformer(semantic_embeddings_with_pe)

        return final_embeddings


class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, dropout, transformer_heads):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()

        self.predict = nn.Linear(in_size, hidden_size, bias=False).apply(init)

        self.layers.append(FastGTNLayer(meta_paths, hidden_size, out_size, dropout, transformer_heads))

    def forward(self, g, h):
        h = self.predict(h)
        for gnn in self.layers:
            h = gnn(g, h)
        return h


class HAN_DTI(nn.Module):
    def __init__(self, all_meta_paths, in_size, hidden_size, out_size, dropout, transformer_heads):
        super(HAN_DTI, self).__init__()
        self.sum_layers = nn.ModuleList()

        for i in range(len(all_meta_paths)):
            self.sum_layers.append(
                HAN(all_meta_paths[i], in_size[i], hidden_size[i], out_size[i], dropout,
                    transformer_heads=transformer_heads)
            )

    def forward(self, s_g, s_h_1, s_h_2):
        h1 = self.sum_layers[0](s_g[0], s_h_1)
        h2 = self.sum_layers[1](s_g[1], s_h_2)
        return h1, h2


class GCN(nn.Module):
    def __init__(self, nfeat, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, 256)
        self.gc2 = GraphConvolution(256, 128)
        self.dropout = dropout

    def forward(self, x, adj):
        x = x.to(device)
        adj = adj.to(device)
        x1 = F.relu(self.gc1(x, adj), inplace=True)
        x1 = F.dropout(x1, self.dropout)
        x2 = self.gc2(x1, adj)
        res = x2
        return res


class CL_GCN(nn.Module):
    def __init__(self, nfeat, dropout, alpha=0.8):
        super(CL_GCN, self).__init__()
        self.gcn1 = GCN(nfeat, dropout)
        self.gcn2 = GCN(nfeat, dropout)
        self.tau = 0.5
        self.alpha = alpha

    def forward(self, x1, adj1, x2, adj2, clm):
        z1 = self.gcn1(x1, adj1)
        z2 = self.gcn1(x2, adj2)

        loss = self.alpha * self.sim(z1, z2, clm) + (1 - self.alpha) * self.sim(z2, z1, clm)

        return z1, z2, loss

    def sim(self, z1, z2, clm):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        sim_matrix = sim_matrix / (torch.sum(sim_matrix, dim=1).view(-1, 1) + 1e-8)
        sim_matrix = sim_matrix.to(device)
        loss = -torch.log(sim_matrix.mul(clm).sum(dim=-1)).mean()

        return loss

    def mix2(self, z1, z2):
        loss = ((z1 - z2) ** 2).sum() / z1.shape[0]
        return loss


class LinearEncoder(nn.Module):
    def __init__(self, in_dim_p, in_dim_d, out_dim=128):
        super().__init__()
        self.linear_p = nn.Linear(in_dim_p, out_dim)
        self.linear_d = nn.Linear(in_dim_d, out_dim)

    def forward(self, x1, x2):
        z1 = self.linear_p(x1)
        z2 = self.linear_d(x2)
        return z1, z2


class MLPEncoder(nn.Module):
    def __init__(self, in_dim_p, in_dim_d, out_dim=128, dropout=0.5):
        super().__init__()
        self.encoder_p = nn.Sequential(
            nn.Linear(in_dim_p, out_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim)
        )

    def forward(self, x1, x2):
        z1 = self.encoder_p(x1)
        z2 = self.encoder_p(x2)
        return z1, z2


class MLP(nn.Module):
    def __init__(self, nfeat):
        super(MLP, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(nfeat, 32, bias=False).apply(init),
            nn.ELU(),
            nn.Linear(32, 2, bias=False),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        output = self.MLP(x)
        return output


class HMTCL(nn.Module):
    def __init__(self, all_meta_paths, in_size, hidden_size, out_size, dropout, transformer_heads):
        super(HMTCL, self).__init__()
        self.HAN_DTI = HAN_DTI(all_meta_paths, in_size, hidden_size, out_size, dropout, transformer_heads)

        self.CL_GCN = CL_GCN(256, dropout)

        self.MLP = MLP(256)

    def forward(self, graph, h, cl, dateset_index, data, topk, save_path, iftrain=True, p=None, d=None):
        if iftrain:
            p, d = self.HAN_DTI(graph, h[0], h[1])

        edge, feature = constructure_graph(data, p, d, save_path)

        f_edge, f_feature = constructure_knngraph(data, p, d, topk=topk)
        feature1, feature2, cl_loss1 = self.CL_GCN(feature, edge, f_feature, f_edge, cl)
        pred1 = self.MLP(torch.cat((feature1, feature2), dim=1)[dateset_index])
        if iftrain:
            return pred1, cl_loss1, p, d
        return pred1
