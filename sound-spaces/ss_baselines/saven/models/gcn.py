
from pdb import set_trace
import pickle

import scipy.sparse as sp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()

        # get and normalize adjacency matrix.
        adjmat_path = r"data/glove_data/adjmat.bin"
        bin_file = open(adjmat_path, "rb")
        A_raw = pickle.load(bin_file)
        bin_file.close()
        A = normalize_adj(A_raw).tocsr().toarray()
        self.A = torch.nn.Parameter(torch.Tensor(A))

        embeddings_path = r"data/glove_data/glove_embeddings_300d.bin"
        bin_file = open(embeddings_path, "rb")
        objects_vector = pickle.load(bin_file)
        regions_vector = pickle.load(bin_file)
        bin_file.close()

        objects = list(sorted(objects_vector.keys()))
        regions = list(sorted(regions_vector.keys()))

        self.n = len(objects) + len(regions)

        all_glove = torch.zeros(self.n, 300)
        i = 0
        for obj in objects:
            all_glove[i, :] = torch.Tensor(objects_vector[obj])
            i += 1
        for reg in regions:
            all_glove[i, :] = torch.Tensor(regions_vector[reg])
            i += 1

        self.all_glove = nn.Parameter(all_glove)
        self.all_glove.requires_grad = False

        self.get_word_embed = nn.Linear(300, self.n)

        self.W0 = nn.Linear(self.n * 2, 1024, bias=False)
        self.W1 = nn.Linear(1024, 1024, bias=False)
        self.W2 = nn.Linear(1024, 1, bias=False)

        self.feature_dims = 256 - 2
        self.final_mapping = nn.Linear(self.n, self.feature_dims)

    def forward(self, class_embed):

        class_embed = class_embed.reshape(1, -1)
        word_embed = self.get_word_embed(self.all_glove.detach())
        x = torch.cat((class_embed.repeat(self.n, 1), word_embed), dim=1)
        x = torch.mm(self.A, x)
        x = F.relu(self.W0(x))
        x = torch.mm(self.A, x)
        x = F.relu(self.W1(x))
        x = torch.mm(self.A, x)
        x = F.relu(self.W2(x))
        x = x.view(1, self.n)
        x = self.final_mapping(x)

        return x

import dgl
from dgl.nn import GraphConv
import scipy.sparse as spp


class DGL_GCN(nn.Module):
    def __init__(self, in_feats=(256-2)*2, o_feats=1, debug=False):
        super(DGL_GCN, self).__init__()
        assert in_feats % 2 == 0, "in_feats should be even."
        self.debug = debug
        adjmat_path = r"data/glove_data/adjmat.bin"
        with open(adjmat_path, 'rb') as f:
            adj_mat = pickle.load(f)
        adj_mat = spp.coo_matrix(adj_mat + np.eye(adj_mat.shape[0]))
        self.g = dgl.DGLGraph(adj_mat)

        embeddings_path = r"data/glove_data/glove_embeddings_300d.bin"
        with open(embeddings_path, 'rb') as bin_file:
            objects_vector = pickle.load(bin_file)
            regions_vector = pickle.load(bin_file)

        objects = list(sorted(objects_vector.keys()))  # 21 objects: ['bathtub', ..., 'tv_monitor']
        regions = list(sorted(regions_vector.keys()))  # 24 regions: ['balcony', ..., 'workout/gym/exercise']
        self.n = len(objects) + len(regions)
        if self.debug:
            print("objects: ", len(objects))  # , objects)
            print("regions: ", len(regions))  # , regions)

        all_glove = torch.zeros(self.n, 300)
        i = 0
        for obj in objects:
            all_glove[i, :] = torch.Tensor(objects_vector[obj])
            i += 1
        for reg in regions:
            all_glove[i, :] = torch.Tensor(regions_vector[reg])
            i += 1
        if self.debug:
            print("all_glove: ", all_glove.shape)
            print("all_glove ", all_glove)

        self.all_glove = nn.Parameter(all_glove)
        self.all_glove.requires_grad = False
        self.feature_dims = int(in_feats / 2)
        self.get_word_embed = nn.Linear(300, self.feature_dims)
        self.get_img_embed = nn.Linear(self.n, self.feature_dims)

        self.conv1 = GraphConv(in_feats, in_feats)
        self.conv2 = GraphConv(in_feats, in_feats)
        self.conv3 = GraphConv(in_feats, o_feats)

        self.final_mapping = nn.Linear(self.n, self.feature_dims)

    def forward(self, class_embed):

        class_embed = class_embed.reshape(1, -1)
        class_embed = self.get_img_embed(class_embed)
        if self.debug:
            print("class_embed: ", class_embed.shape)
        word_embed = self.get_word_embed(self.all_glove)
        if self.debug:
            print("word_embed: ", word_embed.shape)
        x = torch.cat((class_embed.repeat(self.n, 1), word_embed), dim=1)
        if self.debug:
            print("torch.cat((class_embed.repeat(self.n, 1), word_embed), dim=1): ", x.shape)

        if self.debug:
            print("\n**************devices**************")
            print(f"g.device: {self.g.device}")
            print(f"conv1.device: {self.conv1.weight.device}")
            print(f"conv2.device: {self.conv2.weight.device}")
            print(f"conv3.device: {self.conv3.weight.device}")
            print(f"final_mapping.weight.device: {self.final_mapping.weight.device}")
            print("**************devices**************")

        self.g = self.g.to(class_embed.device)
        h = self.conv1(self.g, x)
        h = F.relu(h)
        h = self.conv2(self.g, h)
        h = F.relu(h)
        h = self.conv3(self.g, h)
        o = F.relu(h)
        if self.debug:
            print(f"o: {o.shape}")
        o = o.view(1, -1)
        o = self.final_mapping(o)

        return o

# if __name__ == '__main__':
#     model = DGL_GCN(in_feats=(256-2)*2, debug=True)
#     model = model.to('cuda')
#     joint_rep = torch.rand(45).to('cuda')
#     output = model(joint_rep)
#     print(f'The shape of output of GCN: {output.shape}')
