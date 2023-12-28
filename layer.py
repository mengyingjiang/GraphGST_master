import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)
    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x_embed, position_embed, position_original, r_positive):
        b_xent = nn.BCEWithLogitsLoss().to(device)
        position = F.normalize(position_original)
        position_simi = torch.mm(position,position.T)
        row, colum = torch.where(position_simi > r_positive)
        scores_1 = self.f_k(x_embed[row,:], position_embed[colum,:])
        scores_2 = self.f_k(position_embed[row,:], x_embed[colum,:])
        scores = torch.cat((scores_1, scores_2), dim=0)
        lbl = torch.ones(scores_1.shape[0]*2, 1).to(device)
        loss_ss = b_xent(scores, lbl.float())
        return loss_ss

class position_Encoder(torch.nn.Module):
    def __init__(self, fea_dim, input_dim, n_heads):  # [1966,4]
        super(position_Encoder, self).__init__()
        self.attn = MultiHead(fea_dim, input_dim, n_heads)   # [1966,4]
        self.AN1 = torch.nn.LayerNorm(input_dim)
        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.fc = torch.nn.Linear(input_dim, input_dim)
        self.l2 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

    def forward(self, new_fe, X): # (X_fea_target, X_fea_target_anchor, anchors_features, X)
        output = self.attn(new_fe, X)  # 多头注意力去聚合其他DDI的特征
        X = self.AN1(output + X)  # 残差连接+LayerNorm
        output = self.l2(X)  # FC
        X = self.AN2(output + X)  # 残差连接+LayerNorm
        return X

class MultiHead(torch.nn.Module):
    def __init__(self, fea_dim, input_dim, n_heads):   # [1966,4]
        super(MultiHead, self).__init__()
        self.d_k = self.d_v = input_dim//n_heads
        self.n_heads = n_heads
        self.output_dim = input_dim
        self.W_Q = torch.nn.Linear(fea_dim, self.d_k * self.n_heads, bias=False)  # [1966,491]*4
        # self.W_K = torch.nn.Linear(fea_dim, self.d_k * self.n_heads, bias=False)  # [1966,491]*4
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)  # [1966,491]*4
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.output_dim, bias=False)  # [1966,491]*4

    def forward(self, new_fe, features): # new_fe, new_fe_anchors, anchors_features
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(new_fe).view(-1, self.n_heads, self.d_k).transpose(0, 1)  # [256,1966]->[256,1964]->[256,4,491]->[4,256,625]
        # K = self.W_K(new_fe).view(-1, self.n_heads, self.d_k).transpose(0, 1)  # [256,1966]->[256,1964]->[256,4,491]->[4,256,625]
        V = self.W_V(features).view(-1, self.n_heads, self.d_v).transpose(0, 1)  # [256,1966]->[256,1964]->[256,4,491]->[4,256,625]
        scores = torch.matmul(Q, Q.transpose(-1, -2)) / np.sqrt(self.d_k)  # [4,256,491]*[4,491,256]->[4,256,256]
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        # context = torch.matmul(attn, V)
        context = F.relu(torch.matmul(attn, V)) # [4,256,256]*[4,256,625]->[4,256,491]
        # context: [len_q, n_heads * d_v]
        output = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(output)
        return output

class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, n_heads):  # [1966,4]
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(input_dim, n_heads)   # [1966,4]
        self.AN1 = torch.nn.LayerNorm(input_dim)
        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X):
        output = self.attn(X)  # 多头注意力去聚合其他DDI的特征
        X = self.AN1(output + X)  # 残差连接+LayerNorm
        output = self.l1(X)  # FC
        X = self.AN2(output + X)  # 残差连接+LayerNorm

        return X

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):   # [1966,4]
        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads  # 491.5
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim  #1966
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)  # [1966,491]*4
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)  # [1966,491]*4
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)  # [1966,491]*4
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)  # [1966,491]*4

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)  # [256,1966]->[256,1964]->[256,4,491]->[4,256,625]
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)  # [256,1966]->[256,1964]->[256,4,491]->[4,256,625]
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)  # [256,1966]->[256,1964]->[256,4,491]->[4,256,625]
        scores = torch.matmul(K, Q.transpose(-1, -2)) / np.sqrt(self.d_k) # [4,256,491]*[4,491,256]->[4,256,256]
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = F.relu(torch.matmul(attn, V)) # [4,256,256]*[4,256,625]->[4,256,491]
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))



