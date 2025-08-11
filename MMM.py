import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.parameter import Parameter
import pickle

class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, mask):
        weight = torch.mul(self.weight, mask)
        output = torch.mm(input, weight)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )

# MMM model: ELF + Bipartite model
ELF_embedding_path='/projects/MMM/mcwon/PRIME_MMM/elf_efficientnet_v2_l_REAL.pkl'
with open(ELF_embedding_path, 'rb') as f:
    loaded_data = pickle.load(f)

class MMM(nn.Module):
    def __init__(self, vocab_size, ddi_adj, ddi_mask_H, emb_dim=256, elf_in_dim=128, device=torch.device("cpu:0")):
        super(MMM, self).__init__()

        self.device = device
        
        # pre-embedding
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(2)]
        )
        self.dropout = nn.Dropout(p=0.5)
        
        # GRU(Longitudinal Patient Representation)
        self.encoders = nn.ModuleList(
            [nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(2)]
        )
        
        self.query = nn.Sequential(nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim))

        # Local Bipartite Embedding
        self.bipartite_transform = nn.Sequential(
            nn.Linear(emb_dim, ddi_mask_H.shape[1])
        )
        self.bipartite_output = MaskLinear(ddi_mask_H.shape[1], vocab_size[2], False)
        
        # ELF - Image Embedding
        embedding_matrix = [item[1] for item in loaded_data]
        embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float)
        self.ELF_emb = embedding_tensor
        self.ELF_emb = nn.Parameter(self.ELF_emb, requires_grad=True)
        # ELF - MLP
        self.ELF_proj = nn.Sequential(
            nn.Linear(elf_in_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.ELF_output = nn.Linear(vocab_size[2], vocab_size[2])
        self.ELF_layernorm = nn.LayerNorm(vocab_size[2])
        
        # DDI Adjacency and Mask
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)

        self.init_weights()

    def forward(self, input):

        # patient health representation
        i1_seq = []
        i2_seq = []

        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)
        
        for adm in input:
            i1 = sum_embedding(
                self.dropout(
                    self.embeddings[0](
                        torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )
            i2 = sum_embedding(
                self.dropout(
                    self.embeddings[1](
                        torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )
            i1_seq.append(i1) 
            i2_seq.append(i2)
        i1_seq = torch.cat(i1_seq, dim=1)
        i2_seq = torch.cat(i2_seq, dim=1)
        
        # GRU based forward
        o1, h1 = self.encoders[0](i1_seq) #o1: diagnosis sequence encoding
        o2, h2 = self.encoders[1](i2_seq) #o2: procedure sequence encoding
        patient_representations = torch.cat([o1, o2], dim=-1).squeeze(
            dim=0
        )
        
        query = self.query(patient_representations)[-1:, :] # patient representation final vector

        # ELF embedding
        ELF_MLP = self.ELF_proj(self.ELF_emb) 
        ELF_MLP_T = ELF_MLP.t()
        
        ELF_match = torch.sigmoid(torch.mm(query, ELF_MLP_T)) # (1, med_num)
        ELF_att = self.ELF_layernorm(ELF_match + self.ELF_output(ELF_match))

        # Local bipartite embedding
        bipartite_emb = self.bipartite_output(
            F.sigmoid(self.bipartite_transform(query)), self.tensor_ddi_mask_H.t()
        )

        # element-wise mul(global * local)
        result = torch.mul(bipartite_emb, ELF_att)
        
        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob

        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        
        return result, batch_neg

    def init_weights(self):
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
