import sys
import os
sys.path.append(os.path.dirname(__file__)) 
from HGT import *
from Seqmodels import *
from layers.TSEncoder import *
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.data import Data
from collections import defaultdict
import torch
from torch_geometric.data import Data
from utils import *
import torch.nn.functional as F

feats_to_nodes = {
    'cond_hist': 'co',
    'procedures': 'pr',
    'drugs': 'dh',
    'co': 'cond_hist',
    'pr': 'procedures',
    'dh': 'drugs'
}


graph_meta = (['visit', 'co', 'pr', 'dh'],
[('co', 'acute_in', 'visit'),
 ('co', 'chronic_in', 'visit'),
 ('pr', 'in', 'visit'),
 ('dh', 'in', 'visit'),
 ('visit', 'connect', 'visit'),
 ('visit', 'has_acute', 'co'),
 ('visit', 'has_chronic', 'co'),
 ('visit', 'has', 'pr'),
 ('visit', 'has', 'dh')])

def build_co_graph(dataset, tokenizer, threshold=0.05):

    co_count = defaultdict(int)
    edge_weights = defaultdict(int)

    for patient in dataset:
        visits = patient['cond_hist']
        for visit in visits:
            encoded_visit = tokenizer.batch_encode_2d([visit])[0]
            unique_conditions = list(set(encoded_visit))

            for i, co1 in enumerate(unique_conditions):
                co_count[co1] += 1
                for j in range(i + 1, len(unique_conditions)):
                    co2 = unique_conditions[j]
                    edge = tuple(sorted((co1, co2)))
                    edge_weights[edge] += 1

    # Build tensors
    edge_keys = list(edge_weights.keys())
    edge_vals = torch.tensor([edge_weights[e] for e in edge_keys], dtype=torch.float)

    # Normalize edge weights
    edge_vals = edge_vals / edge_vals.max()

    # Apply threshold
    mask = edge_vals > threshold
    filtered_edges = [edge_keys[i] for i, keep in enumerate(mask) if keep]
    filtered_weights = edge_vals[mask]

    if not filtered_edges:
        raise ValueError("No co-occurrence edges passed the threshold. Try lowering the threshold.")

    edge_index = torch.tensor(filtered_edges, dtype=torch.long).t().contiguous()
    num_nodes = tokenizer.get_vocabulary_size()

    co_graph = Data(x=torch.eye(num_nodes), edge_index=edge_index, edge_attr=filtered_weights)
    print('co_graph',co_graph)
    return co_graph


class MuHGTeL(nn.Module):
    def __init__(
        self,
        Tokenizers,
        hidden_size,
        output_size,
        device,
        graph_meta,
        embedding_dim = 128,
        embedding_dim1=264,
        dropout = 0.5,
        num_heads = 4,
        num_layers = 2,
        pe = False,
    ):
        super(MuHGTeL, self).__init__()
        self.embedding_dim = embedding_dim
        self.feat_tokenizers = Tokenizers
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()
        self.feature_keys = Tokenizers.keys()
        self.device = device
        for feature_key in self.feature_keys:
            self.add_feature_transform_layer(feature_key)

        self.transformer = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.transformer[feature_key] = TransformerLayer(
                feature_size=embedding_dim, dropout=dropout
            )
        self.tim2vec = Time2Vec(8).to(device)
        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)
        self.graphmodel = HGT(hidden_channels = hidden_size, out_channels = output_size, num_heads=num_heads, num_layers = num_layers, metadata = graph_meta).to(device)
        self.pe = pe
        self.spatialencoder = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.spatialencoder[feature_key] = nn.Linear(self.pe*2, embedding_dim)#.to(self.device)
        self.alpha = 0.8
        self.gcn = GCNConv(264, embedding_dim1) #(embedding_dim1,embedding_dim1)
        self.co_gat1 = GATConv(embedding_dim, embedding_dim, heads=4)
        self.co_gat2 = GATConv(embedding_dim*4, embedding_dim, heads=4)
        self.fc1= nn.Linear(512,128)
        self.co_proj = nn.Linear(512, embedding_dim)  # project 512 → 128
        self.final_fc = nn.Linear(528, output_size)#(2 * 264, output_size)
        self.w1 = nn.Parameter(torch.tensor(1.0))  # Learnable scalar
        self.w2 = nn.Parameter(torch.tensor(0.0))  # Learnable bias




    def add_feature_transform_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]
        self.embeddings[feature_key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                self.embedding_dim,
                padding_idx=tokenizer.get_padding_index(),
            )

    def get_embedder(self):
        feature = {}
        for k in self.embeddings.keys():
            lenth = self.feat_tokenizers[k].get_vocabulary_size()
            tensor = torch.arange(0, lenth, dtype=torch.long).to(self.device)
            feature[k] = self.embeddings[k](tensor)
        
        # Enhance condition embeddings using global co graph
        if hasattr(self, 'co_graph'):
            
            x = feature['cond_hist']
            edge_index = self.co_graph.edge_index.to(self.device)
            edge_weight = self.co_graph.edge_attr.to(self.device)

            h = F.relu(self.co_gat1(x, edge_index, edge_weight))
            h = self.co_gat2(h, edge_index, edge_weight)
            h=self.fc1(h)
            x=x+h
            feature['cond_hist'] = x  # Updated co embeddings

        return feature

        
    def process_seq(self, seqdata):
        patient_emb = []
        for feature_key in self.feature_keys:
            x = self.feat_tokenizers[feature_key].batch_encode_3d(
                seqdata[feature_key],
            )
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            x = self.embeddings[feature_key](x)
            x = torch.sum(x, dim=2)
            mask = torch.any(x !=0, dim=2)
            _, x = self.transformer[feature_key](x, mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)
        return logits, patient_emb
    
    def process_graph_fea(self, graph_list, pe):
        f = self.get_embedder()
        for i in range(len(graph_list)):
            for node_type, x in graph_list[i].x_dict.items():
                if node_type!='visit':
                    if self.pe:
                        lpe = graph_list[i][node_type].laplacian_pe.to(self.device)
                        rws = graph_list[i][node_type].random_walk_se.to(self.device)
                        se = self.spatialencoder[feats_to_nodes[node_type]](torch.cat([lpe,rws], dim=-1))
                        
                        # graph_list[i][node_type].x = torch.cat([f[feats_to_nodes[node_type]],\
                        #                                     lpe, \
                        #                                     rws], dim=-1)
                        graph_list[i][node_type].x = f[feats_to_nodes[node_type]] + se

                    else:
                        graph_list[i][node_type].x = f[feats_to_nodes[node_type]]
                if node_type=='visit':
                    timevec = self.tim2vec(torch.tensor(graph_list[i]['visit'].time, dtype = torch.float32, device=self.device))
                    num_visit = graph_list[i]['visit'].x.shape[0]
                    graph_list[i]['visit'].x = torch.cat([pe[i].repeat(num_visit, 1), timevec],dim=-1)
        return Batch.from_data_list(graph_list)
    


    def forward(self, batchdata):  
        seq_logits, patient_features = self.process_seq(batchdata[0])  # patient_features = z_v (initial embedding)
        graph_data = self.process_graph_fea(batchdata[1], patient_features).to(self.device)
        graph_out = self.graphmodel(graph_data.edge_index_dict, graph_data)
        out = self.alpha * graph_out + (1 - self.alpha) * seq_logits  # intermediate prediction
        z = out  # updated patient representation z_v ∈ ℝ^d
        z_norm = F.normalize(z, dim=1)  # [B, d]
        sim_matrix = torch.matmul(z_norm, z_norm.T)  # [B, B], sim(u, v)
        sim_flat = sim_matrix.view(-1, 1)  # [B*B, 1]
        sim_edge_weights = torch.sigmoid(self.w1 * sim_flat + self.w2)  # e_uv ∈ (0,1), shape: [B*B, 1]
        edge_weight = sim_edge_weights.view(sim_matrix.shape)  # [B, B]
        B = z.shape[0]
        src, tgt = torch.meshgrid(torch.arange(B), torch.arange(B), indexing='ij')
        edge_index = torch.stack([src.reshape(-1), tgt.reshape(-1)], dim=0).to(self.device)
        edge_weight = edge_weight.reshape(-1).to(self.device)
        patient_graph = Data(x=z, edge_index=edge_index, edge_attr=edge_weight)
        patient_graph_out = self.gcn(patient_graph.x, patient_graph.edge_index, patient_graph.edge_attr)  # updated z_v^{(l+1)}
        fusion = torch.cat([z, patient_graph_out], dim=1)
        logits = self.final_fc(fusion)

        return logits, patient_graph_out





