import torch
from torch_geometric.data import HeteroData
from layers.TSEncoder import *
from utils import *
from tqdm import *
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from typing import Dict, List
from torch_geometric.data import Dataset as GeoDataset         
from tqdm import tqdm
from layers.TSEncoder import AddGlobalLaplacianPE, AddMetaPathRandomWalkSE
from utils import convert_to_relative_time

class PatientGraph(torch.utils.data.Dataset):                   # behaves like a PyTorch Dataset
    def __init__(self, tokenizer, subset, dim, device, trans_dim = 0, di = False):
        self.c_tokenizer = tokenizer['cond_hist']
        self.d_tokenizer = tokenizer['drugs']
        self.p_tokenizer = tokenizer['procedures']
        self.dataset = subset
        self.di_edge = di
        self.dim = dim
        self.se = False
        if trans_dim!=0:
            self.se = True         
            self.global_pe_transform = AddGlobalLaplacianPE(k=trans_dim, device = device)
            self.local_se_transform = AddMetaPathRandomWalkSE(trans_dim, device = device)

    # ---------- standard Dataset API ----------------------------------------

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> HeteroData:
        """Build and return ONE HeteroData graph (no persistent cache)."""
        sample = self.dataset[idx]
        return self.build_graph(sample)
    
    ########## for per batch process
    

    def build_graph(self, sample):  # sample = one visit dict
        data = HeteroData()
        num_visit = len(sample['procedures'])
        data['visit'].x = torch.zeros(num_visit, self.dim)

        dpc = self.c_tokenizer.batch_encode_2d(sample['cond_hist'], padding=False)
        dpp = self.p_tokenizer.batch_encode_2d(sample['procedures'], padding=False)
        dpd = self.d_tokenizer.batch_encode_2d(sample['drugs'], padding=False)

        data['visit'].time = convert_to_relative_time(sample['adm_time'])

        data['co'].x = torch.zeros(self.c_tokenizer.get_vocabulary_size(), self.dim)
        data['pr'].x = torch.zeros(self.p_tokenizer.get_vocabulary_size(), self.dim)
        data['dh'].x = torch.zeros(self.d_tokenizer.get_vocabulary_size(), self.dim)

        civ = torch.tensor([[item for sublist in dpc for item in sublist],
                            [index for index, sublist in enumerate(dpc) for _ in sublist]], dtype=torch.int64)
        piv = torch.tensor([[item for sublist in dpp for item in sublist],
                            [index for index, sublist in enumerate(dpp) for _ in sublist]], dtype=torch.int64)
        div = torch.tensor([[item for sublist in dpd for item in sublist],
                            [index for index, sublist in enumerate(dpd) for _ in sublist]], dtype=torch.int64)

        data['pr', 'in', 'visit'].edge_index = piv
        data['dh', 'in', 'visit'].edge_index = div

        viv = torch.tensor([[i for i in range(num_visit - 1)],
                            [i + 1 for i in range(num_visit - 1)]], dtype=torch.int64)
        data['visit', 'connect', 'visit'].edge_index = viv if self.di_edge else torch.cat([viv, viv.flip(0)], dim=1)

        fciv = civ.flip(0)
        fpiv = piv.flip(0)
        fdiv = div.flip(0)

        chronic_edges, acute_edges = [], []

        original_times = data['visit'].time        # minutes (already relative)
        sorted_indices = sorted(range(len(original_times)), key=lambda i: original_times[i])  # indices sorted by time
        visit_times = [original_times[i] for i in sorted_indices]  
        # civ: 2×E  ->  row0 = code_ids, row1 = visit_idx
#             print('visit_times',visit_times)
        code_ids   = civ[0].tolist()
        visit_idxs = civ[1].tolist()

        # build code -> list[visit_idx] (for this patient only)
        code2visits = {}
        for cid, v in zip(code_ids, visit_idxs):
            code2visits.setdefault(cid, []).append(v)

        for cid, v_list in code2visits.items():
            # sort visits for this code by time (ascending)
            v_list = sorted(v_list, key=lambda x: visit_times[x])
            first_time = visit_times[v_list[0]]
            for v in v_list:
                gap_days = (visit_times[v] - first_time) / (60*24)
                is_chronic = gap_days > 90 and v != v_list[0]   # first visit is always acute
                if is_chronic:
                    chronic_edges.append((cid, v))
#                         print(f"[Patient {dp}] visit {v} code_id {cid} → CHRONIC (gap {gap_days:.1f} d)")
                else:
                    acute_edges.append((cid, v))
#                         print(f"[Patient {dp}] visit {v} code_id {cid} → ACUTE")


        def make_edge_tensor(edge_list):
            if edge_list:
                idx = torch.tensor(edge_list, dtype=torch.int64).t()
                et = torch.tensor([visit_times[v] for _, v in edge_list], dtype=torch.float32)
            else:
                idx = torch.empty((2, 0), dtype=torch.int64)
                et = torch.empty(0, dtype=torch.float32)
            return idx, et

        chronic_idx, chronic_time = make_edge_tensor(chronic_edges)
        acute_idx, acute_time = make_edge_tensor(acute_edges)

        data['co', 'chronic_in', 'visit'].edge_index = chronic_idx
        data['co', 'chronic_in', 'visit'].edge_time = chronic_time
        data['visit', 'has_chronic', 'co'].edge_index = chronic_idx.flip(0)

        data['co', 'acute_in', 'visit'].edge_index = acute_idx
        data['co', 'acute_in', 'visit'].edge_time = acute_time
        data['visit', 'has_acute', 'co'].edge_index = acute_idx.flip(0)

        data['pr', 'in', 'visit'].edge_time = torch.tensor(
            [index for index, sublist in enumerate(dpp) for _ in sublist], dtype=torch.float32)
        data['dh', 'in', 'visit'].edge_time = torch.tensor(
            [index for index, sublist in enumerate(dpd) for _ in sublist], dtype=torch.float32)

        data['visit', 'has', 'pr'].edge_index = fpiv
        data['visit', 'has', 'dh'].edge_index = fdiv

        # PE + SE
        if self.se:
            data = self.global_pe_transform.apply_laplacian_pe(data)
            f_metapaths = [
                [('co', 'acute_in', 'visit'), ('visit', 'has', 'pr'), ('pr', 'in', 'visit'), ('visit', 'has_acute', 'co')],
                [('co', 'chronic_in', 'visit'), ('visit', 'has', 'pr'), ('pr', 'in', 'visit'), ('visit', 'has_chronic', 'co')],
                [('co', 'chronic_in', 'visit'), ('visit', 'has', 'pr'), ('pr', 'in', 'visit'), ('visit', 'has_acute', 'co')],
                [('co', 'acute_in', 'visit'), ('visit', 'has', 'pr'), ('pr', 'in', 'visit'), ('visit', 'has_chronic', 'co')],
                [('pr', 'in', 'visit'), ('visit', 'has_chronic', 'co'), ('co', 'chronic_in', 'visit'), ('visit', 'has', 'pr')],
                [('pr', 'in', 'visit'), ('visit', 'has_acute', 'co'), ('co', 'acute_in', 'visit'), ('visit', 'has', 'pr')],
                [('dh', 'in', 'visit'), ('visit', 'has', 'dh')]
            ]
            data = self.local_se_transform.forward(data=data, metapaths=f_metapaths)
            for node_type in data.node_types:
                if node_type not in ['co', 'pr', 'dh']:
                    del data[node_type].laplacian_pe
                    del data[node_type].random_walk_se

        return data

 