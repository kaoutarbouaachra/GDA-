"""PyTorch Geometric adapter for JSONL created by scripts/generate_motifs.py.

Compatible with dataset.EgonetLoader; node labels are uniform (one category).
Returns original graph indices to preserve ground truth in separate JSONL.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

class SyntheticGraphsDataset:
    def __init__(self,path,hops=1):
        self.path=Path(path)
        self.records=[json.loads(s) for s in self.path.read_text(encoding='utf-8').splitlines() if s.strip()]
        if not self.records:
            raise ValueError('Empty synthetic JSONL')
        self.hops=int(hops)
        self.num_features=1
        self.num_classes=max(x['label'] for x in self.records)+1
        self.cache={}

    def __len__(self):
        return len(self.records)

    def __getitem__(self,index):
        i=int(index)
        if i in self.cache:
            return self.cache[i]
        record=self.records[i]
        n=int(record['n_nodes'])
        x=torch.ones(n,1,dtype=torch.float32)
        ed=np.asarray(record['edges'],dtype=np.int64)
        if ed.size:
            oriented=np.vstack([ed,ed[:,::-1]])
            edge_index=torch.tensor(oriented.T,dtype=torch.long)
        else:
            edge_index=torch.zeros((2,0),dtype=torch.long)
        ego=[k_hop_subgraph([j],num_hops=self.hops,edge_index=edge_index,
                            relabel_nodes=True,num_nodes=n)[:2] for j in range(n)]
        eidx=[ids.detach().cpu().numpy() for ids,_ in ego]
        eedge=[eds.detach().cpu().numpy() for _,eds in ego]
        data=Data(x=x,edge_index=edge_index,y=torch.tensor(record['label'],dtype=torch.long),
                  nidx=torch.arange(n),egonets_idx=eidx,egonets_edg=eedge,nodes=n)
        self.cache[i]=data
        return data
