import dgl
from NeuroGraph.datasets import NeuroGraphDataset
from dgl.data.utils import load_graphs, save_graphs, Subset
import os
import shutil
import torch
from enum import Enum
import json


class NeuroDatasetName(Enum):
    GENDER = "HCPGender"

    def str_to_dataset(dataset_name: str):
        if dataset_name == "GENDER":
            return NeuroDatasetName.GENDER
        else:
            raise ValueError(f"Dataset {dataset_name} not found")
    
    def list():
        return ["GENDER"]


def load_neuro_indexes(dataset_name: NeuroDatasetName):
    limit = os.environ.get("SIZE_LIMIT")
    if limit is None:
        path = os.path.join('data', 'data_splits', dataset_name.value + ".json")
    else:
        path = os.path.join('data', 'data_splits', f"{dataset_name.value}_{limit}.json")
        limit = int(limit)
    if not os.path.exists(path):
        from generate_splits import generate

        generate(dataset_name, limit)
    with open(path, "r") as f:
        indexes = json.load(f)
    return indexes


class NeuroDatasetPrep(object):
    def __init__(self, dataset_name: NeuroDatasetName):
        self.dataset_name = dataset_name.value
        raw_data_dir = os.path.join('/net/tscratch/people/plgglegeza', 'data', 'datasets', self.dataset_name, 'raw')
        prep_data_dir = os.path.join('/net/tscratch/people/plgglegeza', 'data', 'datasets', self.dataset_name, 'prep')
        
        if os.path.exists(prep_data_dir):
            self.graphs, label_dict = load_graphs(prep_data_dir)
            self.labels = label_dict['labels']
        else:
            if not os.path.exists(raw_data_dir):
                os.makedirs(raw_data_dir)
            dataset = NeuroGraphDataset(name=self.dataset_name, root=raw_data_dir)
            # shutil.rmtree(raw_data_dir)
            self.labels = [graph.y for graph in dataset]
            self.labels = torch.tensor(self.labels).long()
            self.graphs = self._preprocess_graphs(dataset)
            save_graphs(prep_data_dir, self.graphs, labels={'labels': self.labels})

        self.num_class = max([label for label in self.labels]) + 1

    def _preprocess_graphs(self, graphs):
        preprocessed_graphs = []
        for graph in graphs:
            src, dst = graph.edge_index
            num_nodes = graph.num_nodes

            if num_nodes == 1:  # graphs can have one node
                A_ = torch.tensor(1.).view(1, 1)
            else:
                A = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
                A[src, dst] = 1.0
                for i in range(num_nodes):
                    A[i, i] = 1.0
                deg = torch.sum(A, axis=0).squeeze() ** -0.5
                D = torch.diag(deg)
                A_ = D @ A @ D
            e, u = torch.linalg.eigh(A_)

            # fully_connected = torch.ones((num_nodes, num_nodes), dtype=torch.float).nonzero(as_tuple=True)
            # g = dgl.graph(fully_connected, num_nodes=num_nodes)

            ### Note: This should be fully_connected matrix but requires more memory

            adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
            adj[src, dst] = 1.0
            for i in range(num_nodes):
                adj[i, i] = 1.0
            adj = adj.nonzero(as_tuple=True)
            g = dgl.graph(adj, num_nodes=num_nodes)

            g.ndata['e'] = e
            g.ndata['u'] = u

            g.ndata['feat'] = graph['x'].float() # embedding of node features
            g.edata['feat'] = torch.zeros((g.num_edges(), 1)).long() # there are no edge lables/encoding

            preprocessed_graphs.append(g)

            del graph
        del graphs
        return preprocessed_graphs
    
    def __getitem__(self, idx):
        '''Get datapoint with index'''

        if isinstance(idx, int):
            return self.graphs[idx], self.labels[idx]
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            if idx.dim() == 0:
                return self.graphs[idx], self.labels[idx]
            elif idx.dim() == 1:
                return Subset(self, idx.cpu())

        raise IndexError(
            'Only integers and long are valid '
            'indices (got {}).'.format(type(idx).__name__))

    def __len__(self):
        '''Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        '''
        return len(self.graphs)
