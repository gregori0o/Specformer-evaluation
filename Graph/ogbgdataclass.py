import dgl
from dgl.data import TUDataset
from dgl.data.utils import load_graphs, save_graphs, Subset
import os
import shutil
import torch
from torch_geometric.utils import to_dense_adj
from enum import Enum
import json

from ogb.graphproppred import DglGraphPropPredDataset
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from time_measure import time_measure

class OGBGDatasetPrep(object):
    def __init__(self, dataset_name, model_type: str):
        self.dataset_name = dataset_name
        raw_data_dir = os.path.join('/net/tscratch/people/plgglegeza', 'data', 'datasets', self.dataset_name, 'raw')
        prep_data_dir = os.path.join('/net/tscratch/people/plgglegeza', 'data', 'datasets', self.dataset_name, 'prep')
        
        # raw_data_dir = os.path.join('.', 'data', 'datasets', self.dataset_name, 'raw')
        # prep_data_dir = os.path.join('.', 'data', 'datasets', self.dataset_name, 'prep')
        
        if os.path.exists(prep_data_dir) and False:
            self.graphs, label_dict = load_graphs(prep_data_dir)
            self.labels = label_dict['labels']
        else:
            if not os.path.exists(raw_data_dir):
                os.makedirs(raw_data_dir)
            dataset = DglGraphPropPredDataset(name=self.dataset_name, root=raw_data_dir)
            shutil.rmtree(raw_data_dir)
            self.graphs = dataset.graphs
            self.labels = [int(label) for label in dataset.labels]
            self.labels = torch.tensor(self.labels).long()
            self.graphs = time_measure(self._preprocess_graphs, f"spec_{model_type}", self.dataset_name, "preparation")(self.graphs)
            # self.graphs = self._preprocess_graphs(self.graphs)
            save_graphs(prep_data_dir, self.graphs, labels={'labels': self.labels})

        self.num_node_labels = None
        self.num_edge_labels = None
        self.num_class = max([label for label in self.labels]) + 1

    def _preprocess_graphs(self, graphs):
        preprocessed_graphs = []
        for graph in graphs:
            src, dst = graph.edges()
            num_nodes = graph.num_nodes()

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

            fully_connected = torch.ones((num_nodes, num_nodes), dtype=torch.float).nonzero(as_tuple=True)
            g = dgl.graph(fully_connected, num_nodes=num_nodes)

            g.ndata['e'] = e
            g.ndata['u'] = u

            g.ndata['feat'] = graph.ndata['feat'].long()

            # if graph.ndata.get('node_labels') is not None:
            #     g.ndata['feat'] = graph.ndata['node_labels'].long()
            # else:
            #     g.ndata['feat'] = torch.zeros([num_nodes, 1]).long()

            # if graph.edata.get('edge_labels') is not None:
            edge_idx = torch.stack([src, dst], dim=0)
            edge_attr = graph.edata['feat'].long() + 1  # for padding

            if len(edge_attr.shape) == 1:
                edge_attr_dense = to_dense_adj(edge_idx, edge_attr=edge_attr.unsqueeze(-1)).squeeze(0).squeeze(-1).view(-1)
            else:
                if edge_attr.size(0) == 0:   # for graphs without edge
                    edge_attr_dense = torch.zeros([num_nodes ** 2, edge_attr.size(1)]).long()
                else:
                    edge_attr_dense = to_dense_adj(edge_idx, edge_attr=edge_attr, max_num_nodes=num_nodes).squeeze(0).view(-1, edge_attr.shape[-1])
            g.edata['feat'] = edge_attr_dense
            # else:
            #     g.edata['feat'] = torch.zeros([num_nodes ** 2, 1]).long() # ?

            preprocessed_graphs.append(g)
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
