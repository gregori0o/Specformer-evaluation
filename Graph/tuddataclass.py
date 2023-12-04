import dgl
from dgl.data import TUDataset
from dgl.data.utils import load_graphs, save_graphs, Subset
import os
import shutil
import torch
from torch_geometric.utils import to_dense_adj
from enum import Enum
import json


class DatasetName(Enum):
    # ZINC = "ZINC_test"
    DD = "DD"
    NCI1 = "NCI1"
    PROTEINS = "PROTEINS_full"
    ENZYMES = "ENZYMES"
    IMDB_BINARY = "IMDB-BINARY"
    IMDB_MULTI = "IMDB-MULTI"
    REDDIT_BINARY = "REDDIT-BINARY"
    REDDIT_MULTI = "REDDIT-MULTI-5K"
    COLLAB = "COLLAB"

    def str_to_dataset(dataset_name: str):
        # if dataset_name == "ZINC":
        #     return DatasetName.ZINC
        if dataset_name == "DD":
            return DatasetName.DD
        elif dataset_name == "NCI1":
            return DatasetName.NCI1
        elif dataset_name == "PROTEINS":
            return DatasetName.PROTEINS
        elif dataset_name == "ENZYMES":
            return DatasetName.ENZYMES
        elif dataset_name == "IMDB-BINARY":
            return DatasetName.IMDB_BINARY
        elif dataset_name == "IMDB-MULTI":
            return DatasetName.IMDB_MULTI
        elif dataset_name == "REDDIT-BINARY":
            return DatasetName.REDDIT_BINARY
        elif dataset_name == "REDDIT-MULTI":
            return DatasetName.REDDIT_MULTI
        elif dataset_name == "COLLAB":
            return DatasetName.COLLAB
        else:
            raise ValueError(f"Dataset {dataset_name} not found")
    
    def list():
        return ["DD", "NCI1", "PROTEINS", "ENZYMES", "IMDB-BINARY", "IMDB-MULTI", "REDDIT-BINARY", "REDDIT-MULTI", "COLLAB"]


def load_indexes(dataset_name: DatasetName):
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


class TUDatasetPrep(object):
    def __init__(self, dataset_name: DatasetName):
        self.dataset_name = dataset_name.value
        raw_data_dir = os.path.join('/net/tscratch/people/plgglegeza', 'data', 'datasets', self.dataset_name, 'raw')
        prep_data_dir = os.path.join('/net/tscratch/people/plgglegeza', 'data', 'datasets', self.dataset_name, 'prep')
        
        if os.path.exists(prep_data_dir):
            self.graphs, label_dict = load_graphs(prep_data_dir)
            self.labels = label_dict['labels']
        else:
            if not os.path.exists(raw_data_dir):
                os.makedirs(raw_data_dir)
            dataset = TUDataset(self.dataset_name, raw_dir=raw_data_dir)
            shutil.rmtree(raw_data_dir)
            self.graphs, self.labels = map(list, zip(*dataset)) # check if this works
            self.labels = torch.tensor(self.labels).long()
            self.graphs = self._preprocess_graphs(self.graphs)
            save_graphs(prep_data_dir, self.graphs, labels={'labels': self.labels})

        self.num_node_labels = max([graph.ndata['feat'].max().item() for graph in self.graphs]) + 1
        self.num_edge_labels = max([graph.edata['feat'].max().item() for graph in self.graphs]) + 1
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

            if graph.ndata.get('node_labels') is not None:
                g.ndata['feat'] = graph.ndata['node_labels'].long()
            else:
                g.ndata['feat'] = torch.zeros([num_nodes, 1]).long()

            if graph.edata.get('edge_labels') is not None:
                edge_idx = torch.stack([src, dst], dim=0)
                edge_attr = graph.edata['edge_labels'].long() + 1  # for padding

                if len(edge_attr.shape) == 1:
                    edge_attr_dense = to_dense_adj(edge_idx, edge_attr=edge_attr.unsqueeze(-1)).squeeze(0).squeeze(-1).view(-1)
                else:
                    if edge_attr.size(0) == 0:   # for graphs without edge
                        edge_attr_dense = torch.zeros([num_nodes ** 2, edge_attr.size(1)]).long()
                    else:
                        edge_attr_dense = to_dense_adj(edge_idx, edge_attr=edge_attr, max_num_nodes=num_nodes).squeeze(0).view(-1, edge_attr.shape[-1])
                g.edata['feat'] = edge_attr_dense
            else:
                g.edata['feat'] = torch.zeros([num_nodes ** 2, 1]).long() # ?

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
