from dgl_main import main_worker
import datetime
import time
from tuddataclass import DatasetName, TUDatasetPrep, load_indexes
from ogbgdataclass import OGBGDatasetPrep
from iamdataclass import IAMGDatasetPrep
from sklearn.model_selection import train_test_split
import itertools
import torch
import torch.nn.functional as F
from get_dataset import TUEvaluator
import numpy as np
import random
import os
import json
import gc
import time
import argparse
from time_measure import time_measure
from torch.utils.data import DataLoader
from get_dataset import collate_dgl


batch_sizes = {
    "small": {
        "NCI1": 64,
        "PROTEINS_full": 32,
        "ENZYMES": 64,
        "IMDB-BINARY": 64,
        "IMDB-MULTI": 64,
        "COLLAB": 8,
        "ogbg-molhiv": 32,
        "Mutagenicity": 32,
    },
    "medium": {
        "NCI1": 32,
        "ENZYMES": 32,
        "IMDB-BINARY": 8,
        "IMDB-MULTI": 8,
        "COLLAB": 8,
        "ogbg-molhiv": 16,
        "Mutagenicity": 8,
    },
    "large": {
        "NCI1": 8,
        "ENZYMES": 8,
        "IMDB-BINARY": 8,
        "IMDB-MULTI": 8,
        "COLLAB": 8,
        "ogbg-molhiv": 16,
        "Mutagenicity": 2,
    }
}


class Config:
    def __init__(self, d_list=None):
        if d_list is not None:
            for d in d_list:
                for key, value in d.items():
                    setattr(self, key, value)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False


def get_all_params(param_grid):
    return [
        dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())
    ]


def fair_evaluation(dataset_name, model_name="small"):
    run_config = {
        "seed": 0,
        "cuda": 0,
        "dataset": dataset_name,
        "project_name": time.strftime('%Y_%b_%d_at_%Hh%Mm%Ss'),
        "log_step": 1,
        "tuning": False,
        "r_evaluation": 3,
    }
    if model_name == "small":
        model_config = {
            "model": "small",
            "nlayer": 8,
            "nheads": 8,
            "hidden_dim": 160,
            "trans_dropout": 0.1,
            "feat_dropout": 0.1,
            "adj_dropout": 0.3,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "epochs": 100,
            "warm_up_epoch": 5,
            "batch_size": 8, #molhiv and mutagenicity  and proteins=32, NCI1,Enzymes,IMDB-B-M=64, COLLAB=8, and rest=2 
        }
    elif model_name == "medium":
        model_config = {
            "model": "medium",
            "nlayer": 8,
            "nheads": 8,
            "hidden_dim": 272,
            "trans_dropout": 0.3,
            "feat_dropout": 0.1,
            "adj_dropout": 0.1,
            "lr": 5e-4,
            "weight_decay": 5e-3,
            "epochs": 100,
            "warm_up_epoch": 5,
            "batch_size": 16, # enzymes, nci1 - 32; IMDB, COLLAB, mutagenicity - 8, molhiv = 16, Proteins = 8
        }
    elif model_name == "large":
        model_config = {
            "model": "large",
            "nlayer": 10,
            "nheads": 16,
            "hidden_dim": 400,
            "trans_dropout": 0.05,
            "feat_dropout": 0.05,
            "adj_dropout": 0.05,
            "lr": 2e-4,
            "weight_decay": 0.0,
            "epochs": 100,
            "warm_up_epoch": 10,
            "batch_size": 8, # enzymes, NCI1, COLLAB, IMDB B = 8; Mutagenicity = 4
        }
    else:
        raise NotImplementedError("Model name not found!")


    bs = batch_sizes[model_name].get(dataset_name)
    if bs is None:
        return
    model_config["batch_size"] = bs

    print(model_config)

    config = Config([run_config, model_config])

    seed_everything(config.seed)

    if dataset_name.startswith("ogbg"):
        dataset = OGBGDatasetPrep(dataset_name, model_name)
    elif dataset_name in ["Web", "Mutagenicity"]:
        dataset = IAMGDatasetPrep(dataset_name, model_name)
    else:
        dataset_name = DatasetName.str_to_dataset(dataset_name)
        dataset = TUDatasetPrep(dataset_name, model_name)
    indexes = load_indexes(dataset_name)

    data_info = {
        'num_class': dataset.num_class,
        'loss_fn': F.binary_cross_entropy_with_logits,
        'metric': 'acc',
        'metric_mode': 'max',
        'evaluator': TUEvaluator(),
        'num_node_labels': dataset.num_node_labels,
        'num_edge_labels': dataset.num_edge_labels,
    }

    
    for i, fold in enumerate(indexes):
        print(f"FOLD {i}")
        
        data_info['test_dataset'] = dataset[torch.tensor(fold["test"], dtype=torch.long)]
        train_idx, val_idx = train_test_split(fold["train"], test_size=0.1)
        data_info['train_dataset'] = dataset[torch.tensor(train_idx, dtype=torch.long)]
        data_info['valid_dataset'] = dataset[torch.tensor(val_idx, dtype=torch.long)]

        # model, device = main_worker(config, data_info)

        model, device = time_measure(main_worker, f"spec_{model_name}", run_config["dataset"], "training")(config, data_info)

        eval_idx = list(range(128))
        dataset.upload_indexes(eval_idx, eval_idx, eval_idx)
        half_batch_size = max(config.batch_size // 2, 1)
        test = dataset[torch.tensor(eval_idx, dtype=torch.long)]
        eval_loader = DataLoader(test,  batch_size = half_batch_size, num_workers=4, collate_fn=collate_dgl, shuffle = False)

        predictions = time_measure(get_prediction, f"spec_{model_name}", run_config["dataset"], "evaluation")(
            model, device, eval_loader
        )
        break

        
def get_prediction(model, device, dataloader):
    model.eval()
    y_pred = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            e, u, g, length, y = data
            e, u, g, length, y = e.to(device), u.to(device), g.to(device), length.to(device), y.to(device)

            logits = model(e, u, g, length)
            if model.nclass > 1:
                score = logits.detach()
                prediction = score.argmax(dim=1)
            else:
                score = prediction = logits.detach()

            y_pred.append(prediction.cpu())
        
    y_pred = torch.cat(y_pred, dim=0).numpy()
    
    return y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser("run_experiment")
    parser.add_argument("--dataset", help="Dataset name", type=str, default=DatasetName.ENZYMES.value)
    args = parser.parse_args()
    dataset_name = args.dataset
    print(f"Running experiment on {dataset_name}")
    fair_evaluation(dataset_name, model_name="small")
    fair_evaluation(dataset_name, model_name="medium")
    fair_evaluation(dataset_name, model_name="large")
