from dgl_main import main_worker
import datetime
import time
from tuddataclass import DatasetName, TUDatasetPrep, load_indexes
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
            "batch_size": 64,
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
            "epochs": 101,
            "warm_up_epoch": 5,
            "batch_size": 64,
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
            "epochs": 151,
            "warm_up_epoch": 10,
            "batch_size": 64,
        }
    else:
        raise NotImplementedError("Model name not found!")

    tuning_config = {
        "nlayer": [4, 8],
        "nheads": [8, 16],
        "hidden_dim": [100, 160],
    }

    config = Config([run_config, model_config])

    seed_everything(config.seed)

    dataset_name = DatasetName.str_to_dataset(dataset_name)
    dataset = TUDatasetPrep(dataset_name)
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

    scores = {
        "accuracy": [],
        "f1": [],
        "macro f1": [],
        "precision": [],
        "recall": [],
        "roc": [],
    }
    evaluation_result = {}
    evaluation_result["run_config"] = run_config.copy()
    evaluation_result["model_config"] = model_config.copy()
    evaluation_result["tuning_config"] = tuning_config.copy() if config.tuning else None
    evaluation_result["folds"] = {}

    for i, fold in enumerate(indexes):
        print(f"FOLD {i}")
        evaluation_result["folds"][i] = {}
        evaluation_result["folds"][i]["tuning_params"] = None

        ## get best model for train data
        if config.tuning:
            best_acc = 0
            best_params = {}
            train_idx, val_idx = train_test_split(fold["train"], test_size=0.2)
            data_info['train_dataset'] = dataset[torch.tensor(train_idx, dtype=torch.long)]
            data_info['valid_dataset'] = dataset[torch.tensor(val_idx, dtype=torch.long)]
            data_info['test_dataset'] = dataset[torch.tensor(fold["test"], dtype=torch.long)]
            params = get_all_params(tuning_config)
            for param in params:
                for key, value in param.items():
                    setattr(config, key, value)

                acc, _ = main_worker(config, data_info)
                if acc > best_acc:
                    best_acc = acc
                    best_params = param.copy()

            for key, value in best_params.items():
                setattr(config, key, value)
            
            evaluation_result["folds"][i]["tuning_params"] = best_params.copy()
        

        # evaluate model R times
        scores_r = {
            "accuracy": 0,
            "f1": 0,
            "macro f1": 0,
            "precision": 0,
            "recall": 0,
            "roc": 0,
        }
        for _ in range(config.r_evaluation):
            data_info['test_dataset'] = dataset[torch.tensor(fold["test"], dtype=torch.long)]
            train_idx, val_idx = train_test_split(fold["train"], test_size=0.1)
            data_info['train_dataset'] = dataset[torch.tensor(train_idx, dtype=torch.long)]
            data_info['valid_dataset'] = dataset[torch.tensor(val_idx, dtype=torch.long)]

            acc, full_eval = main_worker(config, data_info)
            for key in scores_r.keys():
                scores_r[key] += full_eval[key]

        for key in scores_r.keys():
            scores_r[key] /= config.r_evaluation
        print(f"MEAN SCORE = {scores_r} in FOLD {i}")
        for key in scores_r.keys():
            scores[key].append(scores_r[key])
        evaluation_result["folds"][i]["score"] = scores_r

    # evaluate model
    summ = {}
    for key in scores.keys():
        summ[key] = {}
        summ[key]["mean"] = np.mean(scores[key])
        summ[key]["std"] = np.std(scores[key])

    evaluation_result["evaluation_result"] = {
        "scores_list": scores,
        "summary_scores": summ,
    }
    print(f"Evaluation of model on {dataset_name}")
    print(f"Scores: {scores}")
    print(f"Summary: {summ}")

    dir_path = f"results/fair_evaluation/{config.dataset}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(f"{dir_path}/{config.project_name}.json", "w") as f:
        json.dump(evaluation_result, f)


# def run_model(dataset_name, model_name="small", batch_size=64):
#     run_config = {
#         "seed": 0,
#         "cuda": 0,
#         "dataset": dataset_name,
#         "project_name": datetime.datetime.now().strftime('%m-%d-%X'),
#         "log_step": 5
#     }
#     if model_name == "small":
#         model_config = {
#             "model": "small",
#             "nlayer": 8,
#             "nheads": 4,
#             "hidden_dim": 80,
#             "trans_dropout": 0.1,
#             "feat_dropout": 0.1,
#             "adj_dropout": 0.3,
#             "lr": 1e-4,
#             "weight_decay": 1e-4,
#             "epochs": 51,
#             "warm_up_epoch": 5,
#             "batch_size": batch_size,
#         }
#     elif model_name == "medium":
#         model_config = {
#             "model": "medium",
#             "nlayer": 8,
#             "nheads": 8,
#             "hidden_dim": 272,
#             "trans_dropout": 0.3,
#             "feat_dropout": 0.1,
#             "adj_dropout": 0.1,
#             "lr": 5e-4,
#             "weight_decay": 5e-3,
#             "epochs": 101,
#             "warm_up_epoch": 5,
#             "batch_size": batch_size,
#         }
#     elif model_name == "large":
#         model_config = {
#             "model": "large",
#             "nlayer": 10,
#             "nheads": 16,
#             "hidden_dim": 400,
#             "trans_dropout": 0.05,
#             "feat_dropout": 0.05,
#             "adj_dropout": 0.05,
#             "lr": 2e-4,
#             "weight_decay": 0.0,
#             "epochs": 151,
#             "warm_up_epoch": 10,
#             "batch_size": batch_size,
#         }
#     else:
#         raise NotImplementedError("Model name not found!")

#     config = Config([run_config, model_config])

#     seed_everything(config.seed)

#     dataset_name = DatasetName.str_to_dataset(dataset_name)
#     dataset = TUDatasetPrep(dataset_name)
#     indexes = load_indexes(dataset_name)[0]
#     test_idx = indexes['test']
#     train_idx, val_idx = train_test_split(indexes["train"], test_size=0.2)
    
#     # print("Datase loaded", torch.cuda.memory_allocated())

#     data_info = {
#         'num_class': dataset.num_class,
#         'loss_fn': F.binary_cross_entropy_with_logits,
#         'metric': 'acc',
#         'metric_mode': 'max',
#         'evaluator': TUEvaluator(),
#         'train_dataset': dataset[torch.tensor(train_idx, dtype = torch.long)],
#         'valid_dataset': dataset[torch.tensor(val_idx, dtype = torch.long)],
#         'test_dataset': dataset[torch.tensor(test_idx, dtype = torch.long)],
#         'num_node_labels': dataset.num_node_labels,
#         'num_edge_labels': dataset.num_edge_labels,
#     }
#     del dataset
#     torch.cuda.empty_cache()
#     gc.collect()

#     # print("Before start main worker", torch.cuda.memory_allocated())

#     acc, f1 = main_worker(config, data_info)

#     print(f"Simple evaluation of model on {config.dataset}")
#     print(f"ACC: {acc}")
#     print(f"F1: {f1}")

#     evaluation_result = {}
#     evaluation_result["run_config"] = run_config.copy()
#     evaluation_result["model_config"] = model_config.copy()
#     evaluation_result["evaluation_result"] = {
#         "score": acc,
#         "f1": f1,
#     }

#     dir_path = f"results/simple_run/{config.dataset}"
#     if not os.path.exists(dir_path):
#         os.makedirs(dir_path)

#     with open(f"{dir_path}/{config.project_name}.json", "w") as f:
#         json.dump(evaluation_result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("run_experiment")
    parser.add_argument("--dataset", help="Dataset name", type=str, choices=DatasetName.list(), default=DatasetName.ENZYMES.value)
    args = parser.parse_args()
    dataset_name = args.dataset
    print(f"Running experiment on {dataset_name}")
    fair_evaluation(dataset_name)
