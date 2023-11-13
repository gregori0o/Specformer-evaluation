from dgl_main import main_worker
import datetime
import time
from neurodataclass import NeuroDatasetName, NeuroDatasetPrep, load_neuro_indexes
from sklearn.model_selection import train_test_split
import itertools
import torch
import torch.nn.functional as F
from get_dataset import TUEvaluator
import numpy as np
import random
import os
import json


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


def fair_evaluation(dataset_name):
    run_config = {
        "seed": 0,
        "cuda": 0,
        "dataset": dataset_name,
        "project_name": time.strftime('%Y_%b_%d_at_%Hh%Mm%Ss'),
        "tuning": False,
        "r_evaluation": 1,
    }
    model_config = {
        "model": "small",
        "nlayer": 4,
        "nheads": 8,
        "hidden_dim": 160,
        "trans_dropout": 0.1,
        "feat_dropout": 0.05,
        "adj_dropout": 0.0,
        "lr": 1e-3,
        "weight_decay": 5e-4,
        "epochs": 1,
        "warm_up_epoch": 50,
        "batch_size": 32,
    }
    tuning_config = {
        "nlayer": [4, 8],
        "nheads": [8, 16],
        "hidden_dim": [100, 160],
    }

    config = Config([run_config, model_config])

    seed_everything(config.seed)

    dataset_name = NeuroDatasetName.str_to_dataset(dataset_name)
    dataset = NeuroDatasetPrep(dataset_name)
    indexes = load_neuro_indexes(dataset_name)

    data_info = {
        'num_class': dataset.num_class,
        'loss_fn': F.binary_cross_entropy_with_logits,
        # 'loss_fn': F.cross_entropy,
        'metric': 'acc',
        'metric_mode': 'max',
        'evaluator': TUEvaluator(),
    }

    scores = []
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

                acc = main_worker(config, data_info)
                if acc > best_acc:
                    best_acc = acc
                    best_params = param.copy()

            for key, value in best_params.items():
                setattr(config, key, value)
            
            evaluation_result["folds"][i]["tuning_params"] = best_params.copy()
        

        # evaluate model R times
        data_info['test_dataset'] = dataset[torch.tensor(fold["test"], dtype=torch.long)]
        scores_r_list = []
        for _ in range(config.r_evaluation):
            train_idx, val_idx = train_test_split(fold["train"], test_size=0.2)
            data_info['train_dataset'] = dataset[torch.tensor(train_idx, dtype=torch.long)]
            data_info['valid_dataset'] = dataset[torch.tensor(val_idx, dtype=torch.long)]

            acc = main_worker(config, data_info)
            scores_r_list.append(acc)

        scores_r = np.mean(scores_r_list)
        print(f"MEAN SCORE = {scores_r} in FOLD {i}")
        scores.append(scores_r)
        evaluation_result["folds"][i]["scores_list"] = scores_r_list
        evaluation_result["folds"][i]["score"] = scores_r

    # evaluate model
    mean = np.mean(scores)
    std = np.std(scores)

    evaluation_result["evaluation_result"] = {
        "scores_list": scores,
        "score": mean,
        "score_std": std,
    }

    print(f"Evaluation of model on {config.dataset}")
    print(f"Mean ACC: {mean}")
    print(f"STD ACC: {std}")

    dir_path = f"results/fair_evaluation/{config.dataset}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(f"{dir_path}/{config.project_name}.json", "w") as f:
        json.dump(evaluation_result, f)


def run_model(dataset_name):
    run_config = {
        "seed": 0,
        "cuda": 0,
        "dataset": dataset_name,
        "project_name": datetime.datetime.now().strftime('%m-%d-%X'),
    }
    model_config = {
        "model": "small",
        "nlayer": 4,
        "nheads": 8,
        "hidden_dim": 160,
        "trans_dropout": 0.1,
        "feat_dropout": 0.05,
        "adj_dropout": 0.0,
        "lr": 1e-3,
        "weight_decay": 5e-4,
        "epochs": 1000,
        "warm_up_epoch": 50,
        "batch_size": 32,
    }

    config = Config([run_config, model_config])

    seed_everything(config.seed)

    dataset_name = NeuroDatasetName.str_to_dataset(dataset_name)
    dataset = NeuroDatasetPrep(dataset_name)
    indexes = load_neuro_indexes(dataset_name)[0]
    test_idx = indexes['test']
    train_idx, val_idx = train_test_split(indexes["train"], test_size=0.2)

    data_info = {
        'num_class': dataset.num_class,
        'loss_fn': F.binary_cross_entropy_with_logits,
        # 'loss_fn': F.cross_entropy,
        'metric': 'acc',
        'metric_mode': 'max',
        'evaluator': TUEvaluator(),
        'train_dataset': dataset[torch.tensor(train_idx, dtype = torch.long)],
        'valid_dataset': dataset[torch.tensor(val_idx, dtype = torch.long)],
        'test_dataset': dataset[torch.tensor(test_idx, dtype = torch.long)],
    }
    del dataset

    acc = main_worker(config, data_info)

    print(f"Simple evaluation of model on {config.dataset}")
    print(f"ACC: {acc}")

    evaluation_result = {}
    evaluation_result["run_config"] = run_config.copy()
    evaluation_result["model_config"] = model_config.copy()
    evaluation_result["evaluation_result"] = {
        "score": acc,
    }

    dir_path = f"results/simple_run/{config.dataset}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(f"{dir_path}/{config.project_name}.json", "w") as f:
        json.dump(evaluation_result, f)


if __name__ == '__main__':
    dataset_name = "GENDER"
    run_model(dataset_name)
    # fair_evaluation(dataset_name)
    