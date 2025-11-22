from train import train_model
from utils.config import load_config
from utils.log import TrainingLogger
from copy import deepcopy

import shutil
import os
import numpy as np
import torch
import gc
import json

def run_search(model_name, dataset_name, search_dir: str, max_runs=30):

    config = load_config("configs/base.yaml", f"configs/{model_name}.yaml")
    search_dir = os.path.abspath(search_dir)

    create_search_dir(search_dir=search_dir)

    config["logging"]["log_dir"] = search_dir
    config["model"]["epochs"] = 10

    top_runs = []  
    print(f"Starting hyperparameter search ({max_runs} runs)...")

    for i in range(max_runs):

        if model_name == "fully_connected_net":
            hp_config = fully_connected_net_config(config=config)

        version_dir = train_model(
            model_name=model_name, 
            dataset_name=dataset_name, 
            config=hp_config,
            return_log_dir=True
            )

        update_leaderboard(top_runs=top_runs, version_dir=version_dir)

        # free memory 
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_leaderboard(top_runs=top_runs, save_dir=search_dir)

def fully_connected_net_config(config):

    # random sampling
    hp_config = deepcopy(config)
    hp_config["model"]["learning_rate"] = 10 ** np.random.uniform(-4, -2)
    hidden_dim = int(np.random.choice([32, 64, 128, 256]))
    n_layers = int(np.random.choice([2, 3, 4]))
    hp_config["model"]["hidden_layers"] = [hidden_dim for i in range(n_layers)]
    hp_config["model"]["batch_normalization"] = bool(np.random.choice([True, False]))
    hp_config["dataset"]["batch_size"] = int(np.random.choice([32, 64]))

    return hp_config

def update_leaderboard(top_runs, version_dir):
    meta_path = os.path.join(version_dir, "meta.json")
    if not os.path.exists(meta_path):
        print(f"WARNING: meta.json not found at {version_dir}")
        return

    with open(meta_path, "r") as f:
        meta = json.load(f)

    val_acc = meta.get("metrics", {}).get("accuracy/val", None)
    n_params = meta.get("metrics", {}).get("parameters", None)


    if val_acc is None:
        print(f"WARNING: No val_accuracy for {version_dir}")
        return

    version = version_dir.split("_")[-1]
    top_runs.append({
        "version": version,
        "val_acc": val_acc,
        "parameters": n_params
    })

    top_runs.sort(key=lambda x: x["val_acc"], reverse=True)
    if len(top_runs) > 3:
        del top_runs[3:]


def save_leaderboard(top_runs, save_dir):
    result_path = os.path.join(save_dir, "search_results.json")
    with open(result_path, "w") as f:
        json.dump(top_runs, f, indent=4)


def create_search_dir(search_dir):

    if os.path.exists(search_dir):
        if os.listdir(search_dir):

            reply = input(
                f"Directory '{search_dir}' is NOT empty. Delete it? [y/N]: "
            )
            if reply.lower() != 'y':
                return

            print(f"Clearing existing search directory")
            shutil.rmtree(search_dir)

    os.makedirs(search_dir, exist_ok=True)



if __name__ == "__main__":

    model = "fully_connected_net"
    dataset = "s2pt" 
    search_dir = "search_runs"

    run_search(
        model_name=model,
        dataset_name=dataset,
        search_dir=search_dir
        )