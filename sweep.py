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

def run_search(model_name, dataset_name, search_dir: str, max_runs=200):

    # to log failed runs 
    status_log = os.path.join(search_dir, "status_log.txt")

    config = load_config("configs/base.yaml", f"configs/{model_name}.yaml")
    search_dir = os.path.abspath(search_dir)

    create_search_dir(search_dir=search_dir)

    config["logging"]["log_dir"] = search_dir
    config["trainer"]["epochs"] = 10

    top_runs = []  
    print(f"Starting hyperparameter search ({max_runs} runs)...")

    for i in range(max_runs):

        if model_name == "fully_connected_net":
            hp_config = fully_connected_net_config(config=config)
        elif model_name == "deep_sets":
            hp_config = deep_sets_config(config=config)

        elif model_name == "graph_net":
            hp_config = graph_net_config(config=config)
        

        print(hp_config)

        try:

            version_dir = train_model(
                model_name=model_name, 
                dataset_name=dataset_name, 
                config=hp_config,
                return_log_dir=True
                )

            update_leaderboard(top_runs=top_runs, version_dir=version_dir)
        
        except Exception as e:
            print(f"[Run {i}/{max_runs}] Configuration failed: {e}")

            with open(status_log, "a") as f:
                f.write(f"Run {i} FAILED\n")
                f.write(f"Error: {e}\n")
                f.write("Hyperparameters:\n")
                f.write(f"{hp_config}\n")
                f.write("-" * 80 + "\n\n")

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

def deep_sets_config(config):

    # random sampling
    hp_config = deepcopy(config)

    phi_dim = int(np.random.choice([128, 256, 512, 1024]))
    phi_n_layers = int(np.random.choice([1, 2, 3, 4]))
    hp_config["model"]["phi_layers"] = [phi_dim for i in range(phi_n_layers)]

    rho_dim = int(np.random.choice([128, 256, 512, 1024]))
    rho_n_layers = int(np.random.choice([1, 2, 3]))
    hp_config["model"]["rho_layers"] = [rho_dim for i in range(rho_n_layers)]
    

    # hp_config["model"]["layer_norm"] = bool(np.random.choice([True, False]))
    hp_config["model"]["activation"] = str(np.random.choice(["gelu", "silu"]))
    hp_config["model"]["residual_block"] = bool(np.random.choice([True, False]))
    hp_config["trainer"]["learning_rate"] = 10 ** np.random.uniform(-4, -2)    
    # hp_config["trainer"]["optimizer"] = str(np.random.choice(["adam", "adamw"]))

    hp_config["dataset"]["batch_size"] = int(np.random.choice([16, 32, 64]))

    return hp_config

def graph_net_config(config):

    hp_config = deepcopy(config)

    hp_config["model"]["hidden_dim"] = int(np.random.choice([64, 128, 256]))
    hp_config["model"]["activation"] = str(np.random.choice(["gelu", "relu", "tanh"]))
    hp_config["model"]["use_gat"] = bool(np.random.choice([True, False]))
    hp_config["model"]["gat_heads"] = int(np.random.choice([4, 8]))
    hp_config["model"]["sag_pool"] = bool(np.random.choice([True, False]))
    hp_config["model"]["pool_ratio"] = float(np.random.choice([0.3, 0.4, 0.5]))
    hp_config["model"]["local_pooling"] = str(np.random.choice(["add", "mean", "max"]))
    hp_config["model"]["global_pooling"] = str(np.random.choice(["add", "mean", "max"]))
    hp_config["model"]["deepchem_style"] = bool(np.random.choice([True, False]))
    hp_config["model"]["input_dim"] = int(np.random.choice([1, 4]))

    hp_config["dataset"]["use_weights"] = bool(np.random.choice([True, False]))
    hp_config["dataset"]["batch_size"] = int(np.random.choice([16, 32, 64]))

    hp_config["trainer"]["learning_rate"] = 10 ** np.random.uniform(-4, -2)    
    hp_config["trainer"]["optimizer"] = str(np.random.choice(["adam", "adamw"]))


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

    model = "graph_net"
    dataset = "s2pg" 
    search_dir = "search_runs"

    run_search(
        model_name=model,
        dataset_name=dataset,
        search_dir=search_dir
        )