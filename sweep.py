from train import train_model
from utils.config import load_config

import shutil
import os

def run_search(search_dir: str, max_runs=30):

    model = "fully_connected_net"
    dataset = "s2pt" 
    hp_config = load_config("configs/base.yaml", f"configs/{model}.yaml")

    abs_search_dir = os.path.abspath(search_dir)

    if os.path.exists(abs_search_dir):
        if os.listdir(abs_search_dir):

            reply = input(
                f"Directory '{abs_search_dir}' is NOT empty. Delete it? [y/N]: "
            )
            if reply.lower() != 'y':
                return

            print(f"Clearing existing search directory")
            shutil.rmtree(abs_search_dir)

    os.makedirs(abs_search_dir, exist_ok=True)









    # for i in range(max_runs):

    #     train_model(
    #         model_name=model, 
    #         dataset_name=dataset, 
    #         config=hp_config
    #         )



    # 10^3 to 10^6


if __name__ == "__main__":
    
    search_dir = "search_runs"
    run_search(search_dir=search_dir)