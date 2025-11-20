import os
import json

class TrainingLogger():
    def __init__(self, model_name, dataset_name, log_dir):

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.save_dir = log_dir
        self.version = self._calculate_version()
        self._setup_log()


    def _calculate_version(self):
        version = 0
        while os.path.exists(os.path.join(self.save_dir, f"version_{version}")):
            version += 1
        return str(version) 

    def get_version(self):
        return self.version

    def _setup_log(self):
        
        version_dir = os.path.join(self.save_dir, f"version_{self.version}")
        os.makedirs(version_dir)

        metainfo = {
            "dataset": self.dataset_name,
            "model": self.model_name
        }

        with open(os.path.join(version_dir, "meta.json"), "w") as f:
            json.dump(metainfo, f, indent=4)


    def log_metric(self, name, value):

        version_dir = os.path.join(self.save_dir, f"version_{self.version}")
        meta_path = os.path.join(version_dir, "meta.json")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        if "metrics" not in meta:
            meta["metrics"] = {}

        meta["metrics"][name] = value

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

        print(f"Saved metric '{name}': {value}")

