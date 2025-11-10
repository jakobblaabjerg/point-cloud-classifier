
import os 
import numpy as np
import h5py
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataModule():

    def __init__(self,
                 data_dir,
                 particles = ["proton", "piM"],
                 create_dataset = False,
                 feature_scaling = True
                 ):
        
        self.particles = particles
        self.data_dir = data_dir
        self.create_dataset = create_dataset
        self.data_split = (0.6, 0.2, 0.2) # (train, val, test)
        self.feature_scaling = feature_scaling

        self.datasets = {"train": [],
            "val": [],
            "test": []
            }

    def _find_files(self, particle):
        all_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(self.data_dir)
            for file in files
            if file.endswith((".h5", ".hdf5")) and particle in file
        ]
        print(f"Found {len(all_files)} files for {particle}")
        return all_files


    def _load_h5py_file(self, filepath):
        """Load and return the contents of a single HDF5 file."""
        raise NotImplementedError("Implement HDF5 file loading logic here.")



class Step2PointTabular(DataModule):

    def __init__(self, data_dir, **kwargs):
        super().__init__(data_dir=data_dir, **kwargs)        
        self.name = "S2PT"

        if self.create_dataset:
            print("Creating Step2PointTabular (S2PT) dataset")
            self._create_dataset()
        else:
            self._load_dataset()


    def _preprocess_data(self, data, particle):
        
        df = pd.DataFrame({
            "event_id": data["event_id"],
            "energy": data["energy"],
            "x": data["position"][:, 0],
            "y": data["position"][:, 1],
            "z": data["position"][:, 2],
            "subdetector": data["subdetector"]
        })

        # convert from bytes to strings
        df["subdetector"] = df["subdetector"].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )

        # classify each value into one of three higher-level detector categories
        df["detector"] = df["subdetector"].apply(
            lambda x: "HCal" if "HCal" in x else ("ECal" if "ECal" in x else "Other")
        )

        if len(df[df["detector"] == "Other"]):
            print("Unkown detector part encountered")

        invalid_count = (df["detector"] == "Other").sum()
        if invalid_count > 0:
            print(f"Unknown detector part encountered. Count: {invalid_count}")
            # exclude invalid rows
            df = df[df["detector"] != "Other"]

        # split data into hcal and ecal 
        df_grouped = df.groupby(["event_id", "detector"]).agg(
            energy=("energy", "sum"), 
            hits=("energy", "count")
            ).reset_index()

        df_hcal = (
            df_grouped[df_grouped["detector"] == "HCal"]
            .copy()
            .rename(columns={"energy": "energy_hcal", "hits": "hits_hcal"})
        )

        df_ecal = (
            df_grouped[df_grouped["detector"] == "ECal"]
            .copy()
            .rename(columns={"energy": "energy_ecal", "hits": "hits_ecal"})
        )

        # combine hcal and ecal data
        df_preprocessed = df_hcal.merge(df_ecal, on="event_id", how="outer")
        df_preprocessed["energy_hcal"] = df_preprocessed["energy_hcal"].fillna(0)
        df_preprocessed["energy_ecal"] = df_preprocessed["energy_ecal"].fillna(0)
        df_preprocessed["hits_hcal"] = df_preprocessed["hits_hcal"].fillna(0)
        df_preprocessed["hits_ecal"] = df_preprocessed["hits_ecal"].fillna(0)

        # compute features
        df_preprocessed["energy_total"] = df_preprocessed["energy_ecal"] + df_preprocessed["energy_hcal"]
        df_preprocessed["hits_total"] = df_preprocessed["hits_ecal"] + df_preprocessed["hits_hcal"]
        df_preprocessed["hits_hcal_frac"] = df_preprocessed["hits_hcal"] / df_preprocessed["hits_total"]
        df_preprocessed["energy_hcal_frac"] = df_preprocessed["energy_hcal"] / df_preprocessed["energy_total"]
        df_preprocessed["label"] = particle
        df_preprocessed["label"] = df_preprocessed["label"].map({"proton": 0, "piM": 1})
        df_preprocessed = df_preprocessed[["energy_total", "hits_total", "energy_hcal_frac", "hits_hcal_frac", "label"]]

        if df_preprocessed.isna().any().any():
            print("There are NaN values in the dataset!")
        else:
            print("No NaN values detected.")
        
        return df_preprocessed


    def _split_dataset(self, dataset):

        train_frac, val_frac, test_frac = self.data_split

        train_df, test_df = train_test_split(
            dataset, 
            test_size=test_frac, 
            stratify=dataset["label"], 
            random_state=42
        )

        train_df, val_df = train_test_split(
            train_df,
            test_size=val_frac/(train_frac+val_frac),
            stratify=train_df["label"],
            random_state=42
        )

        return train_df, val_df, test_df

    def _save_datasets(self):

        for split in self.datasets:
            data = self.datasets[split]
            print(f"Saving {split} dataset")
            filepath = os.path.join(self.data_dir, f"{self.name}_{split}.npz")
            np.savez(
                filepath,
                energy_total = data["energy_total"].to_numpy(),
                hits_total   = data["hits_total"].to_numpy(),
                energy_hcal_frac = data["energy_hcal_frac"].to_numpy(),
                hits_hcal_frac   = data["hits_hcal_frac"].to_numpy(),
                label       = data["label"].to_numpy(),
            )
        print("Finished saving data")

    def _load_dataset(self):
        for split in self.datasets:
            filepath = os.path.join(self.data_dir, f"{self.name}_{split}.npz")
            if not os.path.exists(filepath):
                print(f"File not found: {filepath}")
                continue

            print(f"Loading {split} dataset from {filepath}")
            data = np.load(filepath)
            df = pd.DataFrame({
                "energy_total": data["energy_total"],
                "hits_total": data["hits_total"],
                "energy_hcal_frac": data["energy_hcal_frac"],
                "hits_hcal_frac": data["hits_hcal_frac"],
                "label": data["label"]
            })

            self.datasets[split] = df

        print("Finished loading datasets")


    def _create_dataset(self):

        self.datasets = {"train": [],
        "val": [],
        "test": []
        }

        for particle in self.particles:
            files = self._find_files(particle)
        
            for f in files:
                print(f)
                data_raw = self._load_h5py_file(f)
                data_preprocessed  = self._preprocess_data(data_raw, particle)
                train_df, val_df, test_df = self._split_dataset(data_preprocessed) # file level split 
                self.datasets["train"].append(train_df)
                self.datasets["val"].append(val_df)
                self.datasets["test"].append(test_df)

        # concatenate each split
        for split in self.datasets:
            self.datasets[split] = pd.concat(self.datasets[split], ignore_index=True)

        if self.feature_scaling:
            self._scale_features()
        
        self._save_datasets()
        

    def _scale_features(self):

        feature_cols = [col for col in self.datasets["train"].columns if col != "label"]
        print("Scaling the following columns:", feature_cols)

        # extract features
        X_train = self.datasets["train"][feature_cols]
        X_val   = self.datasets["val"][feature_cols]
        X_test  = self.datasets["test"][feature_cols]

        # fit scaler on train
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scaler = scaler

        # transform val and test
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # replace original feature columns with scaled values
        for split, X_scaled in zip(["train", "val", "test"], [X_train_scaled, X_val_scaled, X_test_scaled]):
            df_scaled = self.datasets[split].copy()
            df_scaled[feature_cols] = X_scaled
            self.datasets[split] = df_scaled


    def _load_h5py_file(self, filepath):
       
        with h5py.File(filepath, "r") as f:

            metadata_group = f["metadata"]
            steps_group = f["steps"]

            subdetector_meta = metadata_group["subdetector_names"][:]
            subdetector = steps_group["subdetector"][:]
            subdetector_decoded = np.array([subdetector_meta[i] for i in subdetector])
            
            data = {
                "energy": steps_group["energy"][:],
                "event_id": steps_group["event_id"][:],
                "position": steps_group["position"][:],
                "subdetector": subdetector_decoded,
            }

        return data



class Step2PointPointCloud(DataModule):

    def __init__(self):
        pass

class Step2PointGraph(DataModule):
    
    def __init__(self):
        pass






    

