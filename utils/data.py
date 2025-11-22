
import os 
import numpy as np
import h5py
import pandas as pd
import torch
import joblib
import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

class DataModule():

    def __init__(self,
                 data_dir,
                 particles = ["proton", "piM"],
                 create_dataset = False,
                 feature_scaling = True,
                 batch_size = None,
                 ):
        
        self.particles = particles
        self.data_dir = data_dir
        self.create_dataset = create_dataset
        self.data_split = (0.6, 0.2, 0.2) # (train, val, test)
        self.feature_scaling = feature_scaling
        self.batch_size = batch_size
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
       
        with h5py.File(filepath, "r") as f:

            metadata_group = f["metadata"]
            steps_group = f["steps"]

            subdetector_meta = metadata_group["subdetector_names"][:]
            subdetector = steps_group["subdetector"][:]
            subdetector_decoded = subdetector_meta[subdetector]

            data = {
                "energy": steps_group["energy"][:],
                "event_id": steps_group["event_id"][:],
                "position": steps_group["position"][:],
                "time": steps_group["time"][:],
                "subdetector": subdetector_decoded,
            }

        return data

    def _create_dataset(self):

        self.datasets = {"train": [],
        "val": [],
        "test": []
        }

        for particle in self.particles:
            files = self._find_files(particle)
        
            for f in files:
                print(os.path.basename(f))
                data_raw = self._load_h5py_file(f)
                data_preprocessed  = self._preprocess_data(data_raw, particle)
                data_preprocessed["source_file"] = os.path.basename(f)
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
        
        # remove source file column after data is saved
        for split in self.datasets:
            self.datasets[split] = self.datasets[split].drop(columns=["source_file"])



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

    def _scale_features(self):

        feature_cols = [col for col in self.datasets["train"].columns if col not in ["label", "event_id", "source_file"]]
        print("Scaling the following columns:", feature_cols)

        # extract features
        X_train = self.datasets["train"][feature_cols]
        X_val   = self.datasets["val"][feature_cols]
        X_test  = self.datasets["test"][feature_cols]

        # fit scaler on train
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scaler = scaler
        joblib.dump(scaler, os.path.join(self.data_dir, f"{self.name}_scaler.pkl"))        

        # transform val and test
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # replace original feature columns with scaled values
        for split, X_scaled in zip(["train", "val", "test"], [X_train_scaled, X_val_scaled, X_test_scaled]):
            df_scaled = self.datasets[split].copy()
            df_scaled[feature_cols] = X_scaled
            self.datasets[split] = df_scaled


class Step2PointTabular(DataModule):

    def __init__(self, 
                 data_dir,
                 convert_to_tensor = False, 
                 **kwargs):
        super().__init__(data_dir=data_dir, **kwargs)        
        self.name = "S2PT"
        self.convert_to_tensor = convert_to_tensor

        if self.create_dataset:
            print("Creating Step2PointTabular (S2PT) dataset")
            self._create_dataset()
        else:
            self._load_dataset()


    def _preprocess_data(self, data, particle):
        
        df = pd.DataFrame({
            "event_id": data["event_id"],
            "energy": data["energy"],
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
                raise FileNotFoundError(f"Required file is missing: {filepath}")

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

    def _convert_to_tensor(self, dataset, split):
        X = dataset.drop(columns=["label"]).to_numpy()
        y = np.array(dataset["label"])
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
        return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size, shuffle=(split=="train"))


    def get_train_data(self):
        split ="train"
        if self.convert_to_tensor:
            return self._convert_to_tensor(self.datasets[split], split)
        else:
            return self.datasets[split]

    def get_val_data(self):
        split="val"
        if self.convert_to_tensor:
            return self._convert_to_tensor(self.datasets[split], split)
        else:
            return self.datasets[split]

    def get_test_data(self):
        split="test"
        if self.convert_to_tensor:
            return self._convert_to_tensor(self.datasets[split], split)
        else:
            return self.datasets[split]






class Step2PointPointCloud(DataModule):

    def __init__(self, 
                 data_dir,
                 parts=None,
                 sparse_batching=True,
                 **kwargs):
        super().__init__(data_dir=data_dir, **kwargs)        
        self.name = "S2PPC"


        if sparse_batching:
            self.collate_fn = self._collate_sparse
        else:
            self.collate_fn = self._collate_padded


        if self.create_dataset:
            print("Creating Step2PointPointCloud (S2PPC) dataset")
            self._create_dataset()
        else:
            self._load_dataset()


    def _wrap_dataset(self, split):
        
        # wrap pandas dataframe into pytorch dataset
        df = self.datasets[split]
        
        class _DatasetWrapper(torch.utils.data.Dataset):
            
            def __init__(self, df):
                self.df = df
                self.event_ids = df["event_id"].unique()
                self.grouped = df.groupby("event_id")
                self.feature_cols = [col for col in self.df.columns if col not in ["label", "event_id"]]

            def __len__(self):
                return len(self.event_ids)
            
            def __getitem__(self, idx):

                event_id = self.event_ids[idx]
                event_data = self.grouped.get_group(event_id)                
                features = torch.tensor(event_data[self.feature_cols].values, dtype=torch.float32)
                label  = torch.tensor(event_data["label"].iloc[0], dtype=torch.float32).unsqueeze(0)
                return features, label
       
        return _DatasetWrapper(df)

    def get_train_loader(self):
        return DataLoader(
            self._wrap_dataset("train"), 
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )

    def get_val_loader(self):
        return DataLoader(
            self._wrap_dataset("val"), 
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def get_test_loader(self):
        return DataLoader(
            self._wrap_dataset("test"), 
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def _preprocess_data(self, data, particle):
        
        df = pd.DataFrame({
            "event_id": data["event_id"],
            "energy": data["energy"],
            "position_x": data["position"][:, 0],
            "position_y": data["position"][:, 1],
            "position_z": data["position"][:, 2],
            "time": data["time"]
        })

        df["label"] = particle
        df["label"] = df["label"].map({"proton": 0, "piM": 1})

        if df.isna().any().any():
            print("There are NaN values in the dataset!")
        else:
            print("No NaN values detected.")
        
        return df


    def _save_datasets(self):

        for split in self.datasets:

            data = self.datasets[split]
            source_files = data["source_file"].unique()
            print(f"Saving {split} dataset")
            
            for f in source_files:
                
                basename = os.path.basename(f)
                part_str = basename.split("_")[-1]
                part = int(part_str.replace("file", "").replace(".h5", ""))

                data_filtered = data.copy()
                data_filtered = data_filtered[data_filtered["source_file"]==f]
                filepath = os.path.join(self.data_dir, f"{self.name}_{split}_{part}.npz")
                np.savez(
                    filepath,
                    event_id = data_filtered["event_id"].to_numpy(),
                    energy = data_filtered["energy"].to_numpy(),
                    position_x = data_filtered["position_x"].to_numpy(),
                    position_y = data_filtered["position_y"].to_numpy(),
                    position_z = data_filtered["position_z"].to_numpy(), 
                    time = data_filtered["time"].to_numpy(),
                    label = data_filtered["label"].to_numpy(),
                )
            print("Finished saving data")

    def _load_dataset(self):
        for split in self.datasets:

            pattern = os.path.join(self.data_dir, f"{self.name}_{split}_*.npz")
            file_paths = sorted(glob.glob(pattern))

            if self.parts:
                file_paths = file_paths[:self.parts]

            if len(file_paths) == 0:
                raise FileNotFoundError(f"No files found for pattern: {pattern}")

            print(f"Loading {split} dataset from {len(file_paths)} files")

            dfs = []

            for f in file_paths:

                data = np.load(f)
                df = pd.DataFrame({
                    "event_id": data["event_id"],
                    "energy": data["energy"],
                    "position_x": data["position_x"],
                    "position_y": data["position_y"],
                    "position_z": data["position_z"],
                    "time": data["time"],
                    "label": data["label"]
                })
                dfs.append(df)


            self.datasets[split] = pd.concat(dfs, ignore_index=True)

        print("Finished loading datasets")



    def _collate_sparse(self, batch):
        """
        features:       (sum(N_hits_i), F)
        lengths:        (B,)
        labels:         (B,1)
        """
        features_list, labels = zip(*batch)
        features = torch.cat(features_list, dim=0)
        lengths = torch.tensor([f.size(0) for f in features_list]).long()
        labels = torch.stack(labels).float()

        return features, lengths, labels


    def _collate_padded(self, batch):
        """
        features:  (B, N_max, F)
        mask:   (B, N_max)
        labels: (B,1)
        """
        features_list, labels_list = zip(*batch)
        B = len(batch)
        lengths = [f.size(0) for f in features_list]
        N_max = max(lengths)

        F = features_list[0].size(1)

        features = torch.zeros((B, N_max, F))
        mask  = torch.zeros((B, N_max))

        for i, f in enumerate(features_list):
            
            n = f.size(0)
            features[i, :n] = f
            mask[i, :n] = 1.0

        labels = torch.stack(labels_list).float()

        return features, mask, labels






class Step2PointGraph(DataModule):
    
    def __init__(self):
        pass


if __name__ == "__main__":

    data_dir = r"C:\Users\jakobbm\OneDrive - NTNU\Documents\phd\git\point-cloud-classifier\data\continuous"
    Step2PointPointCloud(data_dir=data_dir, create_dataset=True)