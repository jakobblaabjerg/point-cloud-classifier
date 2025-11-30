
import os 
import numpy as np
import h5py
import pandas as pd
import torch
import joblib
import glob
import pickle

from tqdm import tqdm
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
            particles_group = f["particles"]

            subdetector_meta = metadata_group["subdetector_names"][:]
            subdetector = steps_group["subdetector"][:]
            subdetector_decoded = subdetector_meta[subdetector]

            data = {
                "energy": steps_group["energy"][:],
                "event_id": steps_group["event_id"][:],
                "position": steps_group["position"][:],
                "time": steps_group["time"][:],
                "mcparticle_id": steps_group["mcparticle_id"][:],
                "particle_id": particles_group["id"][:],
                "parent_id": particles_group["parent_id"][:],
                "particle_event_id": particles_group["event_id"][:],
                "subdetector": subdetector_decoded,
            }

        return data

    def _create_dataset(self):

        self.datasets = {"train": [],
        "val": [],
        "test": []
        }

        event_id_offset = 0

        for particle in self.particles:
            files = self._find_files(particle)
        
            for f in files:
                print(os.path.basename(f))
                data_raw = self._load_h5py_file(f)
                num_events = len(np.unique(data_raw["event_id"]))
                data_preprocessed  = self._preprocess_data(data_raw, particle)
                data_preprocessed["source_file"] = os.path.basename(f)
                data_preprocessed["event_id"] = data_preprocessed["event_id"]+event_id_offset
                event_id_offset += num_events
                train_df, val_df, test_df = self._split_dataset(data_preprocessed) # file level split 
                self.datasets["train"].append(train_df)
                self.datasets["val"].append(val_df)
                self.datasets["test"].append(test_df)

        # concatenate each split
        for split in self.datasets:
            self.datasets[split] = pd.concat(self.datasets[split], ignore_index=True)

        total_events = 0
        for split in self.datasets:
            total_events += len(set(self.datasets[split]["event_id"]))

        assert event_id_offset == total_events


        if self.feature_scaling:
            self._scale_features()
        
        self._save_datasets()
        
        # remove source file column after data is saved
        for split in self.datasets:
            self.datasets[split] = self.datasets[split].drop(columns=["source_file"])





    def _scale_features(self):

        ignore_cols = ["label", "event_id", "source_file"]
        feature_cols = [col for col in self.datasets["train"].columns if col not in ignore_cols]
        print("Scaling the following columns:", feature_cols)

        # extract features
        X_train = self.datasets["train"][feature_cols]
        X_val   = self.datasets["val"][feature_cols]
        X_test  = self.datasets["test"][feature_cols]

        # fit scaler on train
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scaler = scaler
        save_dir = os.path.join(self.data_dir, self.name)
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(save_dir, f"{self.name}_scaler.pkl"))        

        # transform val and test
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # replace original feature columns with scaled values
        for split, X_scaled in zip(["train", "val", "test"], [X_train_scaled, X_val_scaled, X_test_scaled]):
            df_scaled = self.datasets[split].copy()
            df_scaled[feature_cols] = X_scaled
            self.datasets[split] = df_scaled


    def _remap_event_ids(self, df):
        df = df.copy()
        unique_event_ids = df["event_id"].unique()
        mapping = {old: new for new, old in enumerate(unique_event_ids)}
        df["event_id"] = df["event_id"].map(mapping)

        return df

    def _split_dataset(self, dataset):
        # should ideally be moved upstream
        train_frac, val_frac, test_frac = self.data_split

        event_ids = dataset["event_id"].unique()

        train_ids, test_ids = train_test_split(
            event_ids, 
            test_size=test_frac,
            stratify=dataset.groupby("event_id")["label"].first(),
            random_state=42
        )

        train_ids, val_ids = train_test_split(
            train_ids,
            test_size=val_frac / (val_frac + train_frac),
            stratify=dataset.groupby("event_id")["label"].first().loc[train_ids],
            random_state=42
        )

        train_df = dataset[dataset["event_id"].isin(train_ids)]
        val_df   = dataset[dataset["event_id"].isin(val_ids)]
        test_df  = dataset[dataset["event_id"].isin(test_ids)]

        return train_df, val_df, test_df


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
        df_preprocessed = df_preprocessed[["event_id", "energy_total", "hits_total", "energy_hcal_frac", "hits_hcal_frac", "label"]]

        df_preprocessed = self._remap_event_ids(df_preprocessed)

        if df_preprocessed.isna().any().any():
            print("There are NaN values in the dataset!")
        else:
            print("No NaN values detected.")
        
        return df_preprocessed



    def _save_datasets(self):

        for split in self.datasets:
            data = self.datasets[split]
            print(f"Saving {split} dataset")
            save_dir = os.path.join(self.data_dir, self.name, split)
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, f"{self.name}_{split}.npz")
            np.savez(
                filepath,
                event_id = data["event_id"].to_numpy(),
                energy_total = data["energy_total"].to_numpy(),
                hits_total   = data["hits_total"].to_numpy(),
                energy_hcal_frac = data["energy_hcal_frac"].to_numpy(),
                hits_hcal_frac   = data["hits_hcal_frac"].to_numpy(),
                label       = data["label"].to_numpy(),
            )
        print("Finished saving data")

    def _load_dataset(self):
        for split in self.datasets:
            save_dir = os.path.join(self.data_dir, self.name, split)
            filepath = os.path.join(save_dir, f"{self.name}_{split}.npz")
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Required file is missing: {filepath}")

            print(f"Loading {split} dataset from {filepath}")
            data = np.load(filepath)
            df = pd.DataFrame({
                "event_id": data["event_id"],
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


    def get_train_loader(self):
        split ="train"
        self.datasets[split] = self.datasets[split].drop(columns=["event_id"])
        if self.convert_to_tensor:
            return self._convert_to_tensor(self.datasets[split], split)
        else:
            return self.datasets[split]

    def get_val_loader(self):
        split="val"
        self.datasets[split] = self.datasets[split].drop(columns=["event_id"])        
        if self.convert_to_tensor:
            return self._convert_to_tensor(self.datasets[split], split)
        else:
            return self.datasets[split]

    def get_test_loader(self):
        split="test"
        self.datasets[split] = self.datasets[split].drop(columns=["event_id"])
        if self.convert_to_tensor:
            return self._convert_to_tensor(self.datasets[split], split)
        else:
            return self.datasets[split]

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


class Step2PointPointCloud(DataModule):

    def __init__(self, 
                 data_dir,
                 parts=None,
                 sparse_batching=True,
                 energy_cutoff=None,
                 **kwargs):
        super().__init__(data_dir=data_dir, **kwargs)        
        self.name = "S2PPC"

        self.parts = parts

        if sparse_batching:
            self.collate_fn = self._collate_sparse
        else:
            self.collate_fn = self._collate_padded

        self.energy_cutoff = energy_cutoff

        if self.create_dataset:
            print("Creating Step2PointPointCloud (S2PPC) dataset")
            self._create_dataset()
        else:
            self._load_dataset()


    def _wrap_dataset(self, split):
        
        # wrap pandas dataframe into pytorch dataset
        df = self.datasets[split]

        # if split=="train":
        #     df["label"] = df["label"].sample(frac=1, random_state=42).values
        
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

        print("Length before:", len(df))
        if self.energy_cutoff:
            df = df[df["energy"] >= self.energy_cutoff]
        print("Length after:", len(df))


        # sum energy per event
        df_grouped = df.groupby("event_id")["energy"].sum().reset_index()
        df_grouped = df_grouped.rename(columns={"energy": "energy_total"})

        # merge back to original df
        df = df.merge(df_grouped, on="event_id", how="left")

        df["energy"] = df["energy"]/df["energy_total"]


        # compute min and max per event
        tmin = df.groupby("event_id")["time"].transform("min")
        tmax = df.groupby("event_id")["time"].transform("max")
        df["time"] = (df["time"] - tmin) / (tmax - tmin + 1e-8)


        df = self._remap_event_ids(df)
        df["label"] = particle
        df["label"] = df["label"].map({"proton": 0, "piM": 1})

        if df.isna().any().any():
            print("There are NaN values in the dataset!")
        else:
            print("No NaN values detected.")
        
        return df



            

    def _save_datasets(self):

        for split in self.datasets:

            save_dir = os.path.join(self.data_dir, self.name, split)
            os.makedirs(save_dir, exist_ok=True)

            data = self.datasets[split]
            source_files = data["source_file"].unique()
            print(f"Saving {split} dataset")
            
            parts = [int(os.path.basename(f).split("_")[-1].replace("file", "").replace(".h5", "")) for f in source_files]
            unique_parts = set(parts)

            for part in unique_parts:

                files_for_part = [f for f in source_files if int(os.path.basename(f).split("_")[-1].replace("file", "").replace(".h5", "")) == part]
                df_part = data.copy()                
                df_part = df_part[df_part["source_file"].isin(files_for_part)]
                filepath = os.path.join(save_dir, f"{self.name}_{split}_{part}.npz")
                np.savez(
                    filepath,
                    event_id = df_part["event_id"].to_numpy(),
                    energy = df_part["energy"].to_numpy(),
                    energy_total = df_part["energy_total"].to_numpy(),
                    position_x = df_part["position_x"].to_numpy(),
                    position_y = df_part["position_y"].to_numpy(),
                    position_z = df_part["position_z"].to_numpy(), 
                    time = df_part["time"].to_numpy(),
                    label = df_part["label"].to_numpy(),
                )
            print("Finished saving data")

    def _load_dataset(self):
        for split in self.datasets:

            save_dir = os.path.join(self.data_dir, self.name, split)

            pattern = os.path.join(save_dir, f"{self.name}_{split}_*.npz")
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
                    "energy_total": data["energy_total"],
                    "position_x": data["position_x"],
                    "position_y": data["position_y"],
                    "position_z": data["position_z"],
                    "time": data["time"],
                    "label": data["label"]
                })
                dfs.append(df)


            self.datasets[split] = pd.concat(dfs, ignore_index=True)

        print("Finished loading datasets")


    # are labels and events matching?

    def _collate_sparse(self, batch):
        """
        features:       (sum(N_hits_i), F)
        lengths:        (B,)
        labels:         (B,1)
        """
        features, labels = zip(*batch)
        x = torch.cat(features, dim=0)
        idx = torch.cat([torch.full((f.size(0),), i, dtype=torch.long) 
                     for i, f in enumerate(features)])
        labels = torch.stack(labels).float()

        return x, idx, labels


    def _collate_padded(self, batch):
        """
        features:  (B, N_max, F)
        mask:   (B, N_max)
        labels: (B,1)
        """
        features, labels_list = zip(*batch)
        B = len(batch)
        lengths = [f.size(0) for f in features]
        N_max = max(lengths)

        F = features[0].size(1)

        x = torch.zeros((B, N_max, F))
        mask  = torch.zeros((B, N_max))

        for i, f in enumerate(features):
            
            n = f.size(0)
            x[i, :n] = f
            mask[i, :n] = 1.0

        labels = torch.stack(labels_list).float()

        return x, mask, labels


class Step2PointGraph(DataModule):
    
    def __init__(self, 
                 data_dir,
                 n_features=4,
                 parts=None,
                 use_weights=True,
                 **kwargs):
        super().__init__(data_dir=data_dir, **kwargs)        

        self.name = "S2PG"
        self.parts = parts
        self.use_weights = use_weights
        self.n_features = n_features

        if self.create_dataset:
            print("Creating Step2PointGraph (S2PG) dataset")
            self._create_dataset()

    def _preprocess_data(self, data, particle):
        
        df_steps = pd.DataFrame({
            "event_id": data["event_id"],
            "energy": data["energy"],
            "pos_x": data["position"][:, 0],
            "pos_y": data["position"][:, 1],
            "pos_z": data["position"][:, 2],
            "time": data["time"],
            "pid": data["mcparticle_id"]
        })

        df_particles = pd.DataFrame({
            "event_id": data["particle_event_id"],
            "pid": data["particle_id"],
            "parent_pid": data["parent_id"]
        })

        # create unique key for each step based on particle id and time.
        df_steps = df_steps.sort_values(["event_id", "pid", "time"]).reset_index(drop=True)
        df_steps["step_key"] = df_steps.groupby("event_id").cumcount()
      
        df_steps_grouped = df_steps.groupby("event_id")
        df_particles_grouped = df_particles.groupby("event_id")
    
        graphs = []
        event_ids = df_steps["event_id"].unique()

        for event in tqdm(event_ids, desc="Processing events", leave=True):

            df_steps_event = df_steps_grouped.get_group(event).copy()
            df_particles_event = df_particles_grouped.get_group(event).copy()

            # find incident particle (has parent -1)
            incident_series = df_particles_event.loc[df_particles_event["parent_pid"] == -1, "pid"]
            incident_pid = int(incident_series.iloc[0]) 
            incident_step_key = df_steps_event["step_key"].max()+1
            
            # sanity check
            assert len(incident_series) == 1, f"Event {event}: expected 1 primary particle, found {len(incident_series)}"
            assert incident_series.iloc[0] == 0, f"Event {event}: primary particle ID is not 0"

            # create extra step to ensure graph is connected.
            incident_step = {
                "event_id": event,
                "energy": 0.0,            
                "pos_x": 0.0,
                "pos_y": 0.0,
                "pos_z": 0.0,
                "time": 0.0,
                "pid": incident_pid,
                "step_key": incident_step_key  
            }
            df_steps_event = pd.concat([df_steps_event, pd.DataFrame([incident_step])], ignore_index=True)

            # convert to numpy arrays
            step_data_event = {
                "pid":          df_steps_event["pid"].to_numpy(),
                "time":         df_steps_event["time"].to_numpy(),
                "energy":       df_steps_event["energy"].to_numpy(),
                "pos_x":        df_steps_event["pos_x"].to_numpy(),
                "pos_y":        df_steps_event["pos_y"].to_numpy(),
                "pos_z":        df_steps_event["pos_z"].to_numpy(),
                "step_key":     df_steps_event["step_key"].to_numpy()
                }

            # create inverse index map for pid into step data event 
            unique_pids, inverse_idx = np.unique(step_data_event["pid"], return_inverse=True)
            indices_map = {}
            for pid, grp_index in zip(unique_pids, np.arange(len(unique_pids))):    
                indices = np.nonzero(inverse_idx == grp_index)[0]
                indices_map[int(pid)] = indices 

            # create map for each child to its parents pids (empty list if none) 
            parent_map = {}  
            for _, row in df_particles_event.iterrows():
                child = int(row["pid"])
                parent = int(row["parent_pid"])

                if child not in parent_map:
                    parent_map[child] = []
                if parent != -1:
                    parent_map[child].append(parent)

            # find edges between nodes (particles) 
            edges = Step2PointGraph._find_edges(unique_pids, 
                                                parent_map, 
                                                step_data_event, 
                                                indices_map
                                                )
            
            # sanity check 
            assert len(step_data_event["pos_x"]) == incident_step_key+1

            total_energy = sum(step_data_event["energy"])

            features = np.stack([
                step_data_event["energy"]/total_energy,
                step_data_event["pos_x"],
                step_data_event["pos_y"],
                step_data_event["pos_z"]
            ], axis=1).astype(np.float32)
            
            weights = Step2PointGraph._compute_weights(features, edges)

            label_map = {"proton": 0, "piM": 1}
            label = label_map[particle]

            graph = {
                "event_id": event,
                "features": features, 
                "edges": edges,
                "weights": weights, 
                "label": label}
        
            graphs.append(graph)

        # remap event_id
        for new_id, g in enumerate(graphs):
            g["event_id"] = new_id

        return graphs 

    @staticmethod
    def _compute_weights(features, edges, eps=1e-6):

        positions = features[:,1:4]
        src_pos = positions[edges[0]]   
        tgt_pos = positions[edges[1]]   
        dists = np.linalg.norm(src_pos - tgt_pos, axis=1)
        sigma = np.median(dists) + eps
        weights = np.exp(- (dists**2) / (2 * sigma**2))

        return np.array(weights, dtype=np.float32)

    @staticmethod
    def _find_edges(unique_pids, parent_map, step_data_event, indices_map):

        ancestor_cache = {}
        time_event = step_data_event["time"]
        step_key_event = step_data_event["step_key"]

        edges_time = [] 
        edges_parent = [] 

        for child_pid in unique_pids:
          
            child_idxs = indices_map.get(child_pid)
            child_idxs_sorted = child_idxs[np.argsort(time_event[child_idxs])]
            n_steps = len(child_idxs)

            if n_steps > 1:
                for i in range(n_steps-1):
                    parent_idx = child_idxs_sorted[i]
                    source = step_key_event[parent_idx]

                    child_idx = child_idxs_sorted[i+1]
                    target = step_key_event[child_idx]
                    
                    edges_time.append((source, target))


            parent_pids = Step2PointGraph._get_ancestor_pids(
                child_pid, 
                unique_pids, 
                parent_map, 
                ancestor_cache
                )
            
            if len(parent_pids) == 0:
                
                if child_pid != 0:
                    print(f"No parents exist for particle {child_pid}")

                continue
            
            child_times = time_event[child_idxs]
            min_time = np.min(child_times)
            min_time_idx_all = np.where(child_times==min_time)[0]                
            child_idx_all = child_idxs[min_time_idx_all]   # idx in steps
            step_keys_child = step_key_event[child_idx_all]

            for parent_pid in parent_pids:

                candidate_idxs = indices_map.get(parent_pid)
                candidate_times = time_event[candidate_idxs]
                candidate_delta_times = abs(candidate_times-min_time)
                min_delta_time = np.min(candidate_delta_times)
                min_delta_time_idx_all = np.where(candidate_delta_times==min_delta_time)[0]                  
                parent_idx_all = candidate_idxs[min_delta_time_idx_all] # idx in steps
                step_keys_parent = step_key_event[parent_idx_all]
                
                for target in step_keys_child:
                    for source in step_keys_parent:
                        edges_parent.append((source, target))

        edges = edges_time + edges_parent


        edges_bidirectional = []
        incident_step_key = step_key_event[-1]
        in_degree = [0]*(incident_step_key+1)

        #bidreactional
        for source, target in edges:
            edges_bidirectional.append([source, target])
            edges_bidirectional.append([target, source])
            in_degree[target]+=1

        # sanity check 
        assert in_degree[incident_step_key] == 0, "Incident particle has parents, which should not happen"
        unconnected_nodes = [k for k, deg in enumerate(in_degree[:-1]) if deg == 0]
        assert len(unconnected_nodes) == 0, f"{len(unconnected_nodes)} nodes with no parents found"

        return np.array(edges_bidirectional, dtype=np.int64).T


    @staticmethod
    def _get_ancestor_pids(pid, unique_pids, parent_map, cache):

        """Return pids of nearest ancestor caches results per particle id."""

        # if particle ancestor already exist in cache.
        if pid in cache:
            return cache[pid]  
        
        collected = []
        visited = set()
        queue = parent_map.get(pid, []).copy()

        while queue:

            cur = queue.pop(0)
            cur = np.int64(cur)

            if cur in visited:
                continue
            visited.add(cur)    

            if cur not in unique_pids:
                
                # look in cache 
                if cur in cache:
                    collected.extend(cache[cur])
                    continue

                # expand search
                else:
                    queue.extend(parent_map.get(cur, []))

            else:
                collected.append(cur)
                for child, parents in parent_map.items():
                    if cur in parents and child not in cache and len(parents)==1:
                        cache[child] = [cur]

        if collected:
            cache[pid] = collected
        return collected

    def _split_dataset(self, dataset):

        train_frac, val_frac, test_frac = self.data_split
        event_ids = [g["event_id"] for g in dataset]
        labels = [g["label"] for g in dataset]

        train_val_ids, test_ids, train_val_labels, test_labels = train_test_split(
            event_ids,
            labels, 
            test_size=test_frac,
            stratify=labels,
            random_state=42
        )

        train_ids, val_ids, _, _ = train_test_split(
            train_val_ids,
            train_val_labels,
            test_size=val_frac / (val_frac + train_frac),
            stratify=train_val_labels,
            random_state=42
        )

        train_df = [g for g in dataset if g["event_id"] in train_ids]
        val_df   = [g for g in dataset if g["event_id"] in val_ids]
        test_df  = [g for g in dataset if g["event_id"] in test_ids]

        return train_df, val_df, test_df

    def _create_dataset(self):

        self.datasets = {"train": [],
        "val": [],
        "test": []
        }

        event_id_offset = 0

        for particle in self.particles:
            files = self._find_files(particle)
        
            for f in files:
                print(os.path.basename(f))
                data_raw = self._load_h5py_file(f)
                num_events = len(np.unique(data_raw["event_id"]))
                data_preprocessed  = self._preprocess_data(data_raw, particle) # list of dicts (graphs)

                for graph in data_preprocessed:
                    graph["source_file"] = os.path.basename(f)
                    graph["event_id"] = graph["event_id"] + event_id_offset

                event_id_offset += num_events

                train_df, val_df, test_df = self._split_dataset(data_preprocessed) # file level split 

                self.datasets["train"].extend(train_df)
                self.datasets["val"].extend(val_df)
                self.datasets["test"].extend(test_df)

        # sanity check
        total_events = 0
        for split in self.datasets:
            total_events += len(self.datasets[split])
        # assert event_id_offset == total_events

        print("total_events:", total_events)
        print("event_id_offset:", event_id_offset)

        if self.feature_scaling:
            self._scale_features()


        self._save_datasets()
        
        # remove source file column after data is saved
        for split in self.datasets:
            for g in self.datasets[split]:
                g.pop("source_file", None)


    @staticmethod
    def _scale_position(features):
        position = features[:,1:4]
        energy = features[:, 0:1]
        position_mean = (position * energy).sum(axis=0) / (energy.sum() + 1e-8)
        position_std = np.sqrt(((energy * (position - position_mean)**2).sum(axis=0)) / (energy.sum() + 1e-8))
        position_norm = (position-position_mean)/(position_std + 1e-8)
        features[:,1:4] = position_norm

        return features

    def _scale_features(self):

        print("Scaling features")
        X_train = np.vstack([Step2PointGraph._scale_position(g["features"]) for g in self.datasets["train"]])
        X_val   = np.vstack([Step2PointGraph._scale_position(g["features"]) for g in self.datasets["val"]])
        X_test  = np.vstack([Step2PointGraph._scale_position(g["features"]) for g in self.datasets["test"]])

        scaler = StandardScaler()
        energy_idx = 0
        X_train[:, energy_idx:energy_idx+1] = scaler.fit_transform(X_train[:, energy_idx:energy_idx+1])
        X_val[:, energy_idx:energy_idx+1]   = scaler.transform(X_val[:, energy_idx:energy_idx+1])
        X_test[:, energy_idx:energy_idx+1]  = scaler.transform(X_test[:, energy_idx:energy_idx+1])
        
        self.scaler = scaler
        save_dir = os.path.join(self.data_dir, self.name)
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(save_dir, f"{self.name}_scaler.pkl"))        

        Step2PointGraph._write_back(self.datasets["train"], X_train)
        Step2PointGraph._write_back(self.datasets["val"],   X_val)
        Step2PointGraph._write_back(self.datasets["test"],  X_test)


    @staticmethod
    def _write_back(graphs, X_scaled):
        start = 0
        for g in graphs:
            n = len(g["features"])
            g["features"] = X_scaled[start:start+n]
            start += n

    def _save_datasets(self):

        for split in self.datasets:

            data = self.datasets[split]
            source_files = [g["source_file"] for g in data]
            source_files = set(source_files)
            print(f"Saving {split} dataset")
            save_dir = os.path.join(self.data_dir, self.name, split)
            os.makedirs(save_dir, exist_ok=True)

            for i, g in enumerate(data):
                filepath = os.path.join(save_dir, f"graph_{i:05d}.npz")
                np.savez(
                    filepath,
                    features=g["features"],     # [n_steps ,F]
                    edges=g["edges"],           # [n_edges, 2]
                    weights=g["weights"],       # [n_steps]
                    label=g["label"],
                    event_id=g["event_id"]
                    )
            print("Finished saving data")

    def _load_dataset(self):

        for split in self.datasets:

            save_dir = os.path.join(self.data_dir, self.name, split)
            pattern = os.path.join(save_dir, f"graph_*.npz")
            file_paths = sorted(glob.glob(pattern))


            if len(file_paths) == 0:
                raise FileNotFoundError(f"No files found for pattern: {pattern}")

            print(f"Loading {split} dataset from {len(file_paths)} files")

            graphs = []

            for file_path in file_paths:

                data = np.load(file_path)

                features = data["features"]
                edges = data["edges"]
                weights = data["weights"]
                label = data["label"]
                event_id = data["event_id"]

                g = {
                    "event_id": event_id,
                    "features": features, 
                    "edges": edges,
                    "weights": weights, 
                    "label": label
                    }
            
                graphs.append(g)
            
            self.datasets[split] = graphs
        print("Finished loading datasets")

    def _wrap_dataset(self, split):
        
        data_dir = os.path.join(self.data_dir, self.name, split)
        n_features = self.n_features
 
        # wrap pandas dataframe into pytorch dataset
        class _DatasetWrapper(torch.utils.data.Dataset):
            
            def __init__(self):
                self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
                if len(self.files) == 0:
                    raise FileNotFoundError(f"No .npz files found in {data_dir}")

            def __len__(self):
                return len(self.files)
            
            def __getitem__(self, idx):

                data = np.load(self.files[idx])

                label = torch.tensor(data["label"], dtype=torch.float32)
                
                if n_features == 1:
                    features = data["features"][:, 0:1]
                elif n_features == 4:
                    features = data["features"]

                features = torch.from_numpy(features)
                edges = torch.from_numpy(data["edges"])
                weights = torch.from_numpy(data["weights"])
 
                graph = {
                    "features": features,
                    "edges": edges,
                    "weights": weights,
                }

                return graph, label

        return _DatasetWrapper()

    def get_train_loader(self):
        return DataLoader(
            self._wrap_dataset("train"), 
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._graph_collate
        )

    def get_val_loader(self):
        return DataLoader(
            self._wrap_dataset("val"), 
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._graph_collate
        )

    def get_test_loader(self):
        return DataLoader(
            self._wrap_dataset("test"), 
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._graph_collate
        )

    def _graph_collate(self, batch):

        X_b = []
        y_b = []
        edges_b = []
        weights_b = []
        membership_b = []
        offset = 0

        for idx, (graph, label) in enumerate(batch):

            features = graph["features"]
            edges = graph["edges"]+offset
            weights = graph["weights"]
            n_steps = features.shape[0] 

            X_b.append(features)
            y_b.append(label)
            edges_b.append(edges)
            weights_b.append(weights)
            membership_b.append(torch.full((n_steps,), idx))
            offset += n_steps

        X_b = torch.cat(X_b, dim=0)
        y_b = torch.stack(y_b).unsqueeze(1)
        edges_b = torch.cat(edges_b, dim=1) ## is dim 1 right?
        membership_b = torch.cat(membership_b, dim=0)

        if self.use_weights:
            weights_b = torch.cat(weights_b, dim=0)
        else:
            weights_b = None

        return X_b, membership_b, edges_b, weights_b, y_b



if __name__ == "__main__":
    data_dir = r"C:\Users\jakobbm\OneDrive - NTNU\Documents\phd\git\point-cloud-classifier\data\continuous"
    Step2PointGraph(data_dir=data_dir, create_dataset=True)



    