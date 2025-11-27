
import os 
import numpy as np
import h5py
import pandas as pd
import torch
import joblib
import glob
import pickle

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
        joblib.dump(scaler, os.path.join(self.data_dir, f"{self.name}_scaler.pkl"))        

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
            filepath = os.path.join(self.data_dir, f"{self.name}_{split}.npz")
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
            filepath = os.path.join(self.data_dir, f"{self.name}_{split}.npz")
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

            data = self.datasets[split]
            source_files = data["source_file"].unique()
            print(f"Saving {split} dataset")
            
            parts = [int(os.path.basename(f).split("_")[-1].replace("file", "").replace(".h5", "")) for f in source_files]
            unique_parts = set(parts)

            for part in unique_parts:

                files_for_part = [f for f in source_files if int(os.path.basename(f).split("_")[-1].replace("file", "").replace(".h5", "")) == part]
                df_part = data.copy()                
                df_part = df_part[df_part["source_file"].isin(files_for_part)]
                filepath = os.path.join(self.data_dir, f"{self.name}_{split}_{part}.npz")
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
                 parts=None,
                 **kwargs):
        super().__init__(data_dir=data_dir, **kwargs)        
        self.name = "S2PG"

        self.parts = parts

        # node features
        # for batch of B graphs: shape (total nodes in batch, number of node features) with particle index
        # all nodes from each graph are stacked together

        # adjacency list:
        # shape (total nodes in batch, max degree)
        
        # degree list 
        # shape (total nodes in batch)
        # 

        # graph member ship
        # shape (total nodes in batch)
        # each node indicate its membership

        if self.create_dataset:
            print("Creating Step2PointGraph (S2PG) dataset")
            self._create_dataset()
        else:
            self._load_dataset()


       

    def _preprocess_data(self, data, particle):
        
        df_steps = pd.DataFrame({
            "event_id": data["event_id"],
            "energy": data["energy"],
            "position_x": data["position"][:, 0],
            "position_y": data["position"][:, 1],
            "position_z": data["position"][:, 2],
            "time": data["time"],
            "particle_id": data["mcparticle_id"]
        })

        df_particles = pd.DataFrame({
            "event_id": data["particle_event_id"],
            "particle_id": data["particle_id"],
            "parent_particle_id": data["parent_id"]
        })


        df_steps = df_steps.sort_values(["event_id", "particle_id", "time"]).reset_index(drop=True)
        df_steps["unique_key"] = df_steps.groupby("event_id").cumcount()

        
        df_steps_grouped = df_steps.groupby("event_id")
        df_particles_grouped = df_particles.groupby("event_id")
    
        graphs = []
        event_ids = df_steps["event_id"].unique()

        for event in event_ids:          
            print("Event:", event)

            df_steps_event = df_steps_grouped.get_group(event).copy()
            df_particles_event = df_particles_grouped.get_group(event).copy()

            incident_series = df_particles_event.loc[df_particles_event["parent_particle_id"] == -1, "particle_id"]
            assert len(incident_series) == 1, f"Event {event}: expected 1 primary particle, found {len(incident_series)}"
            assert incident_series.iloc[0] == 0, f"Event {event}: primary particle ID is not 0"
            incident_id = int(incident_series.iloc[0]) 
            incident_key = df_steps_event["unique_key"].max()+1

            incident_step = {
                "event_id": event,
                "energy": 0.0,            
                "position_x": 0.0,
                "position_y": 0.0,
                "position_z": 0.0,
                "time": 0.0,
                "particle_id": incident_id,
                "unique_key": incident_key  
            }

            df_steps_event = pd.concat([df_steps_event, pd.DataFrame([incident_step])], ignore_index=True)

            step_data_event = {
                "particle_id": df_steps_event["particle_id"].to_numpy(),
                "time": df_steps_event["time"].to_numpy(),
                "energy": df_steps_event["energy"].to_numpy(),
                "pos_x": df_steps_event["position_x"].to_numpy(),
                "pos_y": df_steps_event["position_y"].to_numpy(),
                "pos_z": df_steps_event["position_z"].to_numpy(),
                "unique_key": df_steps_event["unique_key"].to_numpy()
            }


            unique_particle_ids, inverse_idx = np.unique(step_data_event["particle_id"], return_inverse=True)
            indices_by_particle = {}
            for particle_id, grp_index in zip(unique_particle_ids, np.arange(len(unique_particle_ids))):    
                indices = np.nonzero(inverse_idx == grp_index)[0]
                indices_by_particle[int(particle_id)] = indices  # indices into df_steps_event


            parent_map = {}  # child -> list of parent particle ids (empty list if none)
            for _, row in df_particles_event.iterrows():
                child = int(row["particle_id"])
                parent = int(row["parent_particle_id"])
                if child not in parent_map:
                    parent_map[child] = []
                if parent != -1:
                    parent_map[child].append(parent)


            links = self._find_links(unique_particle_ids, parent_map, step_data_event, indices_by_particle)
            
            edge_list = [[] for _ in range(incident_key+1)]
            parent_list = [[] for _ in range(incident_key+1)]

            assert len(step_data_event["pos_x"]) == len(edge_list)


            for parent, child in links:
                edge_list[child].append(parent)
                edge_list[parent].append(child)
                parent_list[child].append(parent)

            degree_list = [len(x) for x in edge_list]
            in_degree_list = [len(x) for x in parent_list]

            feature_list = np.stack([
                step_data_event["energy"]/sum(step_data_event["energy"]),
                step_data_event["pos_x"],
                step_data_event["pos_y"],
                step_data_event["pos_z"]
            ], axis=1)
            
            graph = {
                "event_id": event,
                "nodes": step_data_event["unique_key"],
                "features": feature_list, 
                "edges": edge_list,
                "degree": degree_list, 
                "label": particle}
        
            graphs.append(graph)

            if in_degree_list[incident_key] != 0:
                # print(incident_key)
                # print(in_degree_list[incident_key])
                # print(parent_list[incident_key])
                # print(child_list[incident_key])
                # print(feature_list[incident_key])
                # print(step_data_event["particle_id"][parent_list[incident_key][0]])
                print("Incident particle has parents, which should not happen")


            unconnected_keys = [k for k, deg in enumerate(in_degree_list[:-1]) if deg == 0]

            if len(unconnected_keys) > 0:
                print(f"{len(unconnected_keys)} nodes with no parents found")
   
        # remap event_id
        for new_id, g in enumerate(graphs):
            g["event_id"] = new_id

        return graphs 





    def _find_links(self, unique_particle_ids, parent_map, step_data_event, indices_by_particle):

        ancestor_steps_cache = {}
        step_times = step_data_event["time"]
        step_unique_key = step_data_event["unique_key"]

        links_time = [] 
        links_parent = [] 

        for child_id in unique_particle_ids:

            
            child_idxs = indices_by_particle.get(child_id)
            child_idxs_sorted = child_idxs[np.argsort(step_times[child_idxs])]
            n_steps = len(child_idxs)

            if n_steps > 1:
                for i in range(n_steps-1):
                    parent_idx = child_idxs_sorted[i]
                    k_parent = step_unique_key[parent_idx]
                    child_idx = child_idxs_sorted[i+1]
                    k_child = step_unique_key[child_idx]
                    links_time.append((k_parent, k_child))



            parent_ids = self.get_ancestor_steps(child_id, unique_particle_ids, parent_map, ancestor_steps_cache)
            if len(parent_ids) == 0:
                print(f"No parents exist for particle {child_id}")
                continue
            
            child_times = step_times[child_idxs]
            min_time = np.min(child_times)
            min_time_idx_all = np.where(child_times==min_time)[0]                
            child_idx_all = child_idxs[min_time_idx_all]   # idx in steps
            child_keys = step_unique_key[child_idx_all]


            for parent_id in parent_ids:

                candidate_idxs = indices_by_particle.get(parent_id)
                candidate_times = step_times[candidate_idxs]
                candidate_delta_times = abs(candidate_times-min_time)
                min_delta_time = np.min(candidate_delta_times)
                min_delta_time_idx_all = np.where(candidate_delta_times==min_delta_time)[0]                  
                parent_idx_all = candidate_idxs[min_delta_time_idx_all] # idx in steps
                parent_keys = step_unique_key[parent_idx_all]
                
                for k_child in child_keys:
                    for k_parent in parent_keys:
                        links_parent.append((k_parent, k_child))

        links = links_time + links_parent
        return links 
            

    def get_ancestor_steps(self, particle_id, unique_particle_ids, parent_map, cache):

        """Return indices (into df_steps_event) for pid or nearest ancestor with steps.
            caches results per particle id."""


        # if particle ancestor already exist in cache.
        if particle_id in cache:
            return cache[particle_id]  
        
        collected = []
        visited = set()
        queue = parent_map.get(particle_id, []).copy()


        while queue:

            cur = queue.pop(0)
            cur = np.int64(cur)

            if cur in visited:
                continue
            visited.add(cur)    

            if cur not in unique_particle_ids:
                
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
            cache[particle_id] = collected
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


    def _scale_features(self):

        print("Scaling features")

        X_train = np.vstack([g["features"] for g in self.datasets["train"]])
        X_val   = np.vstack([g["features"] for g in self.datasets["val"]])
        X_test  = np.vstack([g["features"] for g in self.datasets["test"]])

        # fit scaler on train
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scaler = scaler
        joblib.dump(scaler, os.path.join(self.data_dir, f"{self.name}_scaler.pkl"))        

        # transform val and test
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        self.write_back(self.datasets["train"], X_train_scaled)
        self.write_back(self.datasets["val"],   X_val_scaled)
        self.write_back(self.datasets["test"],  X_test_scaled)


    def write_back(self, graphs, X_scaled):
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
            
            parts = [int(os.path.basename(f).split("_")[-1].replace("file", "").replace(".h5", "")) for f in source_files]
            unique_parts = set(parts)

            for part in unique_parts:

                files_for_part = [f for f in source_files if int(os.path.basename(f).split("_")[-1].replace("file", "").replace(".h5", "")) == part]
                g_part = data.copy()                
                g_part = [g for g in g_part if g["source_file"] in files_for_part]
                filepath = os.path.join(self.data_dir, f"{self.name}_{split}_{part}.npz")

                np.savez(
                    filepath,
                    features=np.array([g["features"] for g in g_part], dtype=object),
                    nodes=np.array([g["nodes"] for g in g_part], dtype=object),
                    edges=np.array([g["edges"] for g in g_part], dtype=object),
                    degree=np.array([g["degree"] for g in g_part], dtype=object),
                    labels=np.array([g["label"] for g in g_part], dtype=object),
                    event_ids=np.array([g["event_id"] for g in g_part])
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

                graphs = []

                for file_path in file_paths:

                    data = np.load(file_path, allow_pickle=True)

                    features_arr = data["features"]
                    nodes_arr = data["nodes"]
                    edges_arr = data["edges"]
                    degree_arr = data["degree"]
                    labels_arr = data["labels"]
                    event_ids_arr = data["event_ids"]


                    # reconstruct list of dicts
                    for i in range(len(event_ids_arr)):
                        graph = {
                            "event_id": event_ids_arr[i],
                            "node": nodes_arr[i],
                            "features": features_arr[i],
                            "edges": edges_arr[i],
                            "degree": degree_arr[i],
                            "label": labels_arr[i]
                        }
                        graphs.append(graph)


                self.datasets[split] = graphs
            
            print("Finished loading datasets")


    # def _wrap_dataset(self, split):
        
    #     # wrap pandas dataframe into pytorch dataset
    #     df = self.datasets[split]
    #     features_df = 1
    #     adjacency_df = 1
    #     class _DatasetWrapper(torch.utils.data.Dataset):
            
    #         def __init__(self, df):

    #             self.df = df
    #             self.event_ids = df["event_id"].unique()


    #             self.features = features_df.groupby("event_id")
    #             self.adjacency = adjacency_df.groupby("event_id")


    #             self.feature_cols = [col for col in self.df.columns if col not in ["label", "event_id"]]

    #         def __len__(self):
    #             return len(self.event_ids)
            
    #         def __getitem__(self, idx):

    #             event_id = self.event_ids[idx]
                
                
    #             feature_data = self.features.get_group(event_id) 
    #             adjacency_data = self.adjacency.get_group(event_id)                

    #             membership_data = 
                

    #             # both time index and particle index
                




    #             features = torch.tensor(event_data[self.feature_cols].values, dtype=torch.float32)
    #             label  = torch.tensor(event_data["label"].iloc[0], dtype=torch.float32).unsqueeze(0)
    #             return features, label
       
    #     return _DatasetWrapper(df)








    #     if df.isna().any().any():
    #         print("There are NaN values in the dataset!")
    #     else:
    #         print("No NaN values detected.")
        
    #     return df





if __name__ == "__main__":

    data_dir = r"C:\Users\jakobbm\OneDrive - NTNU\Documents\phd\git\point-cloud-classifier\data\continuous"
    Step2PointGraph(data_dir=data_dir, create_dataset=True)