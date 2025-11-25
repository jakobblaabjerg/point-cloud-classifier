
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

       

    def _preprocess_data(self, data, particle):
        
        df_steps = pd.DataFrame({
            "event_id": data["event_id"],
            "energy": data["energy"],
            "position_x": data["position"][:, 0],
            "position_y": data["position"][:, 1],
            "position_z": data["position"][:, 2],
            "time": data["time"],
            "particle_id": data["mcparticle_id"][:]
        })

        df_particles = pd.DataFrame({
            "event_id": data["particle_event_id"],
            "particle_id": data["particle_id"],
            "parent_particle_id": data["parent_id"],
        })

        df_steps = df_steps.sort_values(["event_id", "particle_id", "time"]) 
        df_steps["unique_key"] = df_steps.groupby("event_id").cumcount()

        
        df_steps_grouped = df_steps.groupby("event_id")
        df_particles_grouped = df_particles.groupby("event_id")
    
        graphs = []
        # count = 0
        event_ids = df_steps["event_id"].unique()

        for event in event_ids:

            print("Event:", event)

            df_steps_event = df_steps_grouped.get_group(event).copy()
            df_particles_event = df_particles_grouped.get_group(event).copy()

            incident = df_particles_event.loc[df_particles_event["parent_particle_id"]==-1, "particle_id"]
            assert len(incident) == 1, f"Event {event}: expected 1 primary particle, found {len(incident)}"
            assert incident.iloc[0] == 0, f"Event {event}: primary particle ID is not 0"

            incident_id = incident.iloc[0] 
            new_key = df_steps_event["unique_key"].max()+1

            incident_step = {
                "event_id": event,
                "energy": 0.0,            
                "position_x": 0.0,
                "position_y": 0.0,
                "position_z": 0.0,
                "time": 0.0,
                "particle_id": incident_id,
                "unique_key": new_key  
            }
            
            # it is not necessary to insert to df_steps only df_steps_event only
            df_steps_event = pd.concat([df_steps_event, pd.DataFrame([incident_step])], ignore_index=True)
           
            particle_links = self._build_particle_links(df_steps_event, df_particles_event)

            child_keys = particle_links["unique_key_child"]
            parent_keys = particle_links["unique_key_parent"]

            parent_list = [[] for _ in range(new_key+1)]
            child_list = [[] for _ in range(new_key+1)]

            for child, parent in zip(child_keys, parent_keys):
                parent_list[child].append(parent)
                #parent_list[parent].append(child)
                child_list[parent].append(child)
            
            in_degree_list = [len(x) for x in parent_list]
            out_degree_list =  [len(x) for x in child_list]
            
            # df step events is sorted by uniiqe key so it will match parent list?
            df_steps_event = df_steps_event.sort_values("unique_key").reset_index(drop=True)
            df_steps_event["energy"] = df_steps_event["energy"]/df_steps_event["energy"].sum()
            feature_list = df_steps_event[["energy", "position_x", "position_y", "position_z"]].to_numpy().astype(np.float32)
            
            key_list = df_steps_event["unique_key"].tolist()
            
            if in_degree_list[new_key] != 0:
                print("Incident particle has parents, which should not happen")

            bad_nodes = [k for k, deg in enumerate(in_degree_list[:-1]) if deg == 0]

            if len(bad_nodes) > 0:
                print(f"{len(bad_nodes)} nodes with no parents found")

            graph = {
                "event_id": event,
                "unique_key": key_list,
                "features": feature_list, 
                "parent": parent_list,
                "child": child_list,
                "in_degree": in_degree_list,
                "out_degree": out_degree_list, 
                "label": particle}
        
            graphs.append(graph)

            # count += 1 
            # if count ==5:
            #     break

        # remap event_id
        for new_id, g in enumerate(graphs):
            g["event_id"] = new_id

        return graphs 



    def _build_particle_links(self, df_steps, df_particles):
        
        particle_links_time = [] # links through time
        particle_links_parent = [] # links from a particle to its parents(s)

        # group dataset by particle id for faster access.
        df_steps_grouped = {particle_id: group.copy() for particle_id, group in df_steps.groupby("particle_id")}

        # loop through unique particle ids
        unique_particle_ids = list(df_steps["particle_id"].unique())
        unique_particle_ids.sort()    




        for particle_id in unique_particle_ids:

            # steps for current and parent(s) particles
            steps_child_particle = df_steps_grouped.get(particle_id) 
            steps_parent_particles = self._collect_parent_steps(particle_id, df_steps_grouped, df_particles)

            # allow one connection between parent and child
            min_time = steps_child_particle["time"].min()
            steps_child_particle_filtered = steps_child_particle.copy()
            steps_child_particle_filtered = steps_child_particle_filtered[steps_child_particle_filtered["time"] == min_time]
            # if len(steps_child_particle_filtered)>1:
            #     print(f"Multiple rows found with identical time for particle {particle_id}")
            #     print(len(steps_child_particle_filtered))  
            
            if not steps_parent_particles.empty:
                
                steps_merged_parent = pd.merge(
                    left=steps_child_particle_filtered, 
                    right=steps_parent_particles, 
                    how="cross", 
                    suffixes=("_child", "_parent")
                    ).copy()
                
                steps_best_match = self._select_best_parent_match(steps_merged_parent)
                particle_links_parent.append(steps_best_match)

            else:
                print(f"No parents exist for particle with id: {particle_id}")

            
            # handle the reamining steps
            n_duplicates = len(steps_child_particle)
            if n_duplicates > 1:
                steps_child_particle = steps_child_particle.sort_values(by='time')

                for i in range(n_duplicates-1):

                    current_row = steps_child_particle.iloc[[i]]   
                    next_row = steps_child_particle.iloc[[i+1]]

                    steps_merged_time = pd.merge(
                        left=next_row, 
                        right=current_row, 
                        how="cross", 
                        suffixes=("_child", "_parent")
                        )

                    steps_merged_time["delta_time"] = steps_merged_time["time_child"]-steps_merged_time["time_parent"]
                    steps_merged_time["depth"] = 0
                    particle_links_time.append(steps_merged_time)

        particle_links_time = self._safe_concat(particle_links_time)
        particle_links_parent = self._safe_concat(particle_links_parent)
        return self._safe_concat([particle_links_parent, particle_links_time])


    def _safe_concat(self, dfs):
        return pd.concat(dfs, axis=0, ignore_index=True) if dfs else pd.DataFrame()

    def _select_best_parent_match(self, steps_merged):
        
        steps_merged["delta_time"] = steps_merged["time_child"]-steps_merged["time_parent"]
        steps_merged["delta_time"] = abs(steps_merged["delta_time"])
        min_delta = steps_merged.groupby(["particle_id_parent", "unique_key_child"])["delta_time"].transform("min")
        steps_best_matches = steps_merged[steps_merged["delta_time"] == min_delta]
        steps_best_matches = steps_best_matches.drop_duplicates()

        return steps_best_matches
                
    def _collect_parent_steps(self, particle_id, df_steps_grouped, df_particles, current_depth=1, max_depth=20, visited=None):
    
        if current_depth > max_depth:
            return pd.DataFrame()

        if visited is None:
            visited = set()

        if particle_id in visited:
            print("Cycle detected")
        visited.add(particle_id)

        parent_ids = self._find_parents(particle_id, df_particles)

        collected_steps = []
        
        for parent_id in parent_ids:


            df_steps_parent = df_steps_grouped.get(parent_id, pd.DataFrame()).copy()

            if not df_steps_parent.empty:
                df_steps_parent["depth"] = current_depth
                collected_steps.append(df_steps_parent)
            
            # recurse upward
            else:
                df_steps_ancestor = self._collect_parent_steps(parent_id, df_steps_grouped, df_particles, current_depth+1, max_depth, visited=visited)
                collected_steps.append(df_steps_ancestor)

        return pd.concat(collected_steps, axis=0, ignore_index=True) if collected_steps else pd.DataFrame()


    def _find_parents(self, particle_id, df_particles):
    
        df_particles_filtered = df_particles.copy()
        df_particles_filtered = df_particles_filtered[(df_particles_filtered["particle_id"] == particle_id) & (df_particles_filtered["parent_particle_id"] != -1)]
        return list(set(df_particles_filtered["parent_particle_id"]))

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
                df_part = data.copy()                
                df_part = [g for g in df_part if g["source_file"] in files_for_part]
                filepath = os.path.join(self.data_dir, f"{self.name}_{split}_{part}.pkl")

                with open(filepath, "wb") as f:
                    pickle.dump(df_part, f)

            print("Finished saving data")


    def _load_dataset(self):
            for split in self.datasets:

                pattern = os.path.join(self.data_dir, f"{self.name}_{split}_*.pkl")
                file_paths = sorted(glob.glob(pattern))

                if self.parts:
                    file_paths = file_paths[:self.parts]

                if len(file_paths) == 0:
                    raise FileNotFoundError(f"No files found for pattern: {pattern}")

                print(f"Loading {split} dataset from {len(file_paths)} files")

                graphs = []

                for file_path in file_paths:
                    with open(file_path, "rb") as f:
                        dataset = pickle.load(f)
                    
                    graphs.extend(dataset)
                
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

    data_dir = r"C:\Users\jakobbm\OneDrive - NTNU\Documents\phd\git\point-cloud-classifier\data\test"
    Step2PointTabular(data_dir=data_dir, create_dataset=True)