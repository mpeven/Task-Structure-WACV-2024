import os
import itertools
import tqdm
import pickle
import numpy as np
import pandas as pd

from functools import cached_property

from machine_config import get_machine_config
CONFIG_DICT = get_machine_config()
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_dataset(dataset_name):
    if dataset_name.lower() == "ikea_asm":
        dataset = IKEA_ASM(CONFIG_DICT['ikea_root'])
    elif dataset_name.lower() == "epic_kitchens":
        dataset = EPIC_Kitchens(CONFIG_DICT['epic_root'])
    elif dataset_name.lower() == "finegym":
        dataset = FineGym(CONFIG_DICT['finegym_root'], full_288_set=True)
    elif dataset_name.lower() == "mhad":
        dataset = MHAD(CONFIG_DICT['mhad_root'])
    else:
        raise ValueError("Can't handle dataset = {dataset_name}")
    return dataset


class DatasetClass:
    def __init__(self, name, root_dir):
        self.name = name
        self.root_dir = root_dir
        self.dataset = self._dataset
        self.distances = {
            "none": np.identity(self.num_labels),
            **self.check_distance_cache(),
        }
        if name != "FineGym":
            for d in [("temporal", "verb_2level"), ("temporal", "object_2level")]:
                self.distances[f"{d[0]}_{d[1]}"] = np.stack([self.distances[d[0]], self.distances[d[1]]], axis=-1)


    def check_distance_cache(self):
        # return {
        #     'temporal': self.load_temporal_distances(),
        #     **self.load_visual_distances()
        # }
        file = f"dataset_info/distances_{self.name.replace(' ', '-')}.npz"
        if os.path.exists(file) or self.name.lower != "finegym":
            return np.load(file, allow_pickle=True)['a'].item()
        else:
            distances = {
                'temporal': self.load_temporal_distances(),
                **self.load_visual_distances()
            }
            np.savez_compressed(file, a=distances)
            return distances

        # Dont need temporal for finegym
        return self.load_visual_distances()

    @property
    def _dataset(self):
        raise NotImplementedError("Subclasses should implement _dataset property")

    @property
    def num_labels(self):
        return len(self.label_info)

    @property
    def label_names(self):
        return list(self.label_info['label'])

    @property
    def label_info(self):
        raise NotImplementedError("Subclasses should implement label_info")

    @property
    def median_activity_length(self):
        all_segments = []
        for vid in self.get_split("train_val"):
            all_segments.append({'start_idx': 0, 'stop_idx': '-', 'label': int(vid['labels'][0])})
            for idx, label in enumerate(vid['labels']):
                if all_segments[-1]['label'] != int(label):
                    all_segments[-1]['stop_idx'] = idx-1
                    all_segments.append({'start_idx': idx, 'stop_idx': '-', 'label': int(label)})
            all_segments[-1]['stop_idx'] = idx
        return int(np.median([x['stop_idx'] - x['start_idx'] + 1 for x in all_segments]))

    def create_windows_gt(self):
        """
        Create dataset of clips from the ground-truth segments in the dataset

        Doesn't work for Epic-Kitchens:
            Only for datasets where there is a single label per frame
        """
        for vid in self.dataset:
            all_windows = []
            start_idx = 0
            for label, group in itertools.groupby(vid['labels']):
                window_size = len(list(group))
                all_windows.append({
                    'first_frame': start_idx,
                    'final_frame': start_idx+window_size-1,
                    'label': label
                })
                start_idx += window_size
            vid['windows_gt'] = all_windows

    def create_windows(self, window_stride=None, fixed_num_windows=None):
        window_size = self.median_activity_length
        window_offset = int((window_size-1)/2)
        for vid in self.dataset:
            if fixed_num_windows != None:
                window_centers = np.linspace(window_offset, len(vid['labels'])-window_offset-1, fixed_num_windows).astype(int)
            elif window_stride != None:
                window_centers = range(window_offset, len(vid['labels'])-window_offset, window_stride)
            else:
                window_centers = range(window_offset, len(vid['labels'])-window_offset, window_size)
            all_windows = []
            for window_center in window_centers:
                all_windows.append({
                    "video_name": vid['video_name'],
                    "split": vid['split'],
                    "frame_directory": vid['frame_directory'],
                    "meta_info": vid['meta_info'],
                    'start_idx': window_center - window_offset,
                    'stop_idx': window_center - window_offset + (window_size-1),
                    'label': int(vid['labels'][window_center]),
                    'window_center_frame': window_center,
                })
                if (window_center - window_offset) < 0:
                    raise ValueError("Made a mistake creating windows, start frame < 0")
                if (window_center - window_offset + (window_size-1)) >= len(vid['labels']):
                    raise ValueError("Made a mistake creating windows, last frame > len(video)")
            vid['windows'] = all_windows

    def get_split(self, train_split):
        """ Returns the portion of self.dataset in the specified split (leaves the class unmodified) """
        if train_split in ['train', 'val', 'test']:
            return [x for x in self.dataset if x['split'] == train_split]
        elif train_split == "train_val":
            return [x for x in self.dataset if x['split'] in ["train", "val"]]
        elif train_split == "all":
            return self.dataset
        else:
            raise ValueError(f"Don't know how to handle train_split='{train_split}'")

    def load_visual_distances(self):
        df = self.label_info.copy()

        distances = {}
        for relationship in [x for x in self.visual_relationships if x != "both"]:
            if '2level' not in relationship:
                distances[relationship] = np.zeros((self.num_labels, self.num_labels)) + 2
                df[relationship] = df[relationship].astype('category').cat.codes.astype(int)
                for _, sub_df in df.groupby(relationship):
                    for idx_a, idx_b in itertools.product(sub_df.index, repeat=2):
                        if idx_a == idx_b:
                            distances[relationship][idx_a, idx_b] = 0
                        else:
                            distances[relationship][idx_a, idx_b] = 1
            else:
                distances[relationship] = np.zeros((self.num_labels, self.num_labels)) + 3
                level1 = relationship.split("_")[0]
                for _, level2_df in df.groupby(relationship):
                    for idx_a, idx_b in itertools.product(level2_df.index, repeat=2):
                        if idx_a == idx_b:
                            distances[relationship][idx_a, idx_b] = 0
                        elif level2_df.loc[idx_a][level1] == level2_df.loc[idx_b][level1]:
                            distances[relationship][idx_a, idx_b] = 1
                        else:
                            distances[relationship][idx_a, idx_b] = 2

        # Add in a both when verb+object in distances
        if "verb" in self.visual_relationships and "object" in self.visual_relationships:
            distances["both"] = np.minimum(distances["verb"], distances["object"])

        return distances


    def load_temporal_distances(self, ignore_set=[0]):
        """ Goes through all training data to find temporal distances between each pair of labels

        ignore_set is a list of labels we don't want to model transitions between
            - This is for examining whether background class transitions should be included

        Returns
        -------
        temporal_dists: 2D numpy matrix of shape {num_labels, num_labels}
            - The value at index [a,b] is the distance between class a and b (1.0 when a == b)
        """
        # Only look at data not in the test set
        dataset = self.get_split("train_val")

        # Get all sequences in the set
        sequences = [[k for k, _ in itertools.groupby(d['labels']) if k not in ignore_set] for d in dataset]
        max_seq = min(max([len(s) for s in sequences]), 200)

        # Create transition length histograms
        def get_intervening_activity_count(sequence, activity_a, activity_b):
            """ Get a list of the minimum number of transitions between a pair of activities """
            activity_a_locations = [i for i,x in enumerate(sequence) if x == activity_a]
            activity_b_locations = [i for i,x in enumerate(sequence) if x == activity_b]
            counts = []
            for a_loc in activity_a_locations:
                intervening_acts = [np.abs(a_loc - b_loc) for b_loc in activity_b_locations]
                counts.append(min(intervening_acts))
            return counts
        transition_lengths = np.zeros((self.num_labels, self.num_labels, max_seq), dtype=int)
        for s in tqdm.tqdm(sequences, desc=f"Finding transition lengths for {self.name}"):
            unique_labels = sorted(set(s))
            for a,b in itertools.product(unique_labels, repeat=2):
                if a != b:
                    transitions = get_intervening_activity_count(s, a, b)
                    if min(transitions) >= 200:
                        continue
                    transition_lengths[a, b, min(transitions)] += 1

        # Get poisson parameters
        params = np.zeros((self.num_labels, self.num_labels))
        label_product = list(itertools.product(range(self.num_labels), repeat=2))
        for a, b in tqdm.tqdm(label_product, desc=f"Estimating Poisson params for {self.name}"):
            distance_hist = transition_lengths[a,b]
            if np.sum(distance_hist) == 0:
                params[a,b] = -1
            else:
                bins = np.arange(len(distance_hist))
                probs = distance_hist/np.sum(distance_hist)
                params[a,b] = np.sum(bins * probs)

        # Create distributions
        temporal_dists = np.zeros((self.num_labels, self.num_labels))
        for a_idx, x in enumerate(params):
            for b_idx, p in enumerate(x):
                if b_idx == a_idx:
                    dist = 1
                elif p == -1.0:
                    dist = 0
                else:
                    k = 1
                    dist = (np.exp(-1.0*p)*(p**k))/(np.math.factorial(k))
                temporal_dists[a_idx, b_idx] = dist

        return temporal_dists

    def convert_labels_to_object(self):
        all_objects = list(self.label_info["object"].unique())
        label_2_object = {i:all_objects.index(x['object']) for i,x in self.label_info.iterrows()}
        print(f"Converting labels to objects, new number of labels = {len(all_objects)}")
        print("Only works for 2-level object hierarchy right now. Need to change distances for ")
        self.distances['none'] = np.identity(len(all_objects))
        for x in self.dataset:
            x['labels'] = [label_2_object[i] for i in x['labels']]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class IKEA_ASM(DatasetClass):
    def __init__(self, root_dir):
        self.visual_relationships = ["verb", "object", "both"]
        super().__init__("IKEA ASM", root_dir)

    @property
    def label_info(self):
        return pd.read_csv(f"{FILE_DIR}/dataset_info/ikea_asm_labels.csv")

    @property
    def _dataset(self):
        # Load all splits
        train_vids = f"{self.root_dir}/indexing_files/train_cross_env.txt"
        test_vids = f"{self.root_dir}/indexing_files/test_cross_env.txt"
        train_files = open(train_vids, "r").read().splitlines()
        test_files = open(test_vids, "r").read().splitlines()
        np.random.seed(0)
        val_files = np.random.choice(train_files, int(len(train_files) * 0.15), replace=False)
        splits = {
            'train': sorted([x for x in train_files if x not in val_files]),
            'val': sorted([x for x in val_files]),
            'test': sorted([x for x in test_files]),
        }

        # Load labels
        label_file = f"{self.root_dir}/annotations/gt_action.npy"
        labels = np.load(label_file, allow_pickle=True).item()
        labels = {s: g for s, g in zip(labels['scan_name'], labels['gt_labels'])}

        # Create dataset
        dataset = []
        for split, files in splits.items():
            for file in files:
                dataset.append({
                    "video_name": file,
                    "split": split,
                    "frame_directory": f"{self.root_dir}/ANU_ikea_dataset_video_frames/{file}/dev3/images",
                    "labels": labels[file],
                    "meta_info": {
                        "furniture_type": file.split("/")[0],
                    }
                })
        return dataset

class EPIC_Kitchens(DatasetClass):
    def __init__(self, root_dir):
        self.visual_relationships = ["verb", "object", "both", "verb_2level", "object_2level"]
        self.labels_cache = f"{FILE_DIR}/dataset_info/epic_kitchens_labels.csv"
        self.dataset_cache = f"{FILE_DIR}/dataset_info/epic_kitchens_dataset.npz"
        super().__init__("EPIC KITCHENS", root_dir)

    @property
    def label_info(self):
        if os.path.exists(self.labels_cache):
            return pd.read_csv(self.labels_cache)
        else:
            labels = self.load_labels()
            labels.to_csv(self.labels_cache, index=False)
            return labels

    @property
    def _dataset(self):
        if os.path.exists(self.dataset_cache):
            return list(np.load(self.dataset_cache, allow_pickle=True)['a'])
        else:
            dataset = self.load_dataset()
            np.savez_compressed(self.dataset_cache, a=dataset)
            return dataset

    def load_labels(self):
        # Read label data
        verb_df = pd.read_csv(f"{FILE_DIR}/dataset_info/EPIC_100_verb_classes.csv")
        noun_df = pd.read_csv(f"{FILE_DIR}/dataset_info/EPIC_100_noun_classes.csv")
        verb_class2name = {i: row["key"] for i, row in verb_df.iterrows()}
        noun_class2name = {i: row["key"] for i, row in noun_df.iterrows()}
        verb_class2name_2level = {i: row["category"] for i, row in verb_df.iterrows()}
        noun_class2name_2level = {i: row["category"] for i, row in noun_df.iterrows()}

        # Read annotations to get actual list of labels
        train_df = pd.read_csv(f"{FILE_DIR}/dataset_info/EPIC_100_train.csv")
        val_df = pd.read_csv(f"{FILE_DIR}/dataset_info/EPIC_100_validation.csv")
        train_actions = list(train_df.apply(lambda x: f"{x['verb_class']}_{x['noun_class']}", axis=1))
        val_actions = list(val_df.apply(lambda x: f"{x['verb_class']}_{x['noun_class']}", axis=1))
        all_actions = set(train_actions + val_actions)

        # Create labels dataframe
        labels = [{'label': 'background', 'verb': 'none', 'object': 'none', 'verb_2level': 'none', 'object_2level': 'none', 'label_id': '-'}]
        for label_id in all_actions:
            verb_name = verb_class2name[int(label_id.split("_")[0])]
            object_name = noun_class2name[int(label_id.split("_")[1])]
            verb_2level_name = verb_class2name_2level[int(label_id.split("_")[0])]
            object_2level_name = noun_class2name_2level[int(label_id.split("_")[1])]
            labels.append({
                'label': f"{verb_name}_{object_name}",
                'label_id': label_id,
                'verb': verb_name,
                'object': object_name,
                'verb_2level': verb_2level_name,
                'object_2level': object_2level_name,
            })
        return pd.DataFrame(labels)

    def load_dataset(self):
        dataset = []
        label_id2value = {row['label_id']:idx for idx,row in self.label_info.iterrows()}
        train_df = pd.read_csv(f"{FILE_DIR}/dataset_info/EPIC_100_train.csv")
        val_df = pd.read_csv(f"{FILE_DIR}/dataset_info/EPIC_100_validation.csv")
        for df, split in [(train_df, "train"), (train_df, "val"), (val_df, "test")]:
            df = df.copy()
            # Set validation files from train set
            np.random.seed(0)
            all_vids = sorted(df["video_id"].unique())
            val_files = np.random.choice(all_vids, int(len(all_vids) * 0.15), replace=False)
            if split == "train":
                df = df[~df["video_id"].isin(val_files)]
            elif split == "val":
                df = df[df["video_id"].isin(val_files)]

            # Add column for label_id
            df["label_id"] = df.apply(lambda x: f"{x['verb_class']}_{x['noun_class']}", axis=1)
            df["label_value"] = df["label_id"].apply(lambda x: label_id2value[x])

            # Load dataset
            for vid, vid_df in df.groupby("video_id"):
                labels = np.zeros(len(os.listdir(f"{self.root_dir}/{vid}")), dtype=int)
                for _,row in vid_df.sort_values("start_frame").iterrows():
                    labels[row["start_frame"]:row["stop_frame"]] = row["label_value"]
                dataset.append({
                    "video_name": vid,
                    "split": split,
                    "frame_directory": f"{self.root_dir}/{vid}",
                    "labels": list(labels),
                    "meta_info": {},
                })
        return dataset

    def load_segments(self, split):
        df = pd.concat([pd.read_csv(f"{FILE_DIR}/dataset_info/EPIC_100_train.csv"),
                        pd.read_csv(f"{FILE_DIR}/dataset_info/EPIC_100_validation.csv")])
        label_id2value = {row['label_id']:idx for idx,row in self.label_info.iterrows()}
        df["label_id"] = df.apply(lambda x: f"{x['verb_class']}_{x['noun_class']}", axis=1)
        df["label_value"] = df["label_id"].apply(lambda x: label_id2value[x])
        segments = []
        for vid in self._dataset:
            if vid["split"] != split:
                continue
            for _,row in df[df["video_id"] == vid["video_name"]].sort_values("start_frame").iterrows():
                segments.append({
                    'vid': vid["video_name"],
                    'split': vid["split"],
                    'start_frame': row["start_frame"],
                    'stop_frame': row["stop_frame"],
                    'label': row["label_value"]
                })
        return segments

class FineGym(DatasetClass):
    def __init__(self, root_dir, full_288_set=False):
        self.visual_relationships = ["verb", "verb_2level"]
        self.full_288_set = full_288_set
        self._labels = self.load_labels()
        super().__init__("FineGym", root_dir)

    @property
    def label_info(self):
        return self._labels

    def load_labels(self):
        num_classes = 288 if self.full_288_set else 99
        df_cat = pd.read_csv(f"/home/mike/Projects/Fine_GYM/annotations/gym{num_classes}_categories.txt",
            sep=";", names=["element_class", "set", "element", "event"]
        )
        df_cat["element_class"] = df_cat["element_class"].apply(lambda x: int(x.split(" ")[-1]))
        df_cat["set"] = df_cat["set"].apply(lambda x: x.split(" ")[-1])
        df_cat["element"] = df_cat["element"].apply(lambda x: x.split(" ")[-1])
        df_cat["event"] = df_cat["event"].apply(lambda x: x[2:4])
        df_cat = df_cat.rename(columns={"element_class": "label", "set": "verb", "event": "verb_2level"})
        df_cat = df_cat.drop(columns="element")
        return df_cat

    @property
    def _dataset(self):
        num_classes = 288 if self.full_288_set else 99
        annotations = {
            'train': f"{self.root_dir}/annotations/gym{num_classes}_train_element_v1.1.txt",
            'test': f"{self.root_dir}/annotations/gym{num_classes}_val_element.txt"
        }
        feats = {
            'train': pickle.load(open(f"{self.root_dir}/data/gym{num_classes}_train_feat_i3d-kin_12_2048.pkl", "rb")),
            'test': pickle.load(open(f"{self.root_dir}/data/gym{num_classes}_val_feat_i3d-kin_12_2048.pkl", "rb"))
        }

        # Read in annotations
        df_train = pd.read_csv(annotations['train'], sep=" ", names=["video_name", "label"])
        df_test = pd.read_csv(annotations['test'], sep=" ", names=["video_name", "label"])
        df_train = df_train[df_train["video_name"].isin(list(feats['train'].keys()))]
        df_test = df_test[df_test["video_name"].isin(list(feats['test'].keys()))]
        df_train["video_id"] = df_train["video_name"].apply(lambda x: x.split("_")[0])

        # Split
        df_train["split"] = "train"
        np.random.seed(0)
        unique_ids = list(df_train["video_id"].unique())
        ids_for_val = np.random.choice(unique_ids, int(len(unique_ids)*.1), replace=False)
        df_train.loc[df_train["video_id"].isin(ids_for_val), "split"] = "val"
        df_test["split"] = "test"
        df = pd.concat([df_train, df_test])

        # Create dataset
        dataset = []
        for _,row in df.iterrows():
            dataset.append({
                "video_name": row["video_name"],
                "split": row["split"],
                "frame_directory": "N/A",
                "labels": row["label"],
                "features": np.squeeze(feats["test"][row["video_name"]] if row["split"] == "test" else feats["train"][row["video_name"]]),
                "frames": 0,
                "meta_info": {}
            })
        return dataset


class MHAD(DatasetClass):
    # first 7 subjects for training and last 5 subjects for testing
    def __init__(self, root_dir):
        self.info = {
            'dataset_name': 'Berkeley MHAD',
            'dataset_type': 'single_action',
        }
        self.visual_relationships = ["verb", "verb_2level"]
        super().__init__("MHAD", root_dir)

    @cached_property
    def label_info(self):
        # Todo - use hierarchy from "Extracting Action Hierarchies from Action Labels", slightly different
        labels = [
            {'label_id': 'A01', 'label': 'Jumping in place',
             'verb': 'jumping', 'verb_2level': 'upper_lower'},
            {'label_id': 'A02', 'label': 'Jumping jacks',
             'verb': 'jumping', 'verb_2level': 'upper_lower'},
            {'label_id': 'A03', 'label': 'Bending - hands up all the way down',
             'verb': 'bending', 'verb_2level': 'upper'},
            {'label_id': 'A04', 'label': 'Punching(boxing)',
             'verb': 'punching', 'verb_2level': 'upper_lower'},
            {'label_id': 'A05', 'label': 'Waving - two hands',
             'verb': 'waving', 'verb_2level': 'upper'},
            {'label_id': 'A06', 'label': 'Waving - one hand(right)',
             'verb': 'waving', 'verb_2level': 'upper'},
            {'label_id': 'A07', 'label': 'Clapping hands',
             'verb': 'clapping', 'verb_2level': 'upper'},
            {'label_id': 'A08', 'label': 'Throwing a ball',
             'verb': 'throwing', 'verb_2level': 'upper_lower'},
            {'label_id': 'A09', 'label': 'Sit down then stand up',
             'verb': 'sit_stand', 'verb_2level': 'lower'},
            {'label_id': 'A10', 'label': 'Sit down',
             'verb': 'sit', 'verb_2level': 'lower'},
            {'label_id': 'A11', 'label': 'Stand up',
             'verb': 'stand', 'verb_2level': 'lower'},
        ]
        return pd.DataFrame(labels)

    @cached_property
    def _dataset(self):
        # Create dataset
        subjects = [f"S{s:02d}" for s in range(1, 13)]
        actions = [f"A{s:02d}" for s in range(1, 12)]
        splits = {
            'S01': 'train', 'S02': 'train', 'S03': 'train', 'S04': 'train', 'S05': 'train',
            'S06': 'train', 'S07': 'train',
            'S08': 'test', 'S09': 'test', 'S10': 'test', 'S11': 'test', 'S12': 'test',
        }
        label_id2value = {x['label_id']:idx for idx,x in self.label_info.iterrows()}
        dataset = []
        for subject in subjects:
            for action in actions:
                for repetition in range(1,6):
                    frame_dir = f"{self.root_dir}/Camera/Cluster01/Cam01/{subject}/{action}/R{repetition:02d}"
                    if not os.path.isdir(frame_dir):
                        continue
                    dataset.append({
                        "video_name": f"{subject}_{action}_r{repetition:02d}".lower(),
                        "split": splits[subject],
                        "frame_directory": frame_dir,
                        "labels": [label_id2value[action] for _ in range(len(os.listdir(frame_dir)))],
                        "meta_info": {},
                    })
        return dataset
