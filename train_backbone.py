import os
import glob
import argparse
import tqdm
import numpy as np
import pandas as pd
import cv2
import albumentations
import torch

# Local imports
from model_tsm import TSM
from datasets import get_dataset
from loss_function import DistanceLoss

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Global variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    parser = argparse.ArgumentParser(description='Train a TSM for video classification')

    # Training options
    parser.add_argument('-v', '--video_dataset', type=str, required=True,
                        choices=['IKEA_ASM', 'EPIC_Kitchens', 'FineGym', 'MHAD'],
                        help='Which dataset to use')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='How many epochs to keep training after validation loss has reached a minimum')
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=12)

    # Loss function options
    parser.add_argument('-d', '--distance_type', type=str, required=True,
                        choices=["none", "both", "temporal", "verb", "object", "verb_object", "furniture",
                                 "verb_2level", "object_2level", "temporal_object", "temporal_verb", "temporal_both",
                                 "temporal_verb_2level"],
                        help='The distance function to use')
    parser.add_argument('-d1', '--distance1', type=float, required=False, default=0.0,
                        help='The distance weight for nodes with shared parents')
    parser.add_argument('-d2', '--distance2', type=float, required=False, default=0.0,
                        help='The distance weight for nodes with shared grandparents')
    parser.add_argument('-dt', '--distance_temporal', type=float, required=False, default=0.0,
                        help='The distance weight for temporal loss')
    parser.add_argument('-dl', '--distance_learnable', required=False, action='store_true',
                        help='Learn distances during training')

    ### TSM options
    parser.add_argument('--dense_sample', action='store_true', help='Use dense sampling with the TSM')
    parser.add_argument('--tsm_segments', type=int, default=8, help='Number of tsm segments')

    ### Other options
    parser.add_argument('-s', '--suffix', type=str, default="", required=False,
                        help='Suffix to add to saved weight file')
    parser.add_argument('--generate_features', action='store_true',
                        help='Whether to generate features')

    args = parser.parse_args()
    return args


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, distance_type, split, tsm_segments=8, dense_sample=False, fixed_num_windows=None):
        self.tsm_segments = tsm_segments
        self.dense_sample = dense_sample
        self.split = split
        self.dataset = self.build_dataset(dataset_name, distance_type, split, fixed_num_windows)
        if split == "train":
            self.augmentations = albumentations.load("augmentation_configs/imagenet_autoalbument.json")
        else:
            self.augmentations = albumentations.load("augmentation_configs/imagenet_val.json")
        self.augmentations.add_targets({f'image{i:02d}': 'image' for i in range(tsm_segments-1)})

    def build_dataset(self, dataset_name, distance_type, split, fixed_num_windows):
        dataset_videos = get_dataset(dataset_name)
        self.num_labels = dataset_videos.num_labels
        dataset_videos.create_windows(window_stride=None, fixed_num_windows=fixed_num_windows)
        distances = dataset_videos.distances[distance_type]

        dataset_clips = []
        for vid in dataset_videos.dataset:
            if vid['split'] != split and self.split != 'all':
                continue
            frames = sorted(glob.glob(vid['frame_directory'] + "/*"))
            for window in vid['windows']:
                dataset_clips.append({
                    'image_files': frames[window['start_idx']:window['stop_idx']+1],
                    'label': window['label'],
                    'distances': distances[window['label']],
                    'video_name': window['video_name'].replace("/", "_"),
                    'frame': window['window_center_frame'],
                })

        return dataset_clips

    def __len__(self):
        return len(self.dataset)

    def load_images(self, file_list):
        image_dict = {
            'image': cv2.cvtColor(cv2.imread(file_list[0]), cv2.COLOR_BGR2RGB),
            **{f'image{i:02d}': cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
            for i, f in enumerate(file_list[1:])}
        }
        transformed = self.augmentations(**image_dict)
        return torch.concat([transformed[k] for k in image_dict.keys()], dim=0)

    def sample_indices(self, num_frames):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + num_frames - 64)
            t_stride = 64 // self.tsm_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % num_frames for idx in range(self.tsm_segments)]
            return np.array(offsets) + 1
        else:
            average_duration = num_frames // self.tsm_segments
            if not (average_duration > 0):
                print("SAMPLING ERROR: FIX THIS --> not (average_duration > 0)")
            offsets = np.multiply(list(range(self.tsm_segments)), average_duration) + np.random.randint(average_duration, size=self.tsm_segments)
            return offsets + 1

    def sample_val_indices(self, num_frames):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + num_frames - 64)
            t_stride = 64 // self.tsm_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % num_frames for idx in range(self.tsm_segments)]
            return np.array(offsets) + 1
        else:
            if num_frames > self.tsm_segments:
                tick = (num_frames) / float(self.tsm_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.tsm_segments)])
            else:
                offsets = np.zeros((self.tsm_segments,))
            return offsets + 1

    def sample_test_indices(self, num_frames):
        if self.dense_sample:
            sample_pos = max(1, 1 + num_frames - 64)
            t_stride = 64 // self.tsm_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % num_frames for idx in range(self.tsm_segments)]
            return np.array(offsets) + 1
        # elif self.twice_sample:
        #     tick = (num_frames) / float(self.tsm_segments)
        #     offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.tsm_segments)] +
        #                        [int(tick * x) for x in range(self.tsm_segments)])
        #     return offsets + 1
        else:
            tick = (num_frames) / float(self.tsm_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.tsm_segments)])
            return offsets + 1

    def __getitem__(self, idx):
        to_return = self.dataset[idx].copy()
        if self.split == "train":
            sampled_indices = self.sample_indices(len(to_return['image_files']))
        elif self.split == "val":
            sampled_indices = self.sample_val_indices(len(to_return['image_files']))
        else:
            sampled_indices = self.sample_test_indices(len(to_return['image_files']))
        image_list = [to_return['image_files'][i] for i in sampled_indices]
        to_return['images'] = self.load_images(image_list)
        return to_return


def epoch(test_mode, model, dataloader, optimizer, loss_func):
    model = model.train(not test_mode)
    loss_func = loss_func.train(not test_mode)

    iterator = tqdm.tqdm(dataloader, ncols=150, desc=f"{'Evaluation' if test_mode else 'Train'} epoch")
    running_loss, running_preds, running_labels = [], [], []

    with torch.inference_mode(test_mode):
        for batch in iterator:
            outputs = model(batch['images'].to(DEVICE))
            loss = loss_func(outputs, batch['label'].to(DEVICE), batch['distances'].to(DEVICE))
            # loss = loss_func(outputs, batch['label'].to(DEVICE))
            running_loss.append(loss.item())
            if not test_mode:
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
                optimizer.step()
                optimizer.zero_grad()
            running_preds += torch.argmax(outputs, 1).cpu().detach().numpy().astype(int).tolist()
            running_labels += batch['label'].cpu().detach().numpy().astype(int).tolist()
            distance_vals = loss_func.get_distance_vals_4display()
            postfix = {
                "Loss": f"{np.mean(running_loss):.3f}",
                "Acc": f"{100*np.mean(np.array(running_preds) == np.array(running_labels)):.2f}",
                "D1": f"{distance_vals[0]:.5f}",
                "D2": f"{distance_vals[1]:.5f}",
                "DT": f"{distance_vals[2]:.5f}",
            }
            iterator.set_postfix(postfix)
    return {
        'avg_loss': np.mean(running_loss),
        'avg_acc': 100*np.mean(np.array(running_preds) == np.array(running_labels))
    }

def generate_features(model, dataloader, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model = model.train(False)
    cur_vid = ""
    vid_feats = []
    with torch.inference_mode(True):
        for batch in tqdm.tqdm(dataloader, ncols=150, desc="Generating features"):
            outputs = model(batch['images'].to(DEVICE), features_only=True)
            for vid_id, frame_feats in zip(batch['video_name'], outputs.cpu().detach().numpy()):
                if vid_id == cur_vid or cur_vid == "":
                    vid_feats.append(frame_feats)
                    cur_vid = vid_id
                else:
                    lines = []
                    line0 = ','.join([f'f{i}' for i in range(2048)])
                    lines.append(line0)
                    for line in vid_feats:
                        lines.append(','.join([f'{x:.4f}' for x in line]))
                    with open(save_dir + f"/{cur_vid}.csv", 'w') as f:
                        f.write('\n'.join(lines))
                    cur_vid = vid_id
                    vid_feats = [frame_feats]

def main():
    args = get_args()
    dataloader_args = {'batch_size': args.batch_size, 'num_workers': os.cpu_count(), 'pin_memory': True}
    train_set = Dataset(args.video_dataset, args.distance_type, 'train', args.tsm_segments, args.dense_sample)
    val_set = Dataset(args.video_dataset, args.distance_type, 'val', args.tsm_segments, args.dense_sample)
    # subset_sampler = torch.utils.data.sampler.RandomSampler(train_set, num_samples=len(train_set)//20)
    # subset_sampler_val = torch.utils.data.sampler.RandomSampler(val_set, num_samples=len(val_set)//3)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **dataloader_args)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **dataloader_args)
    model = TSM(train_set.num_labels, num_segments=args.tsm_segments).to(DEVICE)
    loss_func = DistanceLoss(distance_type=args.distance_type, d1=args.distance1,
                             d2=args.distance2, dt=args.distance_temporal,
                             learnable=args.distance_learnable).to(DEVICE)
    optimizer = torch.optim.RAdam(list(model.base_model.parameters()) + list(loss_func.parameters()), lr=args.learning_rate)
    optimizer.add_param_group({'params': model.classifier.parameters(), 'lr': args.learning_rate*5})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.early_stopping_patience//2)

    # Train
    np.random.seed()
    weights_file = f"tsm_weights_{args.video_dataset}_{args.distance_type}"
    if args.distance_type != "none":
        weights_file += f"_d1{args.distance1}"
    if "2level" in args.distance_type:
        weights_file += f"_d2{args.distance2}"
    if "temporal" in args.distance_type:
        weights_file += f"_dt{args.distance_temporal}"
    if args.suffix != "":
        weights_file += f"_{args.suffix}"
    weights_file += ".pt"
    best_val_epoch = 0
    best_val_loss = np.inf
    train_val_info = []

    print("\n\n\n\n")
    print("Saving to ", weights_file)
    print()
    for epoch_idx in range(100):
        print(f"Epoch {epoch_idx}")
        train_results = epoch(False, model, train_loader, optimizer, loss_func)
        val_results = epoch(True, model, val_loader, optimizer, loss_func)
        scheduler.step(val_results['avg_loss'])
        train_val_info.append({
            'epoch': epoch_idx,
            'train_loss': train_results['avg_loss'],
            'val_loss': val_results['avg_loss'],
            'val_acc': val_results['avg_acc'],
            # **train_results['distances'],
        })
        if val_results['avg_loss'] < best_val_loss:
            best_val_loss = val_results['avg_loss']
            best_val_epoch = epoch_idx
            torch.save(model.state_dict(), weights_file)
        if (epoch_idx - best_val_epoch) == args.early_stopping_patience:  # Early stopping
            break

    # Evaluate on test set and save out results
    test_loader = torch.utils.data.DataLoader(
        Dataset(args.video_dataset, args.distance_type, 'test', args.tsm_segments, args.dense_sample), shuffle=False, **dataloader_args
    )
    model.load_state_dict(torch.load(weights_file))
    test_results = epoch(True, model, test_loader, optimizer, loss_func)

    if args.generate_features:
        test_loader = torch.utils.data.DataLoader(
            Dataset(args.video_dataset, args.distance_type, 'all', args.tsm_segments, args.dense_sample, fixed_num_windows=100), shuffle=False, **dataloader_args
        )
        save_dir = "saved_features/" + weights_file.replace(".pt", "")
        generate_features(model, test_loader, save_dir)

if __name__ == "__main__":
    main()
