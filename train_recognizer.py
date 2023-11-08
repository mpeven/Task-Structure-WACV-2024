import os
import glob
import argparse
import json
import tqdm
import numpy as np
import pandas as pd
import torch

# Local imports
from datasets import get_dataset
from loss_function import DistanceLoss

# Global variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    parser = argparse.ArgumentParser(description='Train an RNN')
    parser.add_argument('-v', '--video_dataset', type=str, required=True,
                        choices=['IKEA_ASM', 'EPIC_Kitchens', 'FineGym'],
                        help='Which dataset to use')
    parser.add_argument('-m', '--model_type', type=str, required=True,
                        choices=['LSTM', 'GRU'],
                        help='Which model to use for training')
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
    parser.add_argument('-f', '--feature_dir', type=str, required=True,
                        help="Directory where saved image features are located")
    parser.add_argument('--results_file', type=str, default="", help="File to save results")
    parser.add_argument('--trial', type=int, default=1, help="Trial # for averaging results")
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='How many epochs to keep training after validation loss has reached a minimum')
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--hidden_dim_size', default=256, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    args = parser.parse_args()
    return args

# Basic dataset for use with dataloaders
class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, distance_type, split, feature_dir):
        self.distances = dataset.distances[distance_type]
        self.dataset = dataset.get_split(split).copy()
        all_features = np.load(os.path.join(feature_dir, "saved_features.npy"))
        features_info = pd.read_csv(os.path.join(feature_dir, "saved_features_info.csv"))
        for vid in self.dataset:
            vid_features_info = features_info[features_info["video"] == vid["video_name"]]
            vid_features_info = vid_features_info.sort_values("frame")
            feature_indices = vid_features_info["array_index"].values
            vid["frames"] = vid_features_info["frame"].tolist()
            vid["features"] = np.copy(all_features[feature_indices])
            vid["labels"] = np.stack([vid["labels"][x] for x in vid["frames"]])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        to_return = self.dataset[idx].copy()
        to_return["distances"] = np.stack([self.distances[l] for l in to_return["labels"]])
        return to_return


class DatasetBasic(torch.utils.data.Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


class Model(torch.nn.Module):
    def __init__(self, model_type, num_inputs, num_outputs, hidden_dim_size, num_layers, dropout, sequence=True):
        super(Model, self).__init__()
        if model_type == "LSTM":
            self.rnn = torch.nn.LSTM(
                input_size=num_inputs,
                hidden_size=hidden_dim_size//2,
                num_layers=num_layers,
                dropout=0.0,
                batch_first=True,
                bidirectional=True,
            )
        elif model_type == "GRU":
            self.rnn = torch.nn.GRU(
                input_size=num_inputs,
                hidden_size=hidden_dim_size//2,
                num_layers=num_layers,
                dropout=0.5,
                batch_first=True,
                bidirectional=True,
            )
        self.pre_logits_dropout = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.Linear(hidden_dim_size, num_outputs)
        self.sequence = sequence

    def forward(self, padded_inputs, seq_lens):
        if not self.sequence:
            out, (hidden, _) = self.rnn(padded_inputs)
            # out, hidden = self.rnn(padded_inputs)
            # print(out.size())
            # print(hidden.size())
            # exit()
            # final_feats = out[:, 6, :]
            final_feats = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            return self.linear(self.pre_logits_dropout(final_feats)).squeeze()
        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(
            padded_inputs.float(), seq_lens, batch_first=True, enforce_sorted=False
        )
        feats, _ = self.rnn(packed_inputs)
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(feats, batch_first=True)
        padded_preds = self.linear(self.pre_logits_dropout(unpacked))
        return padded_preds


def epoch(test_mode, model, dataloader, optimizer, loss_func, sequence=True):
    model = model.train(not test_mode)
    loss_func = loss_func.train(not test_mode)

    iterator = tqdm.tqdm(dataloader, ncols=150, desc=f"{'Evaluation' if test_mode else 'Train'} epoch")
    running_loss, running_videos, running_preds, running_labels, running_frames, running_outputs = [], [], [], [], [], []

    with torch.inference_mode(test_mode):
        for batch in iterator:
            # Forward pass
            optimizer.zero_grad()
            if sequence:
                outputs = model(batch['features'].to(DEVICE), batch['seq_lens'])

                # Calculate loss
                loss = loss_func(
                    torch.cat([o[:s] for o, s in zip(outputs, batch['seq_lens'])]),
                    torch.cat([o[:s] for o, s in zip(batch['labels'].to(DEVICE), batch['seq_lens'])]),
                    torch.cat([o[:s] for o, s in zip(batch['distances'].to(DEVICE), batch['seq_lens'])]),
                )
                running_loss.append(loss.item())

                if not test_mode:
                    loss.backward()
                    optimizer.step()

                running_preds += [
                    torch.argmax(o[:s], 1).cpu().detach().numpy().astype(int).tolist()
                    for o, s in zip(outputs, batch['seq_lens'])
                ]
                running_labels += [
                    l[:s].cpu().detach().numpy().astype(int).tolist()
                    for l, s in zip(batch['labels'], batch['seq_lens'])
                ]

                # Get other info if evaluating
                if test_mode:
                    running_outputs += list(batch['video_name'])
                    # running_outputs += [
                    #     o[:s].cpu().detach().numpy().tolist()
                    #     for o, s in zip(outputs, batch['seq_lens'])
                    # ]
                    running_frames += [
                        l[:s].cpu().detach().numpy().astype(int).tolist()
                        for l, s in zip(batch['frames'], batch['seq_lens'])
                    ]
                    running_videos += list(batch['video_name'])
            else:
                outputs = model(batch['features'].to(DEVICE), None)
                loss = loss_func(outputs, batch['labels'].to(DEVICE), batch['distances'].to(DEVICE))
                running_loss.append(loss.item())
                if not test_mode:
                    loss.backward()
                    optimizer.step()
                running_preds += torch.argmax(outputs, 1).cpu().detach().numpy().astype(int).tolist()
                running_labels += batch['labels'].cpu().detach().numpy().astype(int).tolist()
                if test_mode:
                    running_outputs += list(batch['video_name'])
                    running_frames += batch['frames'].cpu().detach().numpy().astype(int).tolist()
                    running_videos += list(batch['video_name'])
            # Create progress-bar display
            distance_vals = loss_func.get_distance_vals_4display()
            postfix = {
                "Loss": f"{np.mean(running_loss):.3f}",
                "Acc": f"{100*np.mean([np.mean(np.array(p) == np.array(l)) for p, l in zip(running_preds, running_labels)]):.2f}",
                "D1": f"{distance_vals[0]:.5f}",
                "D2": f"{distance_vals[1]:.5f}",
                "DT": f"{distance_vals[2]:.5f}",
            }
            iterator.set_postfix(postfix)

    to_return = {'avg_loss': np.mean(running_loss)}
    if test_mode:
        to_return['avg_acc'] = 100*np.mean([np.mean(np.array(p) == np.array(l))
                                           for p, l in zip(running_preds, running_labels)])
        to_return['all_results'] = [{
            'video_name': str(t), 'frames': f, 'preds': p, 'labels': l, 'outputs': o
        } for t, f, p, l, o in zip(running_videos, running_frames, running_preds, running_labels, running_outputs)]
    to_return['distances'] = {
        'distance_1': distance_vals[0], 'distance_2': distance_vals[1], 'distance_t': distance_vals[2],
    }
    return to_return

def get_dataset_for_rnn(dataset, distance_type, split, feature_dir):
    if dataset.name.lower() in ["epic kitchens", "ikea asm"]:
        return Dataset(dataset, distance_type, split, feature_dir)
    elif dataset.name.lower() == "finegym":
        data_copy = dataset.get_split(split).copy()
        for vid in data_copy:
            vid["distances"] = dataset.distances[distance_type][vid["labels"]]
        return DatasetBasic(data_copy)

def get_collate_fn(dataset_name):
    def collate_fn1(batch):
        return {
            'video_name': [x['video_name'] for x in batch],
            'seq_lens': [len(x['distances']) for x in batch],
            'frames': torch.nn.utils.rnn.pad_sequence([torch.tensor(x['frames']) for x in batch], batch_first=True),
            'distances': torch.nn.utils.rnn.pad_sequence([torch.tensor(x['distances']) for x in batch], batch_first=True),
            'features': torch.nn.utils.rnn.pad_sequence([torch.tensor(x['features']) for x in batch], batch_first=True),
            'labels': torch.nn.utils.rnn.pad_sequence([torch.tensor(x['labels']) for x in batch], batch_first=True),
        }
    if dataset_name.lower() in ["epic_kitchens", "ikea_asm"]:
        return collate_fn1
    elif dataset_name.lower() == "finegym":
        return None

def main():
    args = get_args()

    dataset = get_dataset(args.video_dataset)
    sequence = False if args.video_dataset.lower() == "finegym" else True
    dataloader_args = {'batch_size': args.batch_size, 'collate_fn': get_collate_fn(args.video_dataset),
                       'num_workers': os.cpu_count(), 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        get_dataset_for_rnn(dataset, args.distance_type, "train", args.feature_dir), shuffle=True, **dataloader_args
    )
    val_loader = torch.utils.data.DataLoader(
        get_dataset_for_rnn(dataset, args.distance_type, "val", args.feature_dir), shuffle=False, **dataloader_args
    )
    model = Model(
        model_type=args.model_type,
        num_inputs=2048,
        num_outputs=dataset.num_labels,
        hidden_dim_size=args.hidden_dim_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        sequence=sequence
    ).to(DEVICE)
    loss_func = DistanceLoss(distance_type=args.distance_type, d1=args.distance1,
                             d2=args.distance2, dt=args.distance_temporal,
                             learnable=args.distance_learnable).to(DEVICE)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(loss_func.parameters()), lr=args.learning_rate)

    # Train
    np.random.seed()
    weights_file = f"tmp_weights_{np.random.randint(10000000)}.pt"
    best_val_epoch = 0
    best_val_loss = np.inf
    train_val_info = []

    for epoch_idx in range(100):
        print(f"Epoch {epoch_idx}")
        train_results = epoch(False, model, train_loader, optimizer, loss_func, sequence)
        val_results = epoch(True, model, val_loader, optimizer, loss_func, sequence)
        train_val_info.append({
            'epoch': epoch_idx,
            'train_loss': train_results['avg_loss'],
            'val_loss': val_results['avg_loss'],
            'val_acc': val_results['avg_loss'],
            **train_results['distances'],
        })
        if val_results['avg_loss'] < best_val_loss:
            best_val_loss = val_results['avg_loss']
            best_val_epoch = epoch_idx
            torch.save(model.state_dict(), weights_file)
        if (epoch_idx - best_val_epoch) == args.early_stopping_patience:  # Early stopping
            break

    # Evaluate on test set and save out results
    test_loader = torch.utils.data.DataLoader(
        get_dataset_for_rnn(dataset, args.distance_type, "test", args.feature_dir), shuffle=False, **dataloader_args
    )
    model.load_state_dict(torch.load(weights_file))
    test_results = epoch(True, model, test_loader, optimizer, loss_func, sequence)
    os.remove(weights_file)
    print(f"Test set loss: {test_results['avg_loss']:.3f}")
    print(f"Test set accuracy: {test_results['avg_acc']:.3f}")
    to_save = {
        'dataset': args.video_dataset,
        'trial': args.trial,
        'tsm_hyperparams': {
            'feature_dir': args.feature_dir,
        },
        'rnn_hyperparams': {
            'distance_type': args.distance_type,
            'distance_level1': args.distance1,
            'distance_level2': args.distance2,
            'distance_temporal': args.distance_temporal,
            'distance_learnable': args.distance_learnable,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'model_type': args.model_type,
            'hidden_dim_size': args.hidden_dim_size,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
        },
        'training_info': train_val_info,
        'test_results': test_results,
    }
    if args.results_file != "":
        os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
        np.save(args.results_file, to_save)

if __name__ == "__main__":
    main()
