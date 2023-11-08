import functools
import glob
import re
from itertools import groupby

import Levenshtein
import numpy as np
import pandas as pd
import tqdm
from scipy import stats

from datasets import get_dataset

SORT_KEYS = ["none", "object", "verb", "both", "object_2level", "verb_2level", "temporal", "temporal_object", "temporal_verb"]
pd.set_option("display.max_rows", None)


def get_edit_distance(labels, preds):
    """ Convert to strings and get edit distance """
    labels_grouped = [k for k,_ in groupby(labels)]
    preds_grouped = [k for k,_ in groupby(preds)]
    max_len = max(len(labels_grouped), len(preds_grouped))
    return 100*(1 - Levenshtein.distance(labels_grouped, preds_grouped)/max_len)


def get_accuracy(labels, preds):
    return 100*np.mean(np.equal(preds, labels))


def get_segments(preds, frames):
    # frame_diff = 60 # Median activity length in epic kitchens
    frame_diff = frames[1] - frames[0]
    segments = []
    idx = 0
    for label,group in groupby(preds):
        seg_size = len(list(group))
        segments.append({
            'label': label,
            'start_frame': frames[idx] - frame_diff//2,
            'stop_frame': frames[idx+seg_size-1] + frame_diff//2,
        })
        idx += seg_size
    segments = [x for x in segments if x['label'] != 0]
    return segments


def get_map_values(gt_segments, pred_segments, convert_map={}):
    df_gt = pd.DataFrame(gt_segments)
    df_pred = pd.DataFrame(pred_segments)
    if len(convert_map) > 0:
        df_gt["label"] = df_gt["label"].map(convert_map)
        df_pred["label"] = df_pred["label"].map(convert_map)
    df_merge = df_gt.merge(df_pred, how="left", on=["vid", "label"], suffixes=["_gt", "_pred"])
    df_merge["intersection"] = np.maximum(
        np.minimum(df_merge["stop_frame_gt"], df_merge["stop_frame_pred"]) - np.maximum(df_merge["start_frame_gt"], df_merge["start_frame_pred"]),
        0
    )
    df_merge["union"] = df_merge["stop_frame_gt"] - df_merge["start_frame_gt"] + df_merge["stop_frame_pred"] - df_merge["start_frame_pred"] - df_merge["intersection"]
    df_merge["iou"] = df_merge["intersection"]/df_merge["union"]
    ious = df_merge.groupby(["vid", "start_frame_gt", "stop_frame_gt"])["iou"].max().values
    maps = {f'map_{x:.01f}':np.mean(ious > x) for x in [0.1, 0.2, 0.3, 0.4, 0.5]}
    maps['map_avg'] = np.mean(list(maps.values()))
    return maps


def load_experiment_results(experiment, dataset_name):
    experiment_files = sorted(glob.glob(f"results/{experiment}/*"))

    # Load dataset
    dataset = get_dataset(dataset_name)
    if dataset_name.lower() == "epic_kitchens":
        gt_segments = dataset.load_segments("test")

    # Load results from all experiments
    classified_correctly = []
    experiments = []
    for f in tqdm.tqdm(experiment_files, desc="Loading in results"):
        results_dict = np.load(f, allow_pickle=True).item()

        # Get basic hyperparameters
        experiment_info = {
            'dataset': results_dict['dataset'],
            'trial': results_dict['trial'],
            **{f'rnn_{k}': v for k, v in results_dict['rnn_hyperparams'].items()},
            'accuracy': results_dict['test_results']['avg_acc'],
            'loss': results_dict['test_results']['avg_loss'],
        }
        # if "tsm_hyperparams" in results_dict:
        #     experiment_info.update({f'tsm_{k}': v for k, v in results_dict['tsm_hyperparams'].items()})

        # if 'tsm_distance_type' not in experiment_info:
        #     try:
        #         experiment_info['tsm_distance_type'] = re.search(r'tsm_(.*?)_d1', experiment_info['rnn_feature_dir']).group(1)
        #     except Exception:
        #         experiment_info['tsm_distance_type'] = experiment_info['tsm_feature_dir'].replace("features/epic_", "")

        # Get per video data
        pred_segments = []
        experiment_info['per_video_data'] = []

        if dataset_name.lower() == "finegym":
            labels = [vid['labels'] for vid in results_dict['test_results']['all_results']]
            preds = [vid['preds'] for vid in results_dict['test_results']['all_results']]
            class_accuracies = []
            for c in set(labels):
                mask = (np.array(labels) == c)
                class_accuracies.append(get_accuracy(np.array(labels)[mask], np.array(preds)[mask]))
            experiment_info["class_accuracy"] = np.mean(class_accuracies)
        else:
            classified_correctly_per_vid = []
            for vid in results_dict['test_results']['all_results']:
                per_video_data = {
                    'video_name': vid['video_name'],
                    'accuracy': get_accuracy(vid['labels'], vid['preds']),
                    'edit_distance': get_edit_distance(vid['labels'], vid['preds']),
                }
                visual_relationships = [x for x in dataset.visual_relationships if x != "both"]
                for cat in visual_relationships:
                    label_map = dict(zip(dataset.label_info["label"].astype('category').cat.codes,
                                        dataset.label_info[cat].astype('category').cat.codes))
                    new_labels = np.array([label_map[x] for x in vid['labels']])
                    new_preds = np.array([label_map[x] for x in vid['preds']])
                    per_video_data[cat + '_accuracy'] = get_accuracy(new_labels, new_preds)
                experiment_info['per_video_data'].append(per_video_data)

                classified_correctly_per_vid.append(np.equal(vid['preds'], vid['labels']))

                if dataset_name.lower() == "epic_kitchens":
                    vid_pred_segments = get_segments(vid['preds'], vid['frames'])
                    pred_segments += [{"vid": vid["video_name"], "split":"test", **d} for d in vid_pred_segments]

            # Average results
            avg_keys = [f'{x}_accuracy' for x in visual_relationships] + ['edit_distance']
            for key in avg_keys:
                experiment_info[key] = np.mean([x[key] for x in experiment_info['per_video_data']])

        # Get training info
        # experiment_info['train_loss'] = [e['train_loss'] for e in results_dict['training_info']]
        # experiment_info['val_loss'] = [e['val_loss'] for e in results_dict['training_info']]

        # Get map values
        if dataset_name.lower() == "epic_kitchens":
            maps = get_map_values(gt_segments, pred_segments)
            experiment_info.update(maps)
            for cat in ["object", "verb"]:
                label_map = dict(zip(dataset.label_info["label"].astype('category').cat.codes,
                                     dataset.label_info[cat].astype('category').cat.codes))
                maps = get_map_values(gt_segments, pred_segments, label_map)
                experiment_info.update(**{f"{cat}_{k}":v for k,v in maps.items()})

        experiment_info.pop('per_video_data')
        experiments.append(experiment_info)
        classified_correctly.append(list(classified_correctly_per_vid))


    return pd.DataFrame(experiments), classified_correctly


def backbone_exp(df):
    results = []
    map_results = []
    for dist_type,sub_df in df.groupby("tsm_distance_type"):
        results.append({
            'Distance Function': dist_type,
            'Accuracy': f"{sub_df.accuracy.mean():.02f}   {sub_df.accuracy.std():.02f}",
            'Edit Distance': f"{sub_df.edit_distance.mean():.02f}  {sub_df.edit_distance.std():.02f}",
            'Verb Accuracy': f"{sub_df.verb_accuracy.mean():.02f}  {sub_df.verb_accuracy.std():.02f}",
            'Object Accuracy': f"{sub_df.object_accuracy.mean():.02f}  {sub_df.object_accuracy.std():.02f}",
        })
        if df.iloc[0]["dataset"].lower() == "epic_kitchens":
            results[-1].update({
                'Verb 2level Accuracy': f"{sub_df.verb_2level_accuracy.mean():.02f}  {sub_df.verb_2level_accuracy.std():.02f}",
                'Object 2level Accuracy': f"{sub_df.object_2level_accuracy.mean():.02f}  {sub_df.object_2level_accuracy.std():.02f}",
            })
            map_cols = [c for c in sub_df.columns if "map_" in c]
            map_results.append({
                'Distance Function': dist_type,
                **{c:sub_df[c].mean() for c in map_cols}
            })
    results.sort(key=lambda x: SORT_KEYS.index(x["Distance Function"]))
    map_results.sort(key=lambda x: SORT_KEYS.index(x["Distance Function"]))
    results_df = pd.DataFrame(results)
    print(results_df)
    print(pd.DataFrame(map_results))
    print()
    # print(results_df.style.hide(axis="index").to_latex())


def rnn_dist(df, groups):
    results = []
    for dist_type, sub_df in df.groupby(groups):
        results.append({
            'Distance Function': dist_type,
            'Learnable': sub_df["rnn_distance_learnable"].mean()==1 if "rnn_distance_learnable" in list(sub_df.columns) else False,
            'Accuracy': f"{sub_df.accuracy.mean():.02f}   {sub_df.accuracy.std():.02f}",
            'Edit Distance': f"{sub_df.edit_distance.mean():.02f}  {sub_df.edit_distance.std():.02f}",
            'Verb Accuracy': f"{sub_df.verb_accuracy.mean():.02f}  {sub_df.verb_accuracy.std():.02f}",
            'Object Accuracy': f"{sub_df.object_accuracy.mean():.02f}  {sub_df.object_accuracy.std():.02f}",
        })
    results_df = pd.DataFrame(results)
    print(results_df)


def rnn_dist_improvement(df):
    """ Show the relative improvement over not using a distance function when training an rnn """
    results = []
    acc_none = df[df["rnn_distance_type"] == "none"]["accuracy"].mean()
    ed_none = df[df["rnn_distance_type"] == "none"]["edit_distance"].mean()
    for dist, sub_df in df.groupby("rnn_distance_type"):
        print(sub_df)
        if dist in ["temporal", "temporal_object", "temporal_verb"]:
            results.append({
                'RNN Distance Function': dist,
                'Accuracy': sub_df["accuracy"].mean(),
                'Accuracy Delta': (sub_df["accuracy"].mean()-acc_none),
                'Edit Distance': sub_df["edit_distance"].mean(),
                'Edit Distance Delta': (sub_df["edit_distance"].mean()-ed_none),
            })
    print(f"Acc None: {acc_none}, Edit Dist None: {ed_none}")
    print(pd.DataFrame(results))


def evaluate_differences_between_models():
    df, cc = load_experiment_results("temporal_dist_no_ignore_set_backbone_object", "ikea_asm")
    from scipy import stats
    from statsmodels.stats.contingency_tables import mcnemar
    experiments_no_temporal_loss = df[df["rnn_distance_type"] == "none"]
    experiments_temporal_loss = df[df["rnn_distance_type"] != "none"]
    chi_stats = {}
    for idx1, row1 in experiments_no_temporal_loss.iterrows():
        for idx2, row2 in experiments_temporal_loss.iterrows():
            cc1 = np.array([np.mean(x) >= .5 for x in cc[idx1]])
            cc2 = np.array([np.mean(x) >= .5 for x in cc[idx2]])
            table = np.bincount(2 * (cc1 == True) + (cc2 == True), minlength=2*2).reshape(2, 2)
            comp = f"none_{row2['rnn_distance_type']}"
            if comp not in chi_stats:
                chi_stats[comp] = []
            chi_stats[f"none_{row2['rnn_distance_type']}"].append(mcnemar(table).statistic)
    for k,v in chi_stats.items():
        print(k,np.mean(v),1 - stats.chi2.cdf(np.mean(v), 1))


def main():
    # Ikea rnn temporal distance experiment
    df,_ = load_experiment_results("temporal_dist", "ikea_asm")
    df2 = df[(df["rnn_distance_temporal"] == 0.001) | (df["rnn_distance_type"] == "none")]
    rnn_dist_improvement(df2)


if __name__ == "__main__":
    main()