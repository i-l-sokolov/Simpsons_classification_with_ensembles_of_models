import seaborn as sns
from greedy_search import min_columns_cover
import copy
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_clustermap(df_res, df_res_min):
    g = sns.clustermap(df_res, figsize=(20, 20), dendrogram_ratio=0.1)

    reordered_columns = g.dendrogram_col.reordered_ind

    for idx, i in enumerate(reordered_columns):
        if df_res.columns[i] in df_res_min.columns:
            g.ax_heatmap.hlines([0, len(df_res)], idx, idx + 1, colors='red', lw=4)
            g.ax_heatmap.vlines([idx, idx + 1], 0, len(df_res), colors='red', lw=4)
        elif df_res.columns[i] == 'ensemble':
            g.ax_heatmap.hlines([0, len(df_res)], idx, idx + 1, colors='blue', lw=5)
            g.ax_heatmap.vlines([idx, idx + 1], 0, len(df_res), colors='blue', lw=5)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=20)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=15)
    g.savefig('../report/clustermap_with_ensemble.png')

def get_colors(names, selected_list):
    cols = []
    for x in names:
        if x in selected_list:
            col = 'blue'
        elif x == 'ensemble':
            col = 'red'
        else:
            col = 'green'
        cols.append(col)
    return cols


def get_plot(heights, names, selected_list):
    res = pd.DataFrame({'name' : names, 'score' : heights})
    res = res.sort_values(by='score')
    bar_colors = get_colors(res['name'],selected_list)
    plt.figure(figsize=(30, 10))
    plt.bar(x=res['name'], height=res['score'], color=bar_colors)
    plt.ylim((res['score'].min() - 0.05, 1.05))
    plt.xticks(rotation=90, fontsize=25)
    plt.yticks(fontsize=20)
    plt.ylabel('Best validation accuracy on 20 epochs', fontsize=25)
    plt.title('Comparison of torchvision models performance on classification of Simpsons dataset', fontsize=30)
    plt.savefig('../report/comparison_accuracies.png', bbox_inches='tight', pad_inches=2)
    plt.show()



def get_dfres(wrongs, names):
    df_res = pd.DataFrame(wrongs).T
    df_res.columns = names
    return df_res


def get_df_min(df_res):
    col = min_columns_cover(df_res.apply(lambda x : x == df_res.min(axis=1), axis=0).values)
    return copy.copy(df_res.iloc[:,col])


def get_mask(df_res_min, device):
    mask = ((1 - df_res_min).T / (1 - df_res_min).T.sum(axis=0)).values
    return torch.tensor(mask).to(device)


def get_wrong(wrong, val_dataset):
    wrong = [x.cpu().numpy() for x in wrong]
    wrong = np.bincount(wrong)
    add = np.array([0]*(42 - wrong.shape[0]))
    wrong = np.concatenate((wrong, add))
    val_counts = np.bincount([val_dataset.label_codes[x] for x in val_dataset.labels])
    if 0 in val_counts:
        val_counts += 1
    wrong_f = wrong.astype(np.float64)
    wrong_f /= val_counts
    return wrong_f


def get_submit(df_sub, ensemble, test_dataloader, label_decodes, device):
    all_preds = []
    for X in tqdm(test_dataloader):
        X = X.to(device)
        pred = ensemble(X)
        all_preds += list(pred.cpu().argmax(axis=1).numpy())
    classes = [label_decodes[i] for i in all_preds]
    df_sub['Expected'] = classes
    df_sub.to_csv('../report/submission.csv', index=False)