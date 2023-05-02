import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from glob import glob
import os
import pandas as pd
import numpy as np
import random
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.utils import class_weight
import json
import pickle
from dataset import SimpsonsDataset2
from train_eval import model_choosing, eval_model
from functions import get_clustermap, get_df_min, get_wrong, get_mask, get_dfres, get_plot, get_submit
import argparse
from models.models_init import get_model
from ensemble import Ensemble


parser = argparse.ArgumentParser(description='Trained parameters and models selection')
parser.add_argument('--models', default=1, nargs='+', type=int, help='models from the list of models')
parser.add_argument('--batch_size', default=200, type=int, help='the size of batch during training and validation')
parser.add_argument('--epochs', default=20, type=int, help='the number of epochs for train every NN')
parser.add_argument('--submit', default=0, type=int, help='if non zero then will be created file for submission')

args = parser.parse_args()

model_list = args.models
batch_size = args.batch_size
epochs = args.epochs
submit = args.submit

all_pictures = sorted(glob('../data/train/simpsons_dataset/*/*.jpg'))
all_labels = [x.split('/')[-2] for x in all_pictures]

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

with open('../pickles/split','r') as f:
    data =json.load(f)
train_imgs = [os.path.join('../',x) for x in data['train']][:1000]
val_imgs = [os.path.join('../',x) for x in data['valid']][:1000] #!!!!!!!!! ELIMINATE THEN
train_labels = [os.path.basename(os.path.dirname(x)) for x in train_imgs]
val_labels = [os.path.basename(os.path.dirname(x)) for x in val_imgs]


df_sub = pd.read_csv('../pics/sample_submission.csv')
test_imgs = [os.path.join('../data/testset/testset/', x) for x in df_sub['Id'].values]


with open('../pickles/label_codes.pkl','rb') as f:
    labels_codes = pickle.load(f)

class_counts = np.bincount([labels_codes[label] for label in all_labels])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bal_weights = class_weight.compute_class_weight("balanced", classes=list(labels_codes.keys()), y=all_labels)
bal_weights = torch.FloatTensor(bal_weights).to(device)

probs = (bal_weights / bal_weights.max()).cpu().numpy()

label_decodes = dict(zip(labels_codes.values(),labels_codes.keys()))

train_dataset = SimpsonsDataset2(train_imgs, train_labels, labels_codes, probs, 'train')
val_dataset = SimpsonsDataset2(val_imgs, val_labels, labels_codes, probs, 'valid')
test_dataset = SimpsonsDataset2(test_imgs, train_labels, labels_codes, probs, 'test')

sampler = WeightedRandomSampler(
    weights= [probs[labels_codes[i]] for i in train_dataset.labels],
    num_samples=2*len(train_dataset),
    replacement=True
)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=75, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


def main(device, val_dataloader):
    best_acc_list, model_names, best_wrong_list = [], [], []
    criterion = nn.CrossEntropyLoss()
    for i in model_list:
        model, name = get_model(i)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
        model_names.append(name)
        best_acc, best_wrong = model_choosing(model, epochs, name, optimizer, criterion, scheduler, device,
                                              val_dataloader, train_dataloader)
        model.to('cpu')
        torch.cuda.empty_cache()
        best_acc_list.append(best_acc)
        best_wrong_list.append(get_wrong(best_wrong, val_dataset))

    df_res = get_dfres(best_wrong_list, model_names)
    df_res_min = get_df_min(df_res)
    mask = get_mask(df_res_min, device)
    ensemble = Ensemble(df_res_min.columns, mask, device)
    ensemble.to(device)
    torch.cuda.empty_cache()
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    val_loss, acc, wrong = eval_model(ensemble, val_dataloader, criterion, device, desc='valid assemble')
    best_wrong_list.append(get_wrong(wrong, val_dataset))
    best_acc_list.append(acc)
    model_names.append('ensemble')
    df_res = get_dfres(best_wrong_list, model_names)
    get_clustermap(df_res, df_res_min)
    get_plot(best_acc_list, model_names, df_res_min.columns)
    if submit:
        torch.cuda.empty_cache()
        get_submit(df_sub, ensemble, test_dataloader, label_decodes, device)


if __name__ == '__main__':
    try:
        main(device, val_dataloader)
    except:
        main(torch.device('cpu'), val_dataloader)