import pandas as pd
import torchvision
import torch

def get_model(i, trained=False):
    if isinstance(i,int):
        name = pd.read_csv('models/models_names.csv').query('model_number == @i')['model_name'].values[0]
    if isinstance(i,str):
        name = i
    model = torchvision.models.get_model(name,weights='IMAGENET1K_V1')
    if trained:
        model.load_state_dict(torch.load(f'../trained_models/{name}.pth'))
    return model, name