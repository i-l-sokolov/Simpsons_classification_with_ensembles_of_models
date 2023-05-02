import torch.nn as nn
from models.models_init import get_model
import torch


class Ensemble(nn.Module):
    def __init__(self, names_nn, mask, device):
        super(Ensemble, self).__init__()

        self.names_NN = names_nn

        self.models = []
        for model_name in names_nn:
            model, _ = get_model(model_name,trained=True)
            self.models.append(model.to(device))
        self.mask = mask


    def forward(self, x):

        y_list = []

        for model in self.models:
            y_list.append(torch.softmax(model(x)[:,:42], dim=1))

        out = torch.stack(y_list).permute(1, 0, 2)

        out =  torch.mul(self.mask, out).sum(axis=1)

        return out
