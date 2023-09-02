import torch
import torch.nn as nn
import torch.nn.functional as F

class GAN(nn.Module):
    def __init__(self, nInputFeatureDim = 512, nOutputFeatureDim = 512, nLayerNum = 2):
        super(GAN, self).__init__()
        if nInputFeatureDim != nOutputFeatureDim: 
            raise Exception("nInputFeatureDim != nOutputFeatureDim")
        self.arrmodelLayers = nn.ModuleList([nn.TransformerEncoderLayer(nInputFeatureDim, 4, batch_first=True) for _ in range(nLayerNum)])
        self.modelLinear = nn.Linear(nInputFeatureDim, 2)

    def forward(self, tensorInput): 
        for modelLinearLayer in self.arrmodelLayers: 
            tensorInput = modelLinearLayer(tensorInput)
        tensor1dRes = torch.sum(tensorInput, dim=1)
        tensor1dRes = self.modelLinear(tensor1dRes)
        return torch.softmax(tensor1dRes, dim=-1)
