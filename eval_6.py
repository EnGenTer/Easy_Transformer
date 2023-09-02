### 2023_08_31_csw_NOTE 注意在实际应用中这么写是错的！
### 2023_08_31_csw_NOTE 注意在实际应用中这么写是错的！
### 2023_08_31_csw_NOTE 注意在实际应用中这么写是错的！
### 2023_08_31_csw_NOTE 注意在实际应用中这么写是错的！
### 2023_08_31_csw_NOTE 注意在实际应用中这么写是错的！
### 需要用bleu等指标计算

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import numpy
from matplotlib import pyplot as plt
from tqdm import tqdm

from attention_4 import Attention
from dataloader_2 import MyDataset, Translator

BOS_IDX = 1
EOS_IDX = 2
GENERATE_LOSS_WEIGHT = 1
GAN_LOSS_WEIGHT = 0.5
DEVICE = 'cpu'
if torch.cuda.is_available(): 
    DEVICE = 'cuda'

def Padding(arrarrnInput): 
    nBatch = len(arrarrnInput)
    nMaxLen = max([len(arrInput) for arrInput in arrarrnInput])
    arrnPadLen = [nMaxLen - len(arrInput) for arrInput in arrarrnInput]
    arrarrnInput = [arrInput + [0] * nPadLen for arrInput, nPadLen in zip(arrarrnInput, arrnPadLen)]
    tensorMask = torch.ones(nBatch, nMaxLen, device=DEVICE)
    for i in range(nBatch):
        tensorMask[i, nMaxLen - arrnPadLen[i] - 1 :nMaxLen] = 0
    return arrarrnInput, tensorMask

def GenGANData(tensor1dWordIdx, ChInputData): 
    tensorReal = torch.tensor(ChInputData, device=DEVICE)[:, 1:]
    # tensorFake = tensor1dWordIdx
    tensorFake = torch.tensor(tensor1dWordIdx, device=DEVICE)[:, :]
    tensorRandRes = torch.ones((tensorReal.shape[0], tensorReal.shape[1] + tensorFake.shape[1]), device=DEVICE)
    tensorRand = torch.rand(tensorReal.shape[0], device=DEVICE) > 0.5
    tensorRand = tensorRand * 1
    for nBidx in range(tensorReal.shape[0]): 
        if (tensorRand[nBidx] == 1):
            tensorRandRes[nBidx, :tensorReal.shape[1]] = tensorReal[nBidx]
            tensorRandRes[nBidx, tensorReal.shape[1]:] = tensorFake[nBidx]
        else: 
            tensorRandRes[nBidx, tensorFake.shape[1]:] = tensorReal[nBidx]
            tensorRandRes[nBidx, :tensorFake.shape[1]] = tensorFake[nBidx]
    return tensorRandRes, F.one_hot(tensorRand)


def Eval(Dataset, TranslatorCh, TranslatorEn, modelGenerateModel, modelGan): 
    modelGenerateModel.eval()
    modelGan.eval()
    arrnGenerateLoss = []
    arrnGANLoss = []
    for data in tqdm(Dataset): 
        nBatch = len(data[0])
        for idx in range(nBatch): 
            ChInputData = TranslatorCh.TranslateWord2Idx(data[0][idx]) + [EOS_IDX] 
            EnInputData = TranslatorEn.TranslateWord2Idx(data[1][idx])
            ChInputData, ChPaddingMask = Padding([ChInputData])
            EnInputData, EnPaddingMask = Padding([EnInputData])
            ChInputData = ChInputData[0]
            EnInputData = EnInputData[0]

            tensor2dOutput, tensor2dAttention, tensorChOutput = modelGenerateModel.EvalForward(EnInputData)
            tensor2dOntHot = F.one_hot(torch.tensor(ChInputData), modelGenerateModel.nVocabulary).unsqueeze(0).to(DEVICE)
            nMinLen = min(tensor2dOutput.shape[1], tensor2dOntHot.shape[1])
            ### 2023_08_31_csw_NOTE 注意在实际应用中这么写是错的！
            nGenerateLoss = -(tensor2dOntHot[:, :nMinLen] * tensor2dOutput[:, :nMinLen] * ChPaddingMask[:, :nMinLen, None]).sum() / (ChPaddingMask[:, :nMinLen].sum() + 1e-6)

            tensorRandRes, tensorRand = GenGANData([tensorChOutput], [ChInputData])
            tensorRandRes = tensorRandRes.type(torch.int32)
            tensorGANRes = modelGan(modelGenerateModel.modelChEmbedding(tensorRandRes))

            nGanLoss = torch.log(tensorGANRes * (1 - 1e-10) + 1e-10)
            nGanLoss = torch.where(torch.isinf(nGanLoss), torch.full_like(nGanLoss, -10.0), nGanLoss)
            nGanLoss.clamp_(min=-10.0) # 2023_08_29_csw_NOTE 防止出现意外
            nGanLoss = -torch.mean(nGanLoss * (tensorRand + 1e-10))

            arrnGenerateLoss.append(nGenerateLoss)
            arrnGANLoss.append(nGanLoss)

    return sum(arrnGenerateLoss) / len(arrnGenerateLoss), sum(arrnGANLoss) / len(arrnGANLoss)
