# 2023_08_18_csw_NOTE 模型训练

import tqdm
import json
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import os
from random import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy

from dataloader_2 import MyDataset, Translator
from attention_4 import Attention
from gan_5 import GAN
from eval_6 import Eval

# 2023_08_18_csw_NOTE 超参
REFINED_DATA_FILE_PATH = 'refineddata/'
REFINED_DATA_FILE_NAME_EN = 'news-commentary-v13.ru-en.en'
REFINED_DATA_FILE_NAME_ZH = 'news-commentary-v13.zh-en.zh'
DICTIONARY_FILE_PATH = 'dictionary/'
DICTIONARY_FILE_NAME_EN = 'dictionary_en.json'
DICTIONARY_FILE_NAME_ZH = 'dictionary_zh.json'
CHECKPOINT_PREFIX = "checkpoint/"
BEST_CHECKPOINT_PROFIX = "_best.pth"
LAST_CHECKPOINT_PROFIX = "_last.pth"
CHECKPOINT_FILE_NAME = str(int(time.time()))
CHECK_POINT_PATH = "" # 2023_08_22_csw_NOTE checkpoint读入。不填写则为从头训练

EPOCH_NUM = 10
LEARNING_RATE = 1e-4
LEARNING_RATE_DECAY = 0.5
BOS_IDX = 1
EOS_IDX = 2
BATCH_SIZE = 1
GENERATE_LOSS_WEIGHT = 1
GAN_LOSS_WEIGHT = 0.5
EVAL_EVERY_EPOCH = 50

DEVICE = 'cpu'
if torch.cuda.is_available(): 
    DEVICE = 'cuda'

# 2023_08_18_csw_NOTE 数据读取及翻译器初始化
strChDictDataFileName = DICTIONARY_FILE_PATH + DICTIONARY_FILE_NAME_ZH;
strChJsonDataFileName = REFINED_DATA_FILE_PATH + REFINED_DATA_FILE_NAME_ZH;
strEnDictDataFileName = DICTIONARY_FILE_PATH + DICTIONARY_FILE_NAME_EN;
strEnJsonDataFileName = REFINED_DATA_FILE_PATH + REFINED_DATA_FILE_NAME_EN;

# 2023_08_18_csw_NOTE 构建数据集
with open(strChJsonDataFileName, 'r', encoding='utf-8') as fileData: 
    ChData = json.loads(fileData.read())
    fileData.close() 

with open(strEnJsonDataFileName, 'r', encoding='utf-8') as fileData: 
    EnData = json.loads(fileData.read())
    fileData.close() 

Data = [data for data in zip(ChData, EnData)]
Dataset = DataLoader(MyDataset(Data), batch_size=BATCH_SIZE, shuffle=True)

# 2023_08_18_csw_NOTE 构建翻译器 
with open(strChDictDataFileName, 'r', encoding='utf-8') as fileData: 
    ChData = json.loads(fileData.read())
    TranslatorCh = Translator(ChData)
    fileData.close() 

with open(strEnDictDataFileName, 'r', encoding='utf-8') as fileData: 
    EnData = json.loads(fileData.read())
    TranslatorEn = Translator(EnData)
    fileData.close() 

# 2023_08_18_csw_NOTE 构建模型
modelGenerateModel = Attention(TranslatorCh.nVocabularyNum, TranslatorEn.nVocabularyNum)
modelGan = GAN()
modelGenerateModel.to(DEVICE)
modelGan.to(DEVICE)

# 2023_08_18_csw_NOTE 构建训练器
AdamOptimizer = Adam(modelGenerateModel.parameters(), lr=LEARNING_RATE)
nMinLoss = 2e9

lr_scheduler = torch.optim.lr_scheduler.StepLR(AdamOptimizer, step_size=1, gamma=LEARNING_RATE_DECAY)

if os.path.exists(CHECKPOINT_PREFIX[:-1]) == False:
    os.mkdir(CHECKPOINT_PREFIX[:-1]) 

if CHECK_POINT_PATH != "":
    saved_pth = torch.load(CHECKPOINT_PREFIX + CHECK_POINT_PATH)
    modelGenerateModel.load_state_dict(saved_pth['generate_model'])
    modelGan.load_state_dict(saved_pth['GAN_model'])
    AdamOptimizer.load_state_dict(saved_pth['optimizer'])
    nEpoch = saved_pth['epoch']
else: 
    nEpoch = 0

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
    tensorFake = tensor1dWordIdx
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

def ShowAttention(strEn, strCh, tensorAttention): 
    plt.rcParams['font.sans-serif']=['SimHei'] # 2023_08_22_csw_NOTE 显示中文标签
    plt.rcParams['axes.unicode_minus']=False
    arrstrEn = strEn.split()
    arrstrCh = strCh.split()

    plt.xticks(numpy.arange(len(arrstrCh)), arrstrCh)
    plt.yticks(numpy.arange(len(arrstrEn)), arrstrEn)
    numpy2dAttention = tensorAttention.transpose(-1, -2).cpu().detach().numpy()
    plt.imshow(numpy2dAttention, cmap=plt.cm.Reds, vmin=numpy2dAttention.min(), vmax=numpy2dAttention.max())
    plt.show()

# 2023_08_18_csw_NOTE 训练
while nEpoch < EPOCH_NUM: 
    for data in tqdm(Dataset): 
        modelGenerateModel.train()
        nBatch = len(data[0])
        AdamOptimizer.zero_grad()
        ChInputData = [TranslatorCh.TranslateWord2Idx(data[0][i]) + [EOS_IDX] for i in range(nBatch)]
        EnInputData = [TranslatorEn.TranslateWord2Idx(data[1][i]) for i in range(nBatch)]
        ChInputData, ChPaddingMask = Padding(ChInputData)
        EnInputData, EnPaddingMask = Padding(EnInputData)
        
        tensor1dWordIdx, tensor1dAttention, nGenerateLoss = modelGenerateModel(ChInputData, EnInputData, ChPaddingMask)
        # ShowAttention(data[1][0], TranslatorCh.TranslateIdx2Word(tensor1dWordIdx[0]), tensor1dAttention[0]) # 2023_09_01_csw_NOTE 展示
        # TranslatorCh.TranslateIdx2Word(tensor1dWordIdx[0]) # 2023_08_29_csw_NOTE 展示
        tensorRandRes, tensorRand = GenGANData(tensor1dWordIdx, ChInputData)
        tensorRandRes = tensorRandRes.type(torch.int32)
        tensorGANRes = modelGan(modelGenerateModel.modelChEmbedding(tensorRandRes))
        nGanLoss = torch.log(tensorGANRes * (1 - 1e-10) + 1e-10)
        nGanLoss = torch.where(torch.isinf(nGanLoss), torch.full_like(nGanLoss, -10.0), nGanLoss)
        nGanLoss.clamp_(min=-10.0) # 2023_08_29_csw_NOTE 防止出现意外
        nGanLoss = -torch.mean(nGanLoss * (tensorRand + 1e-10))

        # 2023_08_18_csw_NOTE 反向传播
        nFinalLoss = GENERATE_LOSS_WEIGHT * nGenerateLoss + GAN_LOSS_WEIGHT * nGanLoss
        nFinalLoss.backward()
        torch.nn.utils.clip_grad_norm_(modelGan.parameters(), 100)
        torch.nn.utils.clip_grad_norm_(modelGenerateModel.parameters(), 100)
        AdamOptimizer.step()

    saved_pth = {'epoch': nEpoch,
                'generate_model': modelGenerateModel.state_dict(),
                'GAN_model': modelGan.state_dict(), 
                'optimizer': AdamOptimizer.state_dict(), }
    torch.save(saved_pth, CHECKPOINT_PREFIX + CHECKPOINT_FILE_NAME + LAST_CHECKPOINT_PROFIX)
    lr_scheduler.step()
    if (nEpoch + 1) % EVAL_EVERY_EPOCH == 0: 
        # 2023_08_18_csw_NOTE 保存误差最低的参数
        nMeanGenerateLoss, nMeanGANLoss = Eval(Dataset, TranslatorCh, TranslatorEn, modelGenerateModel, modelGan)
        nMeanLoss = nMeanGANLoss + nMeanGenerateLoss
        if nMeanLoss < nMinLoss: 
            nMinLoss = nMeanLoss
            saved_pth = {'epoch': nEpoch,
                        'generate_model': modelGenerateModel.state_dict(),
                        'GAN_model': modelGan.state_dict(), 
                        'optimizer': AdamOptimizer.state_dict(), }
            torch.save(saved_pth, CHECKPOINT_PREFIX + CHECKPOINT_FILE_NAME + BEST_CHECKPOINT_PROFIX)
        print("{} / {}, generate_loss: {}, gan_loss: {}".format(nEpoch + 1, EPOCH_NUM, nMeanGenerateLoss, nMeanGANLoss))

    nEpoch += 1
    # print("learning rate: {}".format(AdamOptimizer.state_dict()['param_groups'][0]['lr']))
