# 2023_08_18_csw_NOTE 注意力机制

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

MAX_SENTENCE_LEN = 16
END_SYMBOLS = ["<BOS>", "<EOS>", "<PAD>"]

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        #d_model是每个词embedding后的维度
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term2 = torch.pow(torch.tensor(1000.0),torch.arange(0, d_model, 2).float()/d_model)
        div_term1 = torch.pow(torch.tensor(1000.0),torch.arange(1, d_model, 2).float()/d_model)
        #高级切片方式，即从0开始，两个步长取一个。即奇数和偶数位置赋值不一样。直观来看就是每一句话的
        pe[:, 0::2] = torch.sin(position * div_term2)
        pe[:, 1::2] = torch.cos(position * div_term1)
        #这里是为了与x的维度保持一致，释放了一个维度
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.pe = pe

    def forward(self, x):
        x = x.transpose(0, 1) + self.pe[:x.shape[1]]
        return x.transpose(0, 1)

class EncoderBlock(nn.Module):
    def __init__(self, nFeatureDim = 256, nEmbeddingDim = 256, nFeedFowardNetDim = 1024, nDropout = 0.5):
        super(EncoderBlock, self).__init__()
        self.nFeatureDim = nFeatureDim
        self.modelWeightQuery = nn.Linear(nFeatureDim, nFeatureDim, bias=False)
        self.modelWeightKey = nn.Linear(nFeatureDim, nFeatureDim, bias=False)
        self.modelWeightValue = nn.Linear(nFeatureDim, nFeatureDim, bias=False)
        self.modelFeedFowardNetInput = nn.Linear(nFeatureDim, nFeedFowardNetDim)
        self.modelFeedFowardNetOutput = nn.Linear(nFeedFowardNetDim, nEmbeddingDim)
        self.modelLayerNorm = nn.LayerNorm(nFeatureDim)
        self.modelDropout = nn.Dropout(nDropout)

    def forward(self, tensorInputKey, tensorInputValue, tensorInputQuery):
        tensorQuery = self.modelWeightQuery(tensorInputQuery)
        tensorKey = self.modelWeightKey(tensorInputKey)
        tensorValue = self.modelWeightValue(tensorInputValue)

        tensorAttention = torch.softmax(torch.matmul(tensorQuery, tensorKey.transpose(-1, -2)) / (self.nFeatureDim ** 0.5), dim=-1) 
        tensorAttention = self.modelDropout(tensorAttention)
        tensorAttentionRes = torch.matmul(tensorAttention, tensorValue)
        tensorAttentionRes = self.modelDropout(tensorAttentionRes)
        tensorAttentionRes = self.modelLayerNorm(tensorAttentionRes + tensorInputQuery)
        tensorFFNRes = self.modelFeedFowardNetOutput(self.modelFeedFowardNetInput(tensorAttentionRes))
        tensorFFNRes = self.modelDropout(tensorFFNRes)
        tensorRes = self.modelLayerNorm(tensorFFNRes + tensorAttentionRes)
        return tensorRes, tensorAttention
 
class DecoderBlock(nn.Module):
    def __init__(self, nFeatureDim = 256, nEmbeddingDim = 256, nFeedFowardNetDim = 1024, nDropout = 0.5):
        super(DecoderBlock, self).__init__()
        self.nFeatureDim = nFeatureDim
        self.modelWeightQuery = nn.Linear(nFeatureDim, nFeatureDim, bias=False)
        self.modelWeightKey = nn.Linear(nFeatureDim, nFeatureDim, bias=False)
        self.modelWeightValue = nn.Linear(nFeatureDim, nFeatureDim, bias=False)
        self.modelWeightSelfAttentionQuery = nn.Linear(nFeatureDim, nFeatureDim, bias=False)
        self.modelWeightSelfAttentionKey = nn.Linear(nFeatureDim, nFeatureDim, bias=False)
        self.modelWeightSelfAttentionValue = nn.Linear(nFeatureDim, nFeatureDim, bias=False)
        self.modelFeedFowardNetInput = nn.Linear(nFeatureDim, nFeedFowardNetDim)
        self.modelFeedFowardNetOutput = nn.Linear(nFeedFowardNetDim, nEmbeddingDim)
        self.modelSelfAttentionFeedFowardNetInput = nn.Linear(nFeatureDim, nFeedFowardNetDim)
        self.modelSelfAttentionFeedFowardNetOutput = nn.Linear(nFeedFowardNetDim, nEmbeddingDim)
        self.modelLayerNorm = nn.LayerNorm(nFeatureDim)
        self.modelDropout = nn.Dropout(nDropout)

    def forward(self, tensorInputKey, tensorInputValue, tensorInputQuery, tensor2dMask):
        tensorKey = self.modelWeightSelfAttentionKey(tensorInputQuery)
        tensorQuery = self.modelWeightSelfAttentionQuery(tensorInputQuery)
        tensorValue = self.modelWeightSelfAttentionValue(tensorInputQuery)
        
        tensorAttention = torch.softmax(torch.matmul(tensorQuery, tensorKey.transpose(-1, -2)) / (self.nFeatureDim ** 0.5), dim=-1) 
        tensorAttention = tensorAttention * tensor2dMask[None, :, :]
        tensorAttentionRes = torch.matmul(tensorAttention, tensorValue)
        tensorAttentionRes = self.modelDropout(tensorAttentionRes)
        tensorAttentionRes = self.modelLayerNorm(tensorAttentionRes + tensorInputQuery)
        tensorFFNRes = self.modelSelfAttentionFeedFowardNetOutput(self.modelSelfAttentionFeedFowardNetInput(tensorAttentionRes))
        tensorFFNRes = self.modelDropout(tensorFFNRes)
        tensorQuery = self.modelLayerNorm(tensorFFNRes + tensorAttentionRes)

        # tensorQuery = self.modelWeightQuery(tensorInputQuery)
        tensorKey = self.modelWeightKey(tensorInputKey)
        tensorValue = self.modelWeightValue(tensorInputValue)

        tensorAttention = torch.softmax(torch.matmul(tensorQuery, tensorKey.transpose(-1, -2)) / (self.nFeatureDim ** 0.5), dim=-1) 
        tensorAttentionRes = torch.matmul(tensorAttention, tensorValue)
        tensorAttentionRes = self.modelDropout(tensorAttentionRes)
        tensorAttentionRes = self.modelLayerNorm(tensorAttentionRes + tensorInputQuery)
        tensorFFNRes = self.modelFeedFowardNetOutput(self.modelFeedFowardNetInput(tensorAttentionRes))
        tensorFFNRes = self.modelDropout(tensorFFNRes)
        tensorRes = self.modelLayerNorm(tensorFFNRes + tensorAttentionRes)
        return tensorRes, tensorAttention

class Attention(nn.Module):
    def __init__(self, nChVocabulary, nEnVocabulary, nEmbeddingDim = 512, nEncoderBlockNum = 6, nDecoderBlockNum = 8, nDropout = 0.5):
        super(Attention, self).__init__()
        self.modelChEmbedding = nn.Embedding(nChVocabulary, nEmbeddingDim)
        self.modelEnEmbedding = nn.Embedding(nEnVocabulary, nEmbeddingDim)
        self.modelPositionalEmbedding = PositionalEmbedding(nEmbeddingDim)
        self.modelEncoderBlock = EncoderBlock(nEmbeddingDim, nEmbeddingDim, nEmbeddingDim * 2, nDropout)
        self.modelDecoderBlock = DecoderBlock(nEmbeddingDim, nEmbeddingDim, nEmbeddingDim * 2, nDropout)
        self.modelEncoderLayers = nn.ModuleList([copy.deepcopy(self.modelEncoderBlock) for i in range(nEncoderBlockNum)])
        self.modelDecoderLayers = nn.ModuleList([copy.deepcopy(self.modelDecoderBlock) for i in range(nDecoderBlockNum)])
        self.modelChWordLinear = nn.Linear(nEmbeddingDim, nChVocabulary)
        self.setstrEndSymbol = ('<PAD>', '<BOS>', '<EOS>')
        self.modelLoss = nn.CrossEntropyLoss()
        self.nHiddenDim = nEmbeddingDim
        self.nVocabulary = nChVocabulary

    def forward(self, ChInputData, EnInputData, ChPaddingMask): 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ChInputData = torch.tensor(ChInputData, device=device)
        EnInputData = torch.tensor(EnInputData, device=device)

        tensor2dChFeatures = self.modelChEmbedding(ChInputData)
        tensor2dEnFeatures = self.modelEnEmbedding(EnInputData)
        tensor2dChFeatures = self.modelPositionalEmbedding(tensor2dChFeatures)
        tensor2dEnFeatures = self.modelPositionalEmbedding(tensor2dEnFeatures)

        for modelEncoderLayer in self.modelEncoderLayers:
            tensor2dEnFeatures, tensor2dAttention = modelEncoderLayer(tensor2dEnFeatures, tensor2dEnFeatures, tensor2dEnFeatures)
        
        # tensor1dHiddenFeature = torch.mean(tensor2dEnFeatures, dim = -2).unsqueeze(1)        
        tensor2dChMask = (1 - torch.triu(torch.ones((tensor2dChFeatures.shape[-2], tensor2dChFeatures.shape[-2]), device=device)))

        _, tensor2dAttention = self.modelDecoderLayers[0](tensor2dEnFeatures, tensor2dEnFeatures, tensor2dChFeatures, tensor2dChMask)
        for i in range(1, len(self.modelDecoderLayers)): 
            modelDecoderLayer = self.modelDecoderLayers[i]
            tensor2dChFeatures, _ = modelDecoderLayer(tensor2dEnFeatures, tensor2dEnFeatures, tensor2dChFeatures, tensor2dChMask)

        # tensor2dOutput = torch.concat(tensor2dOutput, dim = 1)
        # tensor2dAttention = torch.concat(tensor2dAttention, dim = 1)
        tensor2dOutput = torch.log(torch.softmax(self.modelChWordLinear(tensor2dChFeatures), dim=-1))
        # tensor2dOutput = tensor2dOutput.squeeze(0) # 2023_08_27_csw_NOTE 忽略batch
        # tensor2dAttention = tensor2dAttention.squeeze(0) # 2023_08_27_csw_NOTE 忽略batch

        tensor2dOntHot = F.one_hot(ChInputData, self.nVocabulary)
        tensor1dLoss = -(tensor2dOntHot * tensor2dOutput * ChPaddingMask[:, :, None]).sum() / (ChPaddingMask[:, :].sum() + 1e-6)
        # tensor1dLoss = -torch.mean(tensor2dOntHot[:, :] * tensor2dOutput[:, -1:])
        return torch.max(tensor2dOutput, dim = -1)[1], tensor2dAttention, tensor1dLoss
        
    def EvalForward(self, EnInputData): 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        EnInputData = torch.tensor(EnInputData, device=device)
        tensor2dEnFeatures = self.modelEnEmbedding(EnInputData.unsqueeze(0))
        tensor2dEnFeatures = self.modelPositionalEmbedding(tensor2dEnFeatures)

        for modelEncoderLayer in self.modelEncoderLayers:
            tensor2dEnFeatures, tensor2dAttention = modelEncoderLayer(tensor2dEnFeatures, tensor2dEnFeatures, tensor2dEnFeatures)
        
        nMaxLen = tensor2dEnFeatures.shape[1] + 3
        ChOutput = []
        for nWordIdx in range(nMaxLen):
            tensor2dChFeatures = torch.tensor(ChOutput + [1], device=device).unsqueeze(0)
            tensor2dChFeatures = self.modelChEmbedding(tensor2dChFeatures)
            tensor2dChFeatures = self.modelPositionalEmbedding(tensor2dChFeatures)
            tensor2dChMask = (1 - torch.triu(torch.ones((tensor2dChFeatures.shape[-2], tensor2dChFeatures.shape[-2]), device=device)))
            for modelDecoderLayer in self.modelDecoderLayers:
                tensor2dChFeatures, tensor2dAttention = modelDecoderLayer(tensor2dEnFeatures, tensor2dEnFeatures, tensor2dChFeatures, tensor2dChMask)
            tensor2dOutput = torch.max(self.modelChWordLinear(tensor2dChFeatures).squeeze(0), dim=-1)[1][nWordIdx]
            if int(tensor2dOutput) < 3: 
                break
            ChOutput.append(int(tensor2dOutput))
        return torch.log(torch.softmax(self.modelChWordLinear(tensor2dChFeatures), dim=-1)), tensor2dAttention, ChOutput