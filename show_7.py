import json
import numpy
from matplotlib import pyplot as plt
import torch

from attention_4 import Attention
from dataloader_2 import Translator
import warnings
warnings.filterwarnings("ignore")

CHECK_POINT_PATH = "" # 2023_08_22_csw_NOTE 需要修改
CHECKPOINT_PREFIX = "checkpoint/"
DICTIONARY_FILE_PATH = 'dictionary/'
DICTIONARY_FILE_NAME_EN = 'dictionary_en.json'
DICTIONARY_FILE_NAME_ZH = 'dictionary_zh.json'
SHOW_IMAGE = True

DEVICE = 'cpu'
if torch.cuda.is_available(): 
    DEVICE = 'cuda'

strChDictDataFileName = DICTIONARY_FILE_PATH + DICTIONARY_FILE_NAME_ZH;
strEnDictDataFileName = DICTIONARY_FILE_PATH + DICTIONARY_FILE_NAME_EN;

def ShowAttention(strEn, strCh, tensorAttention): 
    plt.rcParams['font.sans-serif']=['SimHei'] # 2023_08_22_csw_NOTE 显示中文标签
    plt.rcParams['axes.unicode_minus']=False
    arrstrEn = strEn.split()
    arrstrCh = strCh.split()

    plt.xticks(numpy.arange(len(arrstrCh)), arrstrCh)
    plt.yticks(numpy.arange(len(arrstrEn)), arrstrEn)
    numpy2dAttention = tensorAttention.transpose(-1, -2).cpu().detach().numpy()
    plt.imshow(numpy2dAttention, cmap=plt.cm.Reds, vmin=0, vmax=1)
    plt.show()

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
modelGenerateModel.to(DEVICE)

modelpth = torch.load(CHECKPOINT_PREFIX + CHECK_POINT_PATH)
modelGenerateModel.load_state_dict(modelpth["generate_model"], strict = True)
# modelGenerateModel.eval()
modelGenerateModel.train(False)

nEpoch = 1
print("#{} intput: ".format(nEpoch), end="")
strInput = input()
while len(strInput) != 0: 
    strEnInputData = TranslatorEn.TranslateWord2Idx(strInput)
    _, tensor2dAttention, strChOutputData = modelGenerateModel.EvalForward(strEnInputData)
    strChOutputData = TranslatorCh.TranslateIdx2Word(strChOutputData)
    print("#{} output: ".format(nEpoch) + " ".join(strChOutputData.split()))
    if SHOW_IMAGE: 
        ShowAttention(strInput, strChOutputData, tensor2dAttention[0])

    nEpoch += 1
    print("#{} intput: ".format(nEpoch), end="")
    strInput = input()

# the tendency is either excessive restraint europe or a diffusion of the effort the united states
