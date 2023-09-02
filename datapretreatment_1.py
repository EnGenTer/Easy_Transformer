# 2023_08_18_csw_NOTE 创建数据字典和数据处理
import json

# 2023_08_18_csw_NOTE 超参区域
RAW_DATA_FILE_PATH = 'rawdata/'
RAW_DATA_FILE_NAME_EN = 'news-commentary-v13.zh-en.en'
RAW_DATA_FILE_NAME_ZH = 'news-commentary-v13.zh-en.zh'
REFINED_DATA_FILE_PATH = 'refineddata/'
REFINED_DATA_FILE_NAME_EN = 'news-commentary-v13.ru-en.en'
REFINED_DATA_FILE_NAME_ZH = 'news-commentary-v13.zh-en.zh'
DICTIONARY_FILE_PATH = 'dictionary/'
DICTIONARY_FILE_NAME_EN = 'dictionary_en.json'
DICTIONARY_FILE_NAME_ZH = 'dictionary_zh.json'
FILTER_CHARACTERS = ['\n', '\t', '\r', ' ', '　', "’", "-", '!', "'", ".", "?", ",", ";", ":", "，", "。", "：", "（", "）", "“", "…", "•", "—", "《", "》", "『", "』", "(", ")"]
SIMBOL = ['<PAD>','<BOS>', '<EOS>', '<UNK>']
CUTOFF_NUM = 100 # 2023_08_19_csw_NOTE 用于切分得到部分数据，方便debug

# 2023_08_18_csw_NOTE 中分分字
def SplitChStr(strInput):
    ENGLISH_CHARACTERS = 'abcdefghijklmnopqrstuvwxyz0123456789'
    strOutput = []
    strBuffer = ''
    for cChar in strInput:
        if cChar in ENGLISH_CHARACTERS or cChar in ENGLISH_CHARACTERS.upper(): #英文或数字
            strBuffer += cChar
        else: #中文
            if strBuffer:
                strOutput.append(strBuffer)
            strBuffer = ''
            strOutput.append(cChar)
    if strBuffer:
        strOutput.append(strBuffer)
    return strOutput

# 2023_08_18_csw_NOTE 中文数据预处理
# 2023_08_18_csw_NOTE 读取文件
strRawDataFileName = RAW_DATA_FILE_PATH + RAW_DATA_FILE_NAME_ZH; 
strDictDataFileName = DICTIONARY_FILE_PATH + DICTIONARY_FILE_NAME_ZH;
strJsonDataFileName = REFINED_DATA_FILE_PATH + REFINED_DATA_FILE_NAME_ZH;
strJsonData = []
dictData = {'word_idx': {}, 'idx_word': []}
for strSymbol in SIMBOL:
    dictData['word_idx'][strSymbol] = len(dictData['idx_word'])
    dictData['idx_word'].append(strSymbol)

with open(strRawDataFileName, 'r', encoding='utf-8') as fileRawDataFile:
    strRawData = fileRawDataFile.readlines()[:CUTOFF_NUM]
    for strLineData in strRawData:
        strLineData = strLineData.lower()
        for strFilterCharacter in FILTER_CHARACTERS:
            strLineData = strLineData.replace(strFilterCharacter, ' ')
        # arrstrLineData = strLineData.split(); 
        arrstrLineData = SplitChStr(strLineData.replace(' ', ''))
        for nWordIdx in range(len(arrstrLineData)):
            strWord = arrstrLineData[nWordIdx]
            try: 
                if strWord not in dictData['word_idx']:
                    dictData['word_idx'][strWord] = len(dictData['idx_word'])
                    dictData['idx_word'].append(strWord)
            except: 
                arrstrLineData[nWordIdx] = '<UNK>'
        strJsonData.append(" ".join(arrstrLineData))
    fileRawDataFile.close()

with open(strDictDataFileName, 'w+', encoding='utf') as fileDictDataFile:
    fileDictDataFile.write(json.dumps(dictData, ensure_ascii=False))
    fileDictDataFile.close()

with open(strJsonDataFileName, 'w+', encoding='utf') as fileJsonDataFile:
    fileJsonDataFile.write(json.dumps(strJsonData, ensure_ascii=False))
    fileJsonDataFile.close()

# 2023_08_18_csw_NOTE 英文数据预处理
# 2023_08_18_csw_NOTE 读取文件
strRawDataFileName = RAW_DATA_FILE_PATH + RAW_DATA_FILE_NAME_EN; 
strDictDataFileName = DICTIONARY_FILE_PATH + DICTIONARY_FILE_NAME_EN;
strJsonDataFileName = REFINED_DATA_FILE_PATH + REFINED_DATA_FILE_NAME_EN;
strJsonData = []
dictData = {'word_idx': {}, 'idx_word': []}
for strSymbol in SIMBOL:
    dictData['word_idx'][strSymbol] = len(dictData['idx_word'])
    dictData['idx_word'].append(strSymbol)

with open(strRawDataFileName, 'r', encoding='utf-8') as fileRawDataFile:
    strRawData = fileRawDataFile.readlines()[:CUTOFF_NUM]
    for strLineData in strRawData:
        strLineData = strLineData.lower()
        for strFilterCharacter in FILTER_CHARACTERS:
            strLineData = strLineData.replace(strFilterCharacter, ' ')
        arrstrLineData = strLineData.split(); 
        # arrstrLineData = SplitChStr(strLineData.replace(' ', ''))
        for nWordIdx in range(len(arrstrLineData)):
            strWord = arrstrLineData[nWordIdx]
            try: 
                if strWord not in dictData['word_idx']:
                    dictData['word_idx'][strWord] = len(dictData['idx_word'])
                    dictData['idx_word'].append(strWord)
            except: 
                arrstrLineData[nWordIdx] = '<UNK>'
        strJsonData.append(" ".join(arrstrLineData))
    fileRawDataFile.close()

with open(strDictDataFileName, 'w+', encoding='utf') as fileDictDataFile:
    fileDictDataFile.write(json.dumps(dictData, ensure_ascii=False))
    fileDictDataFile.close()

with open(strJsonDataFileName, 'w+', encoding='utf') as fileJsonDataFile:
    fileJsonDataFile.write(json.dumps(strJsonData, ensure_ascii=False))
    fileJsonDataFile.close()
