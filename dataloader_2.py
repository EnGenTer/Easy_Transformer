# 2023_08_18_csw_NOTE 数据导入

import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class Translator:
    def __init__(self, dictionary):
        self.word_idx = dictionary['word_idx']
        self.idx_word = dictionary['idx_word']
        self.nVocabularyNum = len(self.idx_word)

    def TranslateWord2Idx(self, sentence):
        sentence = sentence.split()
        sentence = [self.word_idx[word] if word in self.word_idx else self.word_idx['<UNK>'] for word in sentence]
        return sentence

    def TranslateIdx2Word(self, sentence): 
        sentence = [self.idx_word[idx] for idx in sentence]
        return " ".join(sentence)