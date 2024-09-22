# coding: utf-8
import os
import torch
from torch.utils.data import Dataset
from PIL import Image


# Pytorch Dataset for RoBERTa
class RoBERTaDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text'] # 자동생성자막과 기사 본문만
        label = self.data.iloc[idx]['label']

        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long) # 정수형

# Pytorch Dataset for Transformer
class TransformerDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']
        
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long) # 정수형
    
# Pytorch Dataset for Transformer+LeNet
class TransformerLeNetDataset(Dataset):
    def __init__(self, dataframe, text_tokenizer, max_length, image_path, transform=None):
        self.data = dataframe
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        content = self.data.iloc[idx]['content'] # 내용
        video_id = self.data.iloc[idx]['video_id'] # 영상 ID
        label = self.data.iloc[idx]['label']

        # text 토크나이저
        text_inputs = self.text_tokenizer(content, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = text_inputs['input_ids'].squeeze(0)
        attention_mask = text_inputs['attention_mask'].squeeze(0)

        # image 불러오기 및 전처리
        img_path = os.path.join(self.image_path, f"{video_id}.jpg")
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)            

        return input_ids, attention_mask, image, torch.tensor(label, dtype=torch.long)  # 정수형