# coding : utf-8

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, logging

from common.utils import fixSEED, YouTubeCaptionCrawler
from models.transformer import BD_Transformer
from models.roberta import BD_Roberta


def model_inference(url, transformer=True):
    '''
    모델을 불러와서 새로운 영상에 대해서 Inference를 수행하는 함수
    
    Args
        url : 영상 링크
        transformer : 사용할 모델이 transformer인지 여부
    '''
    
    fixSEED(42) # fix seed for reproducibility
    torch.cuda.empty_cache() # 메모리 캐시 삭제
    logging.set_verbosity_error() # 메시지 출력 off
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # GPU 사용여부 확인 후 지정
    
    caption = YouTubeCaptionCrawler(url).get_caption() # 영상의 자막을 크롤링
    
    if transformer:
        model_checkpoint = 'klue/roberta-small'
        model_path = 'transformer_ACC_0.9231.pth'
        text_config = AutoConfig.from_pretrained(model_checkpoint) # 모델 설정
        model = BD_Transformer(text_config) # 모델
        model.load_state_dict(torch.load(model_path)) # 파라미터 덮어쓰기
    else:        
        model_checkpoint = 'klue/roberta-large'
        model_path = 'roberta_ACC_0.9265.pth'
        model = BD_Roberta(model_checkpoint) # 모델
        model.load_state_dict(torch.load(model_path)) # 파라미터 덮어쓰기
        
    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint) # 토크나이저
    inputs = tokenizer(caption, padding=True, truncation=True, return_tensors="pt") # 토큰화, 정수 매핑, 패딩
    input_ids, attention_mask = inputs["input_ids"].to(device), inputs["attention_mask"].to(device)
    
    # 모델 추론
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = F.softmax(logits, dim=-1)[:, 1].item() # 낚시성일 확률
    
    return round(probabilities, 3) # 소수점 3자리까지 반올림하여 반환