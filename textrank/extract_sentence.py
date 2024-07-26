# coding : utf-8
import json

from common.utils import YouTubeCaptionCrawler, baruen_tokenizer
from textrank.summarizer import KeysentenceSummarizer

def extract_keysentences(url, topk=20, min_sim=0.3):
    '''
    TextRank로 유튜브 영상 자막 중에서 중요 문장 추출
    
    Args:
        url (str): YouTube 영상 URL
        topk (int): 추출할 문장 수
        min_sim (int) : 문장 유사도 임계값
    '''
    
    # Fetch captions by YouTubeCaptionCrawler
    crawler = YouTubeCaptionCrawler(url)
    caption = crawler.get_caption()
    sentences = crawler.split_sentences()
    
    # Summarize the sentences using KeysentenceSummarizer
    summarizer = KeysentenceSummarizer(tokenize=baruen_tokenizer, min_sim=min_sim, verbose=False)
    keysents = summarizer.summarize(sentences, topk=topk)
    
    # Summary data
    summary_data = []
    for sent_idx, rank, sent in keysents:
        summary_entry = {
            'importance': float(rank),
            'index': int(sent_idx),
            'sentence': sent
        }
        summary_data.append(summary_entry)
    
    # Output file path
    output_file = 'summary.json'
    
    # Data to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=4)
    
    # Return the path to the JSON file
    return output_file