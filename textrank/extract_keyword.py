# coding : utf-8
import json

from common.utils import YouTubeCaptionCrawler, baruen_tokenizer
from textrank.summarizer import KeywordSummarizer

def extract_keywords(url, topk=20, min_count=2, min_cooccurrence=1):
    '''
    TextRank로 유튜브 영상 자막 중에서 중요 키워드 추출
    
    Args:
        url (str): YouTube 영상 URL
        topk (int): 추출할 문장 수
        min_count (int) : 단어의 최소 등장 빈도수
        min_cooccurrence (int) : 단어의 최소 동시 등장 빈도수
    '''
    
    # Fetch captions by YouTubeCaptionCrawler
    crawler = YouTubeCaptionCrawler(url)
    caption = crawler.get_caption()
    sentences = crawler.split_sentences()
    
    # Extract keywords using KeywordSummarizer
    summarizer = KeywordSummarizer(tokenize=baruen_tokenizer, min_count=min_count, min_cooccurrence=min_cooccurrence)
    keywords = summarizer.summarize(sentences, topk=topk)
    
    # Keyword data
    keyword_data = []
    for keyword in keywords:
        summary_entry = {
            'keyword': keyword[0],
            'index': keyword[1]
        }
        keyword_data.append(summary_entry)
    
    # Output file path
    output_file = 'keyword.json'
    
    # Data to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(keyword_data, f, ensure_ascii=False, indent=4)
    
    # Return the path to the JSON file
    return output_file