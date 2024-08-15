# coding : utf-8
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer

from common.utils import YouTubeCaptionCrawler, baruen_noun_tokenizer
from textrank.summarizer import KeywordSummarizer

def textrank_keywords(url, 
                      topk=20, 
                      min_count=3, 
                      min_cooccurrence=5, 
                      file_path=None):
    '''
    TextRank로 유튜브 영상 자막 중에서 중요 키워드 추출
    
    Args:
        url (str): YouTube 영상 URL
        topk (int): 추출할 문장 수
        min_count (int) : 단어의 최소 등장 빈도수
        min_cooccurrence (int) : 단어의 최소 동시 등장 빈도수
        file_path(str) : 불용어 파일이 저장된 경로
    '''
    
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), 'korean_stopwords.json')
    
    # Korean stopwords
    with open(file_path, 'r', encoding='utf-8') as file:
        korean_stopwords = json.load(file)
    
    # Fetch captions by YouTubeCaptionCrawler
    crawler = YouTubeCaptionCrawler(url)
    caption = crawler.get_caption()
    sentences = crawler.split_sentences()
    
    # Extract keywords using KeywordSummarizer
    summarizer = KeywordSummarizer(tokenize=baruen_noun_tokenizer, min_count=min_count, min_cooccurrence=min_cooccurrence) # bareun_tokenizer 활성화 시에 common.utils의 bareun_tokenizer 안내사항 참고
    keywords = summarizer.summarize(sentences, topk=topk)
    
    # Keyword data
    keyword_data = []
    for keyword in keywords:
        if keyword[0] not in korean_stopwords:
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


def tfidf_keywords(url, 
                   topk=20, 
                   file_path=None):
    '''
    TF-IDF로 유튜브 영상 자막 중에서 중요 키워드 추출
    
    Args:
        url (str): YouTube 영상 URL
        topk (int): 추출할 키워드 수
        stopwords_path (str): 불용어 파일 경로
    '''
    
    # Fetch captions by YouTubeCaptionCrawler
    crawler = YouTubeCaptionCrawler(url)
    caption = crawler.get_caption()
    sentences = crawler.split_sentences()
    
    # Korean stopwords 불러오기
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), 'korean_stopwords.json')
    
    with open(file_path, 'r', encoding='utf-8') as file:
        korean_stopwords = json.load(file)
    
    # TF-IDF Vectorizer 설정, 바른형태소 분석기를 명사 토크나이저로 사용
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.85, 
        stop_words=korean_stopwords, 
        tokenizer=baruen_noun_tokenizer
    )
    
    # TF-IDF 매트릭스 생성
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    
    # 단어와 TF-IDF 점수 매핑
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1  # 전체 자막에서 단어별 TF-IDF 점수 합계
    tfidf_dict = dict(zip(feature_names, tfidf_scores))
    
    # TF-IDF 점수가 높은 상위 topk 키워드 추출
    sorted_keywords = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)
    keywords = sorted_keywords[:topk]
    
    # 키워드 데이터 생성
    keyword_data = []
    for keyword in keywords:
        summary_entry = {
            'keyword': keyword[0],
            'score': keyword[1]
        }
        keyword_data.append(summary_entry)
    
    # Output file path
    output_file = 'tfidf_keywords.json'
    
    # Data to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(keyword_data, f, ensure_ascii=False, indent=4)
    
    # Return the path to the JSON file
    return output_file