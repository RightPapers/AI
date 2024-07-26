# coding: utf-8
import torch
import random
import json
import pickle

import numpy as np
import regex as re

from sklearn.model_selection import train_test_split
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from collections import Counter
from scipy.sparse import csr_matrix
from bareunpy import Tagger
from common.keys import my_keys

def fixSEED(seed, deterministic=True):
    '''
    파이토치, 넘파이, 랜덤 시드를 고정하는 함수
    
    Args:
        seed: 시드 값
        deterministic: cudnn.deterministic 활성화 여부
    '''
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def is_gpu_memory_availabe(empty=False):
    '''
    GPU 메모리 사용량을 확인하고 비우는 함수
    
    Args:
        empty: GPU 메모리 비우기 여부
    '''
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory
        current_mem = torch.cuda.memory_allocated(0)
        free_mem = total_mem - current_mem
        
        print(f'남은 GPU 메모리: {free_mem / (1024 ** 3):.2f} GB')
        
        if empty:
            torch.cuda.empty_cache()
            after_mem = torch.cuda.memory_allocated(0)
            print(f'GPU 캐시에서 메모리 삭제 후 메모리: {after_mem / (1024 ** 3):.2f} GB')
    else:
        print('CUDA 사용 불가')

def callData(path: str):
    '''
    pickle 파일을 불러오고 레이블 개수 확인하는 함수
    
    Args:
        path(str) : 파일 경로
    '''
    
    with open(path, "rb") as file:
        data = pickle.load(file)

    zero = data[data['label']==0]
    non_zero = data[data['label']==1]

    print(f'레이블 0 개수 : {len(zero)}')
    print(f'레이블 1 개수 : {len(non_zero)}')

    return data

def splitData(data, SEED=42):
    '''
    데이터를 학습용, 검증용, 평가용으로 나누는 함수
    
    Args:
        data: 데이터 프레임
        SEED: 시드 값
    '''
    
    train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=SEED, stratify=data['label'])
    train_data, valid_data = train_test_split(train_val_data, test_size=0.25, random_state=SEED, stratify=train_val_data['label'])

    print('학습용 데이터 개수 : {}개 \n검증용 데이터 개수 : {}개 \n평가용 데이터 개수 : {}개'.format(len(train_data), len(valid_data), len(test_data)))
    del train_val_data # 메모리 절약을 위해 삭제
    return train_data, valid_data, test_data

def convert_time(start, end):
    '''
    아이폰 알람 설정처럼 사용자가 시작과 끝 시간을 입력하면 이를 crawler에서 활용할 수 있는 형태로 변환
    여기서는 입력 형태를 다음과 같다고 정의함 Ex) 00:00
    이후 시간 입력 형태 변화에 따라서 이 함수만 수정하면 됨
    +) start 혹은 end가 없다면 자동으로 처음과 끝을 지정하도록 추가해야 함
    
    Args:
        start: 영상 시작 시간
        end: 영상 끝 시간
    '''
    
    start_min = int(start.split(':')[0])
    start_sec = int(start.split(':')[1])
    
    end_min = int(end.split(':')[0])
    end_sec = int(end.split(':')[1])

    return start_min * 60 + start_sec, end_min * 60 + end_sec

class YouTubeCaptionCrawler:
    '''
    입력 받은 링크를 바탕으로 자동생성자막을 가져오는 매서드
    
    Args:
        url: 유튜브 영상 링크
        start: 자막 시작 시간 (Ex: 00:00)
        end: 자막 끝 시간 (Ex: 00:00)
    '''
    def __init__(self, url, api_key, start=None, end=None):
        self.url = url
        self.ending_markers = ['는다', '니다', '는구나', '구나', '데요', 
                               '네요', '러요', '냐고', '아요', '어요', 
                               '가요', '군요', '래요', '랍니다', '습니까', 
                               '이죠', '겠죠', '겠네요', '까요', '랍니다', 
                               '였어요', '였네요', '시다', '이다', '이야',
                               '이죠', '겠죠', '나요', '려나', '려고',
                               '가죠', '나다', '라고', '라니', '라네']
        self.api_key = api_key
        self.text = ""
        
        # 시간 지정이 있을 경우
        if start is not None and end is not None:
            self.start, self.end = convert_time(start, end)
        else:
            self.start, self.end = None, None

    ### helper functions
    
    def text_cleaning(self, text):
        if not isinstance(text, str):
            return text
        
        # Remove Special Texts
        text = re.sub(r'[☞▶◆#⊙※△▽▼□■◇◎☎○]+', '', text, flags=re.UNICODE)
        text = re.sub(r'〃', '', text)  
        text = re.sub(r'[\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Han}]+', '', text, flags=re.UNICODE)
        text = re.sub(r'\[([^\]]{1,9})\]', '', text)
        text = re.sub(r'\[([^\]]{10,})\]', r'\1', text)
        text = re.sub(r'<[^>]*>', '', text)
        text = re.sub(r'\(\s*\)', '', text)
        text = re.sub(r'[\r\n]+', ' ', text)

        return text

    def get_video_id(self, url):
        if 'youtube.com/shorts/' in url:
            video_id = re.search(r'shorts/(.*?)\?feature', url).group(1)
        else:
            try:
                video_id = re.search(r'(?<=v=)[\w-]+', url).group(0)
            except AttributeError:
                video_id = re.search(r"youtu\.be\/([^/?]+)", url).group(1)
                
        return video_id
    
    def get_metadata(self, video_id):
        youtube = build('youtube', 'v3', developerKey=self.api_key)
        try:
            video_response = youtube.videos().list(
                part='snippet,statistics', 
                id=video_id
            ).execute()
            
            video_details = video_response['items'][0]['snippet']
            video_stats = video_response['items'][0]['statistics']
            
            video_title = video_details['title']
            upload_date = video_details['publishedAt']
            channel_id = video_details['channelId']
            channel_title = video_details['channelTitle']
            like_count = video_stats.get('likeCount', '0')
            hashtags = video_details.get('tags', [])
            thumbnails = video_details['thumbnails']['high']['url']
            
            details = {
                'video_title': video_title,
                'upload_date': upload_date,
                'channel_id': channel_id,
                'channel_title': channel_title,
                'like_count': like_count,
                'hashtags': hashtags,
                'thumbnails': thumbnails
            }
            
            return details
            
        except HttpError as e:
            print(f'HTTP Error {e.resp.status} has occurred: \n{e.content}')
            return None

    ### main functions
    
    def get_caption(self):
        # 영상 ID 가져오기
        video_id = self.get_video_id(self.url)
        
        # 자막 가져오기
        try: 
            caption = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko']) 
        except Exception:
            text = '해당 영상은 자동생성자막이 없거나 한국어 영상이 아닙니다.'
            return text
        
        # 시간 지정이 있을 경우
        if self.start is not None and self.end is not None:
            text = ' '.join([c['text'] for c in caption if self.start <= c['start'] <= self.end])
        # 시간 지정이 없을 경우
        else:
            text = ' '.join([c['text'] for c in caption])
        
        # 자막 전처리
        self.text = self.text_cleaning(text)
        
        return self.text

    def split_sentences(self):
        sentences = []
        start = 0
        for i in range(len(self.text)):
            for ending_marker in self.ending_markers:
                if self.text[i:].startswith(ending_marker):
                    sentence = self.text[start:i + len(ending_marker)].strip()
                    if sentence:
                        sentences.append(sentence)
                    start = i + len(ending_marker)
                    break

        last_sentence = self.text[start:].strip()
        if last_sentence:
            sentences.append(last_sentence)

        return sentences
    
    def save_to_json(self, file_path):
        video_id = self.get_video_id(self.url)
        details = self.get_metadata(video_id)
        captions = self.get_caption()
        if details:
            details['captions'] = captions
            details['ID'] = video_id
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(details, f, ensure_ascii=False, indent=4)
            print(f'Successfully saved JSON to {file_path}')
        else:
            print('JSON file not saved')
            
class YouTubeDataFetcher:
    '''
    유튜브 API를 활용하여 메타데이터와 자막을 가져오는 클래스
    Args:
        api_key: 유튜브 API 키
    '''
    def __init__(self, api_key):
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)

    def get_channel_id(self, youtuber_name):
        try:
            request = self.youtube.search().list(
                part='snippet',
                q=youtuber_name,
                type='channel'
            )
            response = request.execute()
            if not response['items']:
                return None
            return response['items'][0]['snippet']['channelId']
        except HttpError as e:
            if e.resp.status == 403:
                print("Quota exceeded. Please try again later.")
            else:
                print(f"An error occurred: {e}")
            return None

    def get_video_items(self, channel_id, max_results=200):
        try:
            channels_response = self.youtube.channels().list(
                part='contentDetails',
                id=channel_id
            ).execute()
            
            uploads_playlist_id = channels_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            videos = []
            next_page_token = None

            while len(videos) < max_results:
                playlist_items_response = self.youtube.playlistItems().list(
                    part='snippet',
                    playlistId=uploads_playlist_id,
                    maxResults=50,
                    pageToken=next_page_token
                ).execute()
                
                for item in playlist_items_response['items']:
                    if len(videos) >= max_results:
                        break
                    video_id = item['snippet']['resourceId']['videoId']
                    video_title = item['snippet']['title']
                    thumbnail_url = item['snippet']['thumbnails']['high']['url']
                    
                    videos.append({
                        'channel_id': channel_id,
                        'title': video_title,
                        'video_id': video_id,
                        'thumbnail': thumbnail_url
                    })
                
                next_page_token = playlist_items_response.get('nextPageToken')
                
                if not next_page_token:
                    break

            return videos
        except HttpError as e:
            if e.resp.status == 403:
                print("Quota exceeded. Please try again later.")
            else:
                print(f"An error occurred: {e}")
            return []

    def get_caption(self, video_id):
        try: 
            caption = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko']) 
        except: 
            return '⚠ 자막 없음'
        else: 
            text = ' '.join([item['text'] for item in caption])
            return self.cleaning_text(text)

    @staticmethod
    def cleaning_text(text):
        if not isinstance(text, str):
            return text
        
        text = re.sub(r'[☞▶◆#⊙※△▽▼□■◇◎☎○]+', '', text, flags=re.UNICODE)
        text = re.sub(r'〃', '', text)  
        text = re.sub(r'[\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Han}]+', '', text, flags=re.UNICODE)
        text = re.sub(r'\[([^\]]{1,9})\]', '', text)
        text = re.sub(r'\[([^\]]{10,})\]', r'\1', text)
        text = re.sub(r'<[^>]*>', '', text)
        text = re.sub(r'\(\s*\)', '', text)

        return text

# Below is for the textrank
def scan_vocabulary(sents, tokenize=None, min_count=2):
    """
    Arguments
    ---------
    sents : list of str
        Sentence list
    tokenize : callable
        tokenize(str) returns list of str
    min_count : int
        Minumum term frequency

    Returns
    -------
    idx_to_vocab : list of str
        Vocabulary list
    vocab_to_idx : dict
        Vocabulary to index mapper.
    """
    counter = Counter(w for sent in sents for w in tokenize(sent))
    counter = {w:c for w,c in counter.items() if c >= min_count}
    idx_to_vocab = [w for w, _ in sorted(counter.items(), key=lambda x:-x[1])]
    vocab_to_idx = {vocab:idx for idx, vocab in enumerate(idx_to_vocab)}
    return idx_to_vocab, vocab_to_idx

def tokenize_sents(sents, tokenize):
    """
    Arguments
    ---------
    sents : list of str
        Sentence list
    tokenize : callable
        tokenize(sent) returns list of str (word sequence)

    Returns
    -------
    tokenized sentence list : list of list of str
    """
    return [tokenize(sent) for sent in sents]

def vectorize(tokens, vocab_to_idx):
    """
    Arguments
    ---------
    tokens : list of list of str
        Tokenzed sentence list
    vocab_to_idx : dict
        Vocabulary to index mapper

    Returns
    -------
    sentence bow : scipy.sparse.csr_matrix
        shape = (n_sents, n_terms)
    """
    rows, cols, data = [], [], []
    for i, tokens_i in enumerate(tokens):
        for t, c in Counter(tokens_i).items():
            j = vocab_to_idx.get(t, -1)
            if j == -1:
                continue
            rows.append(i)
            cols.append(j)
            data.append(c)
    n_sents = len(tokens)
    n_terms = len(vocab_to_idx)
    x = csr_matrix((data, (rows, cols)), shape=(n_sents, n_terms))
    return x

def baruen_tokenizer(s):
    pos_list = ['NNG', 'NNP', 'NP', 'VV', 'VA', 'MAG', 'MMA', 'MMD', 'MMN', 'XPN', 'XSN', 'XSV', 'XSA']
    API_KEY = my_keys('bareun')
    baruen_tagger = Tagger(API_KEY, 'localhost')
    
    return [token for token, tag in baruen_tagger.pos(s) if tag in pos_list]