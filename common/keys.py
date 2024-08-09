# coding: utf-8
def my_keys(usage='youtube', alter=False):
    '''
    API key를 반환하는 함수
    
    Args:
        usage : 어떤 API 서비스 이용할지
        alternative : 대체 키를 사용할지 여부(youtube의 경우만 해당)
    '''
    
    if usage == 'youtube':
        key = 'AIzaSyBXRUyGKQAgnHB3O6AO8EEEd6pG9sl7OY4'
        if alter:
            key = 'AIzaSyC4h9FzEi5aStNer8Hc8KASdwxD_FuY_94'
        
    elif usage == 'openai':
        key = 'sk-proj-doWEvkWrlm0QTwpW2jYDT3BlbkFJwbdHkomvlUkLPEKhNNKK'
        
    elif usage == 'bareun':
        key = "koba-TOE4CCA-76WULKY-WLGME4Y-NOXDU4I"
        
    elif usage == 'naver':
        key = {'client_id': '68RoCmyR17urwap_vvNC', 'client_sever':'WDlfFKv_Qk'}
        
    return key
    