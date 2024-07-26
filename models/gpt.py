# coding : utf-8
import json
from openai import OpenAI
from common.keys import my_keys

# OpenAI GPT for Video Summarization
class VS_GPT:
    def __init__(self, file_path, model='gpt-3.5-turbo'):
        self.api_key = my_keys('openai')
        self.client = OpenAI(api_key=self.api_key)
        self.file_path = file_path  # json file_path
        self.model = model  # default: gpt-3.5-turbo / possible: gpt-4-turbo, gpt-4o-mini
    
    def _create_prompt(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        prompt = ""
        for i, line in enumerate(data):
            sentence_info = f'{i+1}번째 문장의 중요도는 {line["importance"]}이며, 내용은 "{line["sentence"]}"입니다.\n'
            prompt += sentence_info
        
        return prompt
    
    def generate_summary(self):
        prompt = self._create_prompt()
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "너는 주어진 문장의 중요도와 내용을 바탕으로 비디오를 요약해주는 기능을 가지고 있어."},
                {"role": "user", "content": "다음 정보를 바탕으로 짜임새 있는 요약문을 작성해줘."},
                {"role": "user", "content": f"{prompt}"}
            ]
        )
        
        return completion.choices[0].message.content

    
    