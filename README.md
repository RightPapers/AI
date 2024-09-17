# AI 
이 레포지토리는 [Right Paper](https://right-paper.vercel.app/) 서비스에 탑재된 AI 모델과 각종 기능을 구현했습니다.


## 🌟 레포지토리 소개
---
- 유튜브 자동생성자막 및 메타데이터 크롤링
- 영상의 낚시성 여부를 판단하는 AI 모델
- 영상의 내용을 요약하는 GPT 모델
- 영상의 내용과 연관된 신문 기사를 검색

## 🤖 AI 모델 소개
---
Right Paper 서비스에 탑재된 낚시성 영상 탐지 AI 모델은 2가지 입니다.
AI 모델은 유튜브 영상의 자동생성자막을 입력받아, 해당 영상의 낚시성 확률을 출력합니다.
모델 학습에는 GCP의 L4 가상머신을 활용했습니다.

### Transformer Encoder를 변형한 자체 제작 모델
![그림1](https://github.com/user-attachments/assets/dc04b8db-2ac5-4633-bbe1-9179daf4c768)
- Transformer의 인코더 구조를 일부 변형하고, Feed Forward Network와 Classifier를 추가하여 이진분류를 수행합니다.
- ```klue/roberta-small```을 체크포인트로 활용했습니다.
- 모델의 최종 파라미터 개수는 68,812,290개(68.8M)입니다.

### RoBERTa 활용 모델
![그림2](https://github.com/user-attachments/assets/becf07b7-f154-4971-85c1-69f34358bc43)
- 허깅페이스의 ```AutoModelForSequenceClassification``` 라이브러리를 활용하여 시퀀스 분류에 최적화된 모델을 Fine-tuning했습니다.
- ```klue/roberta-large```을 체크포인트로 활용했습니다.
- Full Fine-tuning이 아닌 LoRA를 활용해서 Fine-tuning을 진행했습니다.
- 모델의 최종 파라미터 개수는 338,889,732개(338.9M)이며, LoRA를 통해 Fine-tuning한 파라미터 개수는 2,231,298개(2.23M)입니다.

### GPT 모델
OpenAI GPT 모델을 활용하여 자동생성자막을 빠르게 요약하는 모델입니다.
별도 모델 학습과 설계 없이 OpenAI API를 활용하였습니다.

## 📠 기능
---
Right Paper 서비스에 필요한 각종 기능을 구현했습니다.

### 유튜브 자동생성자막 및 메타데이터 크롤링
- ```YoutubeTranscriptApi```와 Youtube Data API V3를 사용하여 자동생성자막과 영상 제목, 썸네일 등을 가져옵니다.

### 영상의 내용과 연관된 신문 기사를 검색
- NAVER Search API를 사용하여 유튜브 영상의 주제와 유사한 신문 기사를 검색 후 반환합니다.
- 바른 형태소 분석기와 Okt를 활용하여 명사를 추출 후 검색어 쿼리를 작성합니다.

## ⚙️ 기술 스택
---
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"> <img src="https://img.shields.io/badge/googlecloud-F9AB00?style=for-the-badge&logo=googlecloud&logoColor=white"> 

## ✏️ 현재 개발 중인 내용
---
- 멀티모달
- 검색어 쿼리 생성 모델
- 사전 질의 차단 기능





