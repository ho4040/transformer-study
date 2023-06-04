# Learn

## TorchText

TorchText는 PyTorch의 사전 학습 과정에 도움이 되는 텍스트 데이터를 로딩하고 처리하기 위한 도구로서, NLP (Natural Language Processing)을 위한 여러 유용한 기능들을 제공합니다. 이 라이브러리의 핵심 기능들에는 다음과 같은 것들이 포함됩니다. 

* 데이터 로딩: TorchText는 텍스트 파일, JSON 데이터, CSV 데이터 등 다양한 형태의 원시 텍스트 데이터를 로딩하는 기능을 제공합니다.

*  토큰화: TorchText는 문장을 개별 토큰으로 분리하는 여러 가지 방법을 지원합니다. 여기에는 공백 기반의 토큰화, 정규 표현식 기반의 토큰화, Spacy, NLTK 같은 외부 라이브러리 기반의 토큰화 방법이 포함됩니다.

* 어휘집 생성: TorchText는 토큰화된 텍스트 데이터를 바탕으로 어휘집(Vocabulary)을 생성하는 기능을 제공합니다. 어휘집은 각 토큰에 고유한 숫자 ID를 부여하는 역할을 합니다.

* 데이터 배치 처리: TorchText는 텍스트 데이터를 미니 배치로 구성하고, 패딩 등의 데이터 전처리를 자동으로 수행하는 기능을 제공합니다.

* 데이터셋과 데이터 로더: TorchText는 PyTorch의 Dataset과 DataLoader 클래스를 확장하여, 텍스트 데이터에 특화된 기능을 제공합니다.

* 도메인 특화 데이터셋: TorchText는 IMDB, SNLI, Multi30k 등의 여러 NLP 작업을 위한 사전 구성된 데이터셋을 제공합니다.


## Datasets

Hugging Face의 Datasets 라이브러리는 대량의 NLP 데이터셋에 빠르고 쉽게 접근하고, 이를 위한 효율적인 데이터 처리 기능을 제공하는 Python 라이브러리입니다. 다음은 주요 기능과 특징들에 대한 개요입니다.

* 방대한 데이터셋: Datasets 라이브러리는 100개 이상의 자연어 처리(NLP) 데이터셋에 대한 접근을 제공하며, 이 수는 계속해서 증가하고 있습니다. 이들에는 SQuAD, GLUE, WikiText, CommonVoice, WMT 등의 많은 고명한 데이터셋이 포함됩니다.

* 데이터 처리 및 변환: Datasets 라이브러리는 맵(map), 필터(filter), 셔플(shuffle), 그리고 데이터를 분할하는 기능 등을 제공합니다. 이를 통해 데이터를 효율적으로 처리하고 변환할 수 있습니다.

* 데이터 효율성: Datasets 라이브러리는 Apache Arrow를 사용하여 데이터를 메모리에 효율적으로 저장하고 처리합니다. 이를 통해 대용량 데이터셋을 빠르게 로드하고 변환할 수 있습니다.

* 데이터 형식: Datasets 라이브러리는 다양한 형식의 데이터를 지원합니다. 이는 CSV, JSON, Parquet, txt, TSV 등을 포함합니다.

* 통합된 메타데이터: Datasets 라이브러리는 각 데이터셋에 대한 메타데이터를 제공합니다. 이를 통해 데이터의 레이블, 기능, 출처 등을 이해하고, 데이터를 보다 쉽게 처리할 수 있습니다.

* 머신러닝 연동: Datasets 라이브러리는 Hugging Face의 Transformers 라이브러리와 잘 연동됩니다. 이를 통해 텍스트 분류, 질문 응답, 기계 번역 등의 다양한 NLP 작업을 수행할 수 있습니다.

# requirements

python 3.10 사용.

## 영어 토크나이저 

```bash
python -m spacy download en
```

## 한국어 토크나이저

```bash
python -m spacy download ko_core_news_sm
```


```bash
pip install -r requirements.txt
python -m spacy download en 
python -m spacy download ko_core_news_sm # 한국어 토크나이저 
```