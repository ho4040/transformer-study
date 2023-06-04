#%%
import spacy

#%%
spacy_en = spacy.load('en_core_web_sm') # 영어 토큰화(tokenization)
#%%
spacy_ko = spacy.load('ko_core_news_sm') # 한국어 토큰화(tokenization)
# %%
tokenized = spacy_en.tokenizer("I am a graduate student.")
for i, token in enumerate(tokenized):
    print(f"인덱스 {i}: {token.text}")
#%%    
tokenized = spacy_ko.tokenizer("나는 대학원에 다닙니다.")
for i, token in enumerate(tokenized):
    print(f"인덱스 {i}: {token.text}")
# %%
# 한국어 문장을 토큰화 하는 함수 (순서를 뒤집지 않음)
def tokenize_ko(text):
    return [token.text for token in spacy_ko.tokenizer(text)]

# 영어(English) 문장을 토큰화 하는 함수
def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]
# %%
# HuggingFace Datasets 패키지에서 제공하는 한국어-영어 데이터셋 준비
# load_dataset 사용법 :  https://huggingface.co/docs/datasets/v2.12.0/en/package_reference/loading_methods#datasets.load_dataset

from datasets import load_dataset
dataset = load_dataset("msarmi9/korean-english-multitarget-ted-talks-task")

# %%
# Datasets 객체는 딕셔너리 형태 접근이 가능함
train_dataset = dataset["train"]
valid_dataset = dataset["validation"]
test_dataset = dataset['test']
print(train_dataset[0])
# %%
print(f"학습 데이터셋(training dataset) 크기: {len(train_dataset)}개")
print(f"평가 데이터셋(validation dataset) 크기: {len(valid_dataset)}개")
print(f"테스트 데이터셋(testing dataset) 크기: {len(test_dataset)}개")
# %%
def extract_src_and_dst(dataset):
    en = [example['english'] for example in dataset]
    ko = [example['korean'] for example in dataset]
    return en, ko

train_src, train_dst = extract_src_and_dst(dataset['train'])
valid_src, valid_dst = extract_src_and_dst(dataset['validation'])
test_src, test_dst = extract_src_and_dst(dataset['test'])
# %%
# HuggingFace Datasets으로부터 얻은 데이터를 CSV 파일로 저장한다. 
# 이후 TorchText 에서 불러와 이용하기 위함.

import pandas as pd
# Create pandas DataFrames
train_df = pd.DataFrame(list(zip(train_src, train_dst)), columns=['src', 'dst'])
valid_df = pd.DataFrame(list(zip(valid_src, valid_dst)), columns=['src', 'dst'])
test_df = pd.DataFrame(list(zip(test_src, test_dst)), columns=['src', 'dst'])

# Save to csv
train_df.to_csv('train.csv', index=False)
valid_df.to_csv('valid.csv', index=False)
test_df.to_csv('test.csv', index=False)

#%%
from torchtext.data import Field, TabularDataset

# CSV 파일로 저장된 데이터를 TabularDataset 에 불러온다.
# Field : 데이터 전처리를 담당합니다. 텍스트 전처리 과정을 캡슐화 하는 것을 도와주는 클래스

# Define Fields
SRC = Field(sequential=True, tokenize=tokenize_en, lower=True, init_token="", eos_token="", batch_first=True)
DST = Field(sequential=True, tokenize=tokenize_ko, lower=True, init_token="", eos_token="", batch_first=True)

# Create TabularDatasets
# CSV 와 Field 를 연결.
train_dataset, valid_dataset, test_dataset = TabularDataset.splits(
    path='.', 
    train='train.csv', 
    validation='valid.csv', 
    test='test.csv', 
    format='csv', 
    fields=[('src', SRC), ('dst', DST)],  # csv에서 어떤 컬럼이 어떤 전처리기와 연결될지 지정한다.
    skip_header=True
)

#%% 
# 어휘집을 생성한다. (최소 2회 이상 등장한 단어만)
SRC.build_vocab(train_dataset, min_freq=2) 
DST.build_vocab(train_dataset, min_freq=2)

# %%
# 어휘집 크기 확인
print(f"English vocab size: {len(SRC.vocab)}")
print(f"Korean vocab size: {len(DST.vocab)}")
# %%
print(DST.vocab.stoi["abcabc"]) # 없는 단어: 0
print(DST.vocab.stoi[DST.pad_token]) # 패딩(padding): 1
print(DST.vocab.stoi[""]) # : 2
print(DST.vocab.stoi["안녕"])
print(DST.vocab.stoi["세상"])
print(DST.vocab.stoi["대학원"])
print(DST.vocab.stoi["대학원에"])
print(DST.vocab.stoi["대학원으로"])
# %%
# 학습에 필요한 HyperParameters
BATCH_SIZE = 1
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(DST.vocab)
HIDDEN_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
LEARNING_RATE = 0.0005
CLIP = 1
#%%
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#%%
from torchtext.data import BucketIterator
"""
# BucketIterator : 
BucketIterator 는 TorchText에서 제공하는 데이터 로더의 한 종류로, 특히 텍스트 데이터를 효율적으로 처리하는 데 유용합니다.
텍스트 데이터의 특성 상, 각각의 예제 (일반적으로 문장 또는 문서)는 서로 다른 길이를 가질 수 있습니다. 이는 배치 처리를 어렵게 만드는데, 딥러닝 모델은 입력 데이터의 길이가 일정해야 하기 때문입니다.
이 문제를 해결하기 위해, 일반적으로 가장 긴 예제에 맞춰서 더 짧은 예제들에 패딩을 추가하는 방식을 사용합니다. 하지만, 매우 긴 예제와 짧은 예제가 같은 배치에 들어가게 되면, 많은 양의 패딩이 필요하게 되어 메모리 낭비와 계산 비효율성을 초래합니다.
BucketIterator는 이러한 문제를 해결하기 위해 사용되며, 비슷한 길이를 가진 예제들을 함께 묶어줍니다. 이를 통해 각 배치 내에서 필요한 패딩의 양을 최소화하게 되어, 메모리 사용량을 줄이고 계산 효율성을 향상시킵니다.
"""
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_dataset, valid_dataset, test_dataset),
    batch_size=BATCH_SIZE,
    sort_key=lambda row: len(row.src),
    sort_within_batch=True,
    device=device)
# %%
for i, batch in enumerate(train_iterator):
    src = batch.src # 영어
    dst = batch.dst # 한국어
    print(f"첫 번째 배치 크기: {src.shape}")
    # 현재 배치에 있는 하나의 문장에 포함된 정보 출력
    for i in range(src.shape[1]):
        print(f"인덱스 {i}: {src[0][i].item()}") # 여기에서는 [Seq_num, Seq_len]
    # 첫 번째 배치만 확인
    break
# %%
from transformer import Encoder, Decoder, Transformer

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = DST.vocab.stoi[DST.pad_token]

# 인코더(encoder)와 디코더(decoder) 객체 선언
enc = Encoder(INPUT_DIM, HIDDEN_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
dec = Decoder(OUTPUT_DIM, HIDDEN_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)

# Transformer 객체 선언
model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters')
# %%
from torch import nn
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
model.apply(initialize_weights)
# %%
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
# %%
def train(model, iterator, optimizer, criterion, clip, loopLimit=-1):
    model.train() # 학습 모드
    epoch_loss = 0
    # 전체 학습 데이터를 확인하며
    for i, batch in enumerate(iterator):
        src = batch.src
        dst = batch.dst
        optimizer.zero_grad() # 그라디언트 초기화
        # 출력 단어의 마지막 인덱스()는 제외
        # 입력을 할 때는 부터 시작하도록 처리
        
        # forward
        output, _ = model(src, dst[:,:-1]) # teacher forcing

        # output: [배치 크기, trg_len - 1, output_dim]
        # trg: [배치 크기, trg_len]
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        # 출력 단어의 인덱스 0()은 제외
        dst = dst[:,1:].contiguous().view(-1)

        # output: [배치 크기 * trg_len - 1, output_dim]
        # trg: [배치 크기 * trg len - 1]
        # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
        loss = criterion(output, dst) 
        loss.backward() # 기울기(gradient) 계산

        # 기울기(gradient) clipping 진행 ??
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # 파라미터 업데이트
        optimizer.step()

        # 전체 손실 값 계산
        epoch_loss += loss.item()

        if loopLimit != -1 and i >= loopLimit:
            break # do just once.

    return epoch_loss / len(iterator)

#train(model, train_iterator, optimizer, criterion, CLIP, 10)
# %%
# 모델 평가(evaluate) 함수
def evaluate(model, iterator, criterion):
    model.eval() # 평가 모드
    epoch_loss = 0
    with torch.no_grad():
        # 전체 평가 데이터를 확인하며
        for i, batch in enumerate(iterator):
            src = batch.src
            dst = batch.dst

            # 출력 단어의 마지막 인덱스()는 제외
            # 입력을 할 때는 부터 시작하도록 처리
            output, _ = model(src, dst[:,:-1])
            # output: [배치 크기, trg_len - 1, output_dim]
            # trg: [배치 크기, trg_len]
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            # 출력 단어의 인덱스 0()은 제외
            dst = dst[:,1:].contiguous().view(-1)
            # output: [배치 크기 * trg_len - 1, output_dim]
            # trg: [배치 크기 * trg len - 1]
            # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
            loss = criterion(output, dst)
            # 전체 손실 값 계산
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

evaluate(model, test_iterator, criterion)
# %%
import math
import time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs