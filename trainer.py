
import preproc
import torch
from torch import nn
import torch.optim as optim
from torchtext.data import BucketIterator
from transformer import Encoder, Decoder, Transformer
from tqdm import tqdm
import math
import time
import math
import random


preproc.download()
SRC, DST, train_dataset, valid_dataset, test_dataset, max_langth = preproc.preproc()

# 학습에 필요한 HyperParameters
BATCH_SIZE = 2
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
N_EPOCHS = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_dataset, valid_dataset, test_dataset),
    batch_size=BATCH_SIZE,
    sort_key=lambda row: len(row.src),
    sort_within_batch=True,
    device=device)


SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = DST.vocab.stoi[DST.pad_token]

# 인코더(encoder)와 디코더(decoder) 객체 선언
enc = Encoder(INPUT_DIM, HIDDEN_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device, max_langth)
dec = Decoder(OUTPUT_DIM, HIDDEN_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device, max_langth)
model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

model.apply(initialize_weights)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

# 모델 평가(evaluate) 함수
def evaluate(model, iterator, criterion):
    model.eval() # 평가 모드
    epoch_loss = 0
    with torch.no_grad():
        # 전체 평가 데이터를 확인하며
        for i, batch in enumerate(tqdm(iterator)):
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


def train(model, iterator, optimizer, criterion, clip, loopLimit=-1):
    model.train() # 학습 모드
    epoch_loss = 0
    # 전체 학습 데이터를 확인하며
    for i, batch in enumerate(tqdm(iterator)):
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


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time() # 시작 시간 기록

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time() # 종료 시간 기록
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'transformer_german_to_english.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
    print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {math.exp(valid_loss):.3f}')