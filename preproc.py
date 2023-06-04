from datasets import load_dataset # load_dataset 사용법 :  https://huggingface.co/docs/datasets/v2.12.0/en/package_reference/loading_methods#datasets.load_dataset
from torchtext.data import Field, TabularDataset
import pandas as pd
import os
import pickle
import tokenizer

def download(force=False):
    if force or os.path.exists("train.csv") == False:        
        dataset = load_dataset("msarmi9/korean-english-multitarget-ted-talks-task")

        train_dataset = dataset["train"]
        valid_dataset = dataset["validation"]
        test_dataset = dataset['test']

        print(f"학습 데이터셋(training dataset) 크기: {len(train_dataset)}개")
        print(f"평가 데이터셋(validation dataset) 크기: {len(valid_dataset)}개")
        print(f"테스트 데이터셋(testing dataset) 크기: {len(test_dataset)}개")

        def extract_src_and_dst(dataset):
            en = [example['english'] for example in dataset]
            ko = [example['korean'] for example in dataset]
            return en, ko

        train_src, train_dst = extract_src_and_dst(dataset['train'])
        valid_src, valid_dst = extract_src_and_dst(dataset['validation'])
        test_src, test_dst = extract_src_and_dst(dataset['test'])

        # Create pandas DataFrames
        train_df = pd.DataFrame(list(zip(train_src, train_dst)), columns=['src', 'dst'])
        valid_df = pd.DataFrame(list(zip(valid_src, valid_dst)), columns=['src', 'dst'])
        test_df = pd.DataFrame(list(zip(test_src, test_dst)), columns=['src', 'dst'])

        train_df.to_csv('train.csv', index=False)
        valid_df.to_csv('valid.csv', index=False)
        test_df.to_csv('test.csv', index=False)
    else:
        print("already downloaded. skip download")


def preproc(force=False):
    VOCAB_PATH = "vocab.pkl"

    # Define Fields
    # Field : 데이터 전처리를 담당합니다. 텍스트 전처리 과정을 캡슐화 하는 것을 도와주는 클래스
    SRC = Field(sequential=True, tokenize=tokenizer.tokenize_en, lower=True, init_token="", eos_token="", batch_first=True)
    DST = Field(sequential=True, tokenize=tokenizer.tokenize_ko, lower=True, init_token="", eos_token="", batch_first=True)

    # Create TabularDatasets
    # CSV 파일로 저장된 데이터를 TabularDataset 에 불러온다.
    # CSV 와 Field 를 연결.
    print("load TabularDataset from csv")
    train_dataset, valid_dataset, test_dataset = TabularDataset.splits(
        path='.', 
        train='train.csv', 
        validation='valid.csv', 
        test='test.csv', 
        format='csv', 
        fields=[('src', SRC), ('dst', DST)],  # csv에서 어떤 컬럼이 어떤 전처리기와 연결될지 지정한다.
        skip_header=True
    )

    if force or os.path.exists(VOCAB_PATH) == False:
        print("build vocab")
        # 어휘집을 생성한다. (최소 2회 이상 등장한 단어만)
        SRC.build_vocab(train_dataset, min_freq=2) 
        DST.build_vocab(train_dataset, min_freq=2)
        with open(VOCAB_PATH, 'wb') as f:
            pickle.dump({"SRC_vocab":SRC.vocab, "DST_vocab":DST.vocab }, f)
    else:
        print("load vocab from "+VOCAB_PATH)
        with open(VOCAB_PATH, 'rb') as f:
            obj = pickle.load(f)
            SRC.vocab = obj["SRC_vocab"]
            DST.vocab = obj["DST_vocab"]
    
    train_max_length = max(len(example.src) for example in train_dataset)
    valid_max_length = max(len(example.src) for example in valid_dataset)
    test_max_length = max(len(example.src) for example in test_dataset)
    src_max_length = max(train_max_length, valid_max_length, test_max_length)

    return SRC, DST, train_dataset, valid_dataset, test_dataset, src_max_length

if __name__ == "__main__":
    download()
    SRC, DST, train_dataset, valid_dataset, test_dataset = preproc()
    print(f"English vocab size: {len(SRC.vocab)}")
    print(f"Korean vocab size: {len(DST.vocab)}")

    
    # print(DST.vocab.stoi["abcabc"]) # 없는 단어: 0
    # print(DST.vocab.stoi[DST.pad_token]) # 패딩(padding): 1
    # print(DST.vocab.stoi[""]) # : 2
    # print(DST.vocab.stoi["안녕"])
    # print(DST.vocab.stoi["세상"])
    # print(DST.vocab.stoi["대학원"])
    # print(DST.vocab.stoi["대학원에"])
    # print(DST.vocab.stoi["대학원으로"])