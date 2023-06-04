import spacy

spacy_en = spacy.load('en_core_web_sm') # 영어 토큰화(tokenization)
spacy_ko = spacy.load('ko_core_news_sm') # 한국어 토큰화(tokenization)

def tokenize_ko(text):
    return [token.text for token in spacy_ko.tokenizer(text)]

# 영어(English) 문장을 토큰화 하는 함수
def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]


if __name__ == "__main__":
    tokenized = spacy_en.tokenizer("I am a graduate student.")
    for i, token in enumerate(tokenized):
        print(f"인덱스 {i}: {token.text}")
    tokenized = spacy_ko.tokenizer("나는 대학원에 다닙니다.")
    for i, token in enumerate(tokenized):
        print(f"인덱스 {i}: {token.text}")