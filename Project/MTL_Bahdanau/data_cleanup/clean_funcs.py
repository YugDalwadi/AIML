import re
import spacy
import time
from typing import Tuple, List

Sentence = List[str]
Corpus = List[Sentence]

EOS = "2"
SOS = "1"
PAD = "0"
UNK = "3"

MAX_LEN = 11  # max_len = 9, but 2 special tokens considered.
MIN_FREQ = 5

en_tok = spacy.load("en_core_web_sm")
fr_tok = spacy.load("fr_core_news_sm")


def log_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.5f}s to complete")
        return result
    return wrapper


@log_time
def dataCleaner(text) -> Sentence:
    # Precompile regex patterns for speed
    pattern1 = re.compile('\u202f')
    pattern2 = re.compile('\xa0')
    # Use list comprehension for faster processing
    return [
        pattern2.sub(' ', pattern1.sub(' ', sentence)).lower()
        for sentence in text
    ]


@log_time
def tokenizer(text: Sentence, lang: str) -> Corpus:
    if lang == "eng":
        return [[token.text for token in doc] for doc in en_tok.pipe(text, batch_size=4096, n_process=-1)]
    elif lang == "fra":
        return [[token.text for token in doc] for doc in fr_tok.pipe(text, batch_size=4096, n_process=-1)]

    else:
        return [[""]]


@log_time
def padData(data: Corpus) -> Corpus:
    for ix in range(len(data)):
        dataLen = len(data[ix])
        if (dataLen < MAX_LEN):
            for _ in range(MAX_LEN-dataLen):
                data[ix].append(PAD)

    return data


@log_time
def insertSpecialTokens(data: Corpus) -> Corpus:
    for ix in range(len(data)):
        data[ix].append(EOS)
        data[ix].insert(0, SOS)

    return data


@log_time
def trimData(source_tok: Corpus, target_tok: Corpus) -> Tuple[Corpus, Corpus]:
    '''
    Trims sentences to MAX_LEN
    '''
    to_remove = []
    for i in range(len(source_tok)):
        if len(source_tok[i]) > MAX_LEN or len(target_tok[i]) > MAX_LEN:
            to_remove.append(i)

    for i in reversed(to_remove):  # Removed from end to avoid index issues
        del source_tok[i]
        del target_tok[i]

    return source_tok, target_tok


@log_time
def getWordEncAndFreq(data: Corpus) -> Tuple[dict, dict]:
    word_freq = {}
    word_enc = {'0': 0, '1': 1, '2': 2, '3': 3}
    counter = 4  # counters start from 4 due to the 4 unique tokens we have defined above

    for ix in range(len(data)):
        for word in data[ix]:
            word = str(word)

            if (word not in word_freq.keys()):
                word_freq[word] = 1

                if (word not in ("0", "1", "2", "3")):
                    word_enc[word] = counter
                    counter += 1
            else:
                word_freq[word] += 1

    return word_enc, word_freq


@log_time
def replaceUnk(data: Corpus, word_freq: dict) -> Corpus:
    unk_words = []

    for word in word_freq:
        if (word_freq[word] <= MIN_FREQ):
            unk_words.append(word)

    # Convert unk_eng and unk_fra to sets for O(1) lookup
    unk_word_set = set(unk_words)

    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] in unk_word_set:
                data[i][j] = UNK

    return data


@log_time
def encodeData(data: Corpus, wordEnc: dict) -> List[List[int]]:
    source_tok_enc = [[] for _ in range(len(data))]

    for i in range(len(data)):
        for word in (data[i]):
            encoding = wordEnc[word]
            source_tok_enc[i].append(encoding)

    return source_tok_enc
