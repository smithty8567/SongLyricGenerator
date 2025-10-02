import collections
import json
import os
import re
import pandas as pd

from tqdm import tqdm


def string_cleaner(lyric):

    # Lowercases all the lyrics
    lyric = lyric.lower()

    # Gets rid of all names in brackets
    lyric = re.sub(r'\[.*]\n', '', lyric)

    # Gets rid of adlibs in parantheses
    # lyric = re.sub(r'\(.*\)\n', '', lyric)

    # Removes any repeated lines back to back
    lyric = re.sub(r'\b(\w+)(\n+\1)+', '\1', lyric)
    lyric = re.sub(r'\n+', '\n+', lyric)

    return lyric


def identify_tokens(lines, num_ngrams=3000):
    """
    Build a vocabulary of the most frequent character n-grams.

    - Counts all n-grams from length 1 to 5
    - Keeps only the top `num_ngrams`
    - Returns dictionaries for both token->id and id->token
    """
    ngram_counts = collections.Counter()

    for line in tqdm(lines, desc=f"Creating and indexing top {num_ngrams} n-grams"):
        length = len(line)
        for n in range(1, 6):  # n-grams of length 1 to 5
            for i in range(length - n + 1):
                ngram_counts[line[i:i + n]] += 1

    top_n = [token for token, _ in ngram_counts.most_common(num_ngrams - 4)]
    token2idx = {token: idx for idx, token in enumerate(top_n)}
    token2idx['^'] = len(token2idx)
    token2idx['$'] = len(token2idx)
    token2idx['%'] = len(token2idx)
    token2idx['\n'] = len(token2idx)
    idx2token = {idx: token for token, idx in token2idx.items()}
    return token2idx, idx2token


def tokenize(text, token2idx):
    """
    Convert text into a list of token IDs.

    Uses greedy longest-match-first tokenization with a max token length of 5.
    """
    text = text.lower()
    text = re.sub(r"[^a-z,.?!':; %$^()\n-]", '', text)
    tokens = []
    idx = 0
    while idx < len(text):
        for token_length in range(min(5, len(text) - idx), 0, -1):
            substring = text[idx: idx + token_length]
            if substring in token2idx:
                tokens.append(token2idx[substring])
                idx += token_length
                break
        else:
            # If no token matched (shouldn't usually happen), skip one char
            idx += 1
    return tokens


def tokenize_list(lines, token2idx):
    """
    Tokenize a list of lines and return them sorted by length.
    """
    tokenized_lines = [tokenize('^'+line+'$', token2idx) for line in tqdm(lines, desc="Tokenizing corpus")]
    return tokenized_lines


def process_corpus(song_file = "english_songs.csv"):

    # files = [f for f in os.listdir(corpus_dir) if f.endswith(".txt")]
    df = pd.read_csv(song_file)
    songs = df['lyrics']

    lyrics = []
    for song_lyrics in tqdm(songs, desc="Reading and cleaning files"):
        lines = song_lyrics.split('\n')
        cleaned_lines = [string_cleaner(line) for line in lines]
        if cleaned_lines:
            cleaned_lines[-1] += '%'
        lyrics.append('\n'.join(cleaned_lines))

    # Build vocab and tokenize corpus
    if all(os.path.exists(f) for f in ("token2idx.json", "idx2token.json")):
        with open("token2idx.json", 'r', encoding="utf-8") as f:
            token2idx = json.load(f)
        with open("idx2token.json", 'r', encoding="utf-8") as f:
            idx2token = {int(idx): token for idx, token in json.load(f).items()}

    else:
        token2idx, idx2token = identify_tokens(lyrics)

    print("Tokens identified")

    #Saving dictionaries
    print("Saving dictionaries")
    with open("token2idx.json", 'w', encoding="utf-8") as f:
        json.dump(token2idx, f)
    with open("idx2token.json", 'w', encoding="utf-8") as f:
        json.dump(idx2token, f)

    # Split at end of song (odd indexed songs)
    lines_odd = lyrics[1::2]
    # lines_odd = [lines_odd.strip() for lines_odd in lines_odd if 7 < len(lines_odd) < 250]

    # Getting tokenized odd songs
    tokens_odd = tokenize_list(lines_odd, token2idx)
    # Saving odd corpus as corpus1
    print("Saving odd corpus")
    with open("corpus1.json", 'w', encoding="utf-8") as f:
        json.dump(tokens_odd, f)
    # Clearing memory for next corpus
    del tokens_odd
    del lines_odd

    # Split at end of newline (even indexed songs)
    lines_even = lyrics[0::2]
    # lines_even = [lines_even.strip() for lines_even in lines_even if 7 < len(lines_even) < 250]
    # Getting tokenized even songs
    tokens_even = tokenize_list(lines_even, token2idx)
    # Saving even corpus as corpus2
    print("Saving even corpus")
    with open("corpus2.json", 'w', encoding="utf-8") as f:
        json.dump(tokens_even, f)


    return token2idx, idx2token


def get_cleaned_corpus():
    """
    Load processed corpus from disk if available, otherwise generate it.
    """
    if all(os.path.exists(f) for f in ("token2idx.json", "idx2token.json", "corpus1.json", "corpus2.json")):
        with open("corpus1.json", 'r', encoding="utf-8") as f:
            corpus1 = json.load(f)
        with open("corpus2.json", 'r', encoding="utf-8") as f:
            corpus2 = json.load(f)
        with open("token2idx.json", 'r', encoding="utf-8") as f:
            token2idx = json.load(f)
        with open("idx2token.json", 'r', encoding="utf-8") as f:
            idx2token = {int(idx): token for idx, token in json.load(f).items()}
        return corpus1, corpus2, token2idx, idx2token

    return process_corpus()


def get_dictionaries():
    """
    Load token dictionaries from disk if available, otherwise build them.
    """
    if all(os.path.exists(f) for f in ("token2idx.json", "idx2token.json")):
        with open("token2idx.json", 'r', encoding="utf-8") as f:
            token2idx = json.load(f)
        with open("idx2token.json", 'r', encoding="utf-8") as f:
            idx2token = {int(idx): token for idx, token in json.load(f).items()}
        return token2idx, idx2token

    return get_cleaned_corpus()[1:]
process_corpus()