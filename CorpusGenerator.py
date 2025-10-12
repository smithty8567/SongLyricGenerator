import collections
import json
import os
import re
import pandas as pd
import random

from tqdm import tqdm

def identify_tokens(lines, num_ngrams=10000):

    ngram_counts = collections.Counter()

    for line in tqdm(lines, desc=f"Creating and indexing top {num_ngrams} n-grams"):
        if type(line) == float:
            # An empty line (nan) was created in the lyric list when splitting the song string
            continue
        length = len(line)
        for n in range(1, 9):  # n-grams of length 1 to 8
            for i in range(length - n + 1):
                ngram_counts[line[i:i + n]] += 1

    top_n = [token for token, _ in ngram_counts.most_common(num_ngrams - 2)]
    token2idx = {token: idx for idx, token in enumerate(top_n)}
    token2idx['^'] = len(token2idx)
    token2idx['$'] = len(token2idx)
    idx2token = {idx: token for token, idx in token2idx.items()}
    return token2idx, idx2token


def tokenize(text, token2idx):

    # Convert text into a list of token IDs
    # Uses greedy longest-match-first tokenization with a max token length of 8

    text = text.lower()
    text = re.sub(r"[^a-z,.?!':; %$^()\n-]", '', text)
    tokens = []
    idx = 0
    while idx < len(text):
        for token_length in range(min(8, len(text) - idx), 0, -1):
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

    # Tokenize a list of song lyrics
    tokenized_lines = [tokenize('^'+line+'$', token2idx) for line in lines]
    return tokenized_lines


def process_corpus(song_file = "cleaned_songs.csv", num_songs = 1000000):

    df = pd.read_csv(song_file,header=None)

    # Drops all nan indicies and resets the indicies
    songs = df[0].dropna().reset_index(drop=True)
    print("Number of songs:", len(songs))

    # Build vocab and tokenize corpus
    if all(os.path.exists(f) for f in ("token2idx.json", "idx2token.json")):
        with open("token2idx.json", 'r', encoding="utf-8") as f:
            token2idx = json.load(f)
        with open("idx2token.json", 'r', encoding="utf-8") as f:
            idx2token = {int(idx): token for idx, token in json.load(f).items()}
    else:
        token2idx, idx2token = identify_tokens(songs)
    print("Tokens identified")

    # Saving dictionaries
    print("Saving dictionaries")
    with open("token2idx.json", 'w', encoding="utf-8") as f:
        json.dump(token2idx, f)
    with open("idx2token.json", 'w', encoding="utf-8") as f:
        json.dump(idx2token, f)

    # Making a songs folder to hold all the jsons
    folder = "songs_corpus"
    os.makedirs(folder, exist_ok=True)

    # Tokenizing each song into its own separate json file
    for i, song in enumerate(tqdm(songs, desc="Tokenizing songs"),start=1):
        if i == num_songs:
            break

        # Splitting each song into a list of strings
        lines = song.split("\n")
        tokenized_lines = tokenize_list(lines, token2idx)
        corpus_name = f"song{i}.json"
        file_path = os.path.join(folder, corpus_name)
        with open(file_path, 'w', encoding="utf-8") as f:
            json.dump(tokenized_lines, f)

    return token2idx, idx2token

# Run process_corpus to generate song json files
def get_cleaned_corpus(num_songs = 125000):

    # Run process_corpus to generate song json files
    corpus = grab_new_songs(num_songs)

    if all(os.path.exists(f) for f in ("token2idx.json", "idx2token.json")):
        with open("token2idx.json", 'r', encoding="utf-8") as f:
            token2idx = json.load(f)
        with open("idx2token.json", 'r', encoding="utf-8") as f:
            idx2token = {int(idx): token for idx, token in json.load(f).items()}

    return corpus, token2idx, idx2token

def grab_new_songs(num_songs = 125000, directory="songs_corpus"):
    corpus = []
    if os.path.isdir(directory):
        items = os.listdir(directory)

        # Grabbing a set number of songs from the corpus
        for _ in range(0, num_songs):
            song_idx = random.randint(1, len(items) - 1)
            corpus_name = f"song{song_idx}.json"
            file_path = os.path.join(directory, corpus_name)
            with open(file_path, 'r', encoding="utf-8") as f:
                curr_song = json.load(f)
                corpus.extend(curr_song)
    return corpus

def get_dictionaries():

    # Load token dictionaries from disk if available, otherwise build them

    if all(os.path.exists(f) for f in ("token2idx.json", "idx2token.json")):
        with open("token2idx.json", 'r', encoding="utf-8") as f:
            token2idx = json.load(f)
        with open("idx2token.json", 'r', encoding="utf-8") as f:
            idx2token = {int(idx): token for idx, token in json.load(f).items()}
        return token2idx, idx2token

    return get_cleaned_corpus()[1:]
