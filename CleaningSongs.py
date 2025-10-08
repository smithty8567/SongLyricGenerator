import csv
import re
import pandas as pd

from tqdm import tqdm

def string_cleaner(lyric):

    # Lowercases all the lyrics
    lyric = lyric.lower()

    # Gets rid of all names in brackets
    lyric = re.sub(r'\[.*]\n', '', lyric)

    # Gets rid of adlibs in parantheses
    lyric = re.sub(r'\(.*\)', '', lyric)

    # Removes any repeated lines back to back
    lyric = re.sub(r'\b(\w+)(\n+\1)+', r'\1', lyric)
    lyric = re.sub(r'\n+', r'\n', lyric)

    lyric = re.sub(r'igga', r'****', lyric)

    return lyric

def clean_songs(data = "english_songs.csv"):
    print("Loading csv")
    df = pd.read_csv(data,encoding='utf-8')
    print("csv loaded")
    songs = df['lyrics']
    cleaned_songs = []
    for song_lyrics in tqdm(songs, desc="Reading and cleaning files"):
        cleaned_lyrics = string_cleaner(song_lyrics)
        if cleaned_lyrics:
            cleaned_lyrics += '\n%'
        cleaned_songs.append(cleaned_lyrics)
    with open('cleaned_songs.csv', 'w',encoding='utf-8', newline='') as csv_file:
        wr = csv.writer(csv_file)
        for song in tqdm(cleaned_songs, desc="Writing csv"):
            wr.writerow([song])