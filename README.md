# Song Lyric Generator using a GRU
This project is able to generate lines of text that resembled english song lyrics using a 2-layer GRU when given a line or two.

## Dataset
Our dataset was from kaggle which grabbed millions of song lyrics from Genius which is a community driven website to post song lyrics.
(https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information/data)

## Cleaning the Data
Cleaning the data involved shrinking the dataset (song_lyrics.csv) to only include english songs. All genius lyrics also included 
descriptions in the lyrics like who is singing and what kind of lyrics were being sung, [Taylor Swift, Ed Sheeran] or [Hook] [Chorus].
We used regex to clean the lyrics by getting rid of text inside [], (), and any repeated lines. This was to make sure repeated lines and adlibs, denoted in (), overwhelm our
model to just generate the same line that was given or to generate (yeah) repeatedly.

The dataset ended up containing 2.5 million songs after the cleaning (cleaned_songs.csv).

## Tokens and Corpus Generation
Our project uses a 10000 token dictionary that is filled with n-grams from length 1 to 8. N-grams are a sequence of characters of length n.
Using n-grams helps the GRU learn words without actually being given words. Our dictionary grabs the 10000 most frequent n-grams found throughout all 2.5 million songs to use at its vocabulary.

To generate our corpus, we tokenized every line with a greedy algorithm to grab the longest sequence of characters that is part of our dictionary. We then converted a set amount
of our songs into a json file where each song is its own json file. We decided on 1 million songs as our size for this GRU.

## Example of generated lyrics
Giving our GRU two lines for some context to generate lyrics, our GRU was able to fully generate the following lines on its own:

cause i could get her this world <br>
but i'm just like a star with me <br>
the sound for you <br>
but if you want to give it up <br>
it's time that you're gonna be right <br>
