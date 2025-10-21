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

To generate our corpus, we tokenized eveSury line with a greedy algorithm to grab the longest sequence of characters that is part of our dictionary. We then converted a set amount of our songs into a json file where each song is its own json file. We decided on 1 million songs as our size for this GRU.

## Example of generated lyrics
### Generating lyrics from a song in the training set
------------------------------------------------------
#### Given Lines: 

Life's a game but it's not fair

break the rules, so I don't care

------------------------------------------------------

#### Prediction with temperature = 0.15
Life's a game, but it's not fair

I break the rules, so I don't care

And I ain't gotta get down

I gotta make 'em

### Generating lyrics from an artist who's lyrics are in the training set, but the song is not in the training set
------------------------------------------------------
#### Given lines:

I could change your life so easily

I keep beggin' you to stay, but you're leavin' me

------------------------------------------------------
#### Prediction with temperature = 0.15

o could change your life so easily

i keep beggin' you to stay, but you're leavin' me

and i don't wanna be the b****

she got no time she got no feelin'

And i don't wanna know what she want


### Generating lyrics with no context
------------------------------------------------------
Note temperature is higher to generate lyrics without context, otherwise model will just output the end of song character.

#### Prediction with temperature = 0.25

this is my life

it's time that we'll be the time

and i can't help

just like you, yeah

#### Prediction with temperature = 0.3

you're gonna be like it

and that's the same

it's a wonderful day

it's time that you want to go

if i don't want to say


## Successes of our model
------------------------------------------------------

Model is able to consistantly generate real English words from both seed lyrics and from scratch. The lyrics that are generated make sense and generally fit the context of the song.

## Shortcomings of our model
------------------------------------------------------

The generated lyrics lack creativity and tend to repeat various lines and phrases. This could be improved by training on a larger portion of the dataset. The lyrics also lack many literary devices that songs have such as alliteration, rhyming, and repetition. Additionally the network tends to end a song when the temperature is too low. Finally, our dataset is skewed heavily towards rap lyrics so the model struggles to generate other genre lyrics. 
