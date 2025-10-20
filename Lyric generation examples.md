# Gru Model Generation examples:


### Generating lyrics from a song in the training set
#### Given Lines: 

Life's a game but it's not fair

break the rules, so I don't care

------------------------------------------------------

#### Prediction with temperature = 0.15
Life's a game, but it's not fair

I break the rules, so I don't care

And I ain't gotta get down

I gotta make 'em

------------------------------------------------------
### Generating lyrics from an artist who's lyrics are in the training set, but the song is not in the training set
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

Note temperature is higher to generate lyrics without context.

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
