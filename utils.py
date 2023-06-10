# Python script for general utility functions needed to call APIs and format datasets/results
from textblob import TextBlob

# requires the textblob library, run this is the cli to install: conda install -c conda-forge textblob

# Takes in a corpus of text and corrects spelling errors using the TextBlob library.
def correct_spelling(text):
    text = TextBlob(text)
    result = text.correct()
    return result
