# IMPORTING LIBRARIES
import pickle
import numpy as np
import pandas as pd
import re
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.corpus import opinion_lexicon, stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# LOADING TRAINED MODEL
with open('sentiment_model.pkl', 'rb') as file:
    sentiment_model = pickle.load(file)
    print("Sentiment model loaded successfully!")

# LOADING TOKENIZER
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)
    print("Tokenizer loaded successfully!")


# LIST OF IMPORTANT WORDS THAT WERE NOT REMOVED 
important_words = {
    'not', 'no', 'never', 'none', 'neither', 'nor', 'hardly', 'barely', 'scarcely',
    'but', 'however', 'although', 'though', 'yet', 'still', 'whereas', 'nonetheless', 'nevertheless', 'on the other hand', 'conversely', 'in contrast',
    'despite', 'in spite of', 'even though', 'even if', 'notwithstanding',
    'if', 'unless', 'provided', 'except', 'otherwise',
    'only', 'just', 'merely', 'almost', 'nearly', 'barely', 'solely'
}

# FUNCTION FOR PREPROCESSING NEW REVIEW
def preprocess_review(review):
    stop_words = set(stopwords.words('english')) - important_words
    review = unicodedata.normalize('NFKD', review).lower()
    review = re.sub(r'[^a-zA-Z]', ' ', review)
    review = re.sub(r'\s+[a-zA-Z]\s+', ' ', review)
    review = re.sub(r'\s+', ' ', review)
    words = review.split()
    filtered_words = [word for word in words if word not in stop_words]
    review = ' '.join(filtered_words)
    return review

# FUNCTION FOR PREDICTING SENTIMENT OF THE PROCESSED NEW REVIEW
def predict_sentiment(review):
    processed_review = preprocess_review(review)
    
    # TOKENIZING AND PADDING
    sequence = tokenizer.texts_to_sequences([processed_review])
    padded_sequence = pad_sequences(sequence, maxlen=210, padding='post')
    
    # EXTRACTING ADDITIONAL FEATURES
    review_length = len(processed_review)
    positive_word_count = len([word for word in processed_review.split() if word in opinion_lexicon.positive()])
    negative_word_count = len([word for word in processed_review.split() if word in opinion_lexicon.negative()])
    
    # COMBINING FEATURES
    additional_features = np.array([[positive_word_count, negative_word_count, review_length]])
    combined_input = np.hstack((padded_sequence, additional_features))
    
    # PREDICTING SENTIMENT
    sentiment = sentiment_model.predict(combined_input)
    sentiment_label = ['Negative', 'Neutral', 'Positive']
    return sentiment_label[sentiment[0]]