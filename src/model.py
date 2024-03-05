import numpy as np
import os
import json
import nltk
import pickle
from functions import train_model

lemmatizer = nltk.stem.WordNetLemmatizer()

base_path = os.path.dirname(os.path.realpath(__file__))
raw_data_path = os.path.join(base_path, '../raw_data/intents.json')
clean_data_path = os.path.join(base_path, '../clean_data/')

with open(raw_data_path, 'r') as f:
    data = json.loads(open(raw_data_path).read())

words = []
tags = []
word_tag_pairs = []
ignore = ['?', '!', '.', ',', '\'s']

for intent in data['intents']:
    for pattern in intent['patterns']:
        w = nltk.tokenize.word_tokenize(pattern)
        words.extend(w)
        word_tag_pairs.append((w, intent['tag'])) # list of tuples containing list of words and tag
        if intent['tag'] not in tags:
            tags.append(intent['tag']) # unique list of tags

# find base form of word and remove ignore words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore]
words = sorted(list(set(words))) # remove duplicates and sort

tags = sorted(list(set(tags)))

train_x = []
train_y = []

for pair in word_tag_pairs:
    words_encoded = []
    pattern_words = pair[0] # list of words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        words_encoded.append(1 if w in pattern_words else 0)

    tags_encoded = [0] * len(tags)
    tags_encoded[tags.index(pair[1])] = 1 # pair[1] is the tag

    train_x.append(words_encoded)
    train_y.append(tags_encoded)

train_x = np.array(train_x)
train_y = np.array(train_y)

# build model
model = train_model(train_x, train_y)

# save model and data
model.save(os.path.join(clean_data_path, "model.keras"))
# pickle.dump(model, open(os.path.join(clean_data_path, "model.pkl"), "wb"))
pickle.dump( {'words':words, 'tags':tags, 'train_x':train_x, 'train_y':train_y}, open( os.path.join(clean_data_path, "data.pkl"), "wb"))