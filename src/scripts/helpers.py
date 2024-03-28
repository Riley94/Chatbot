import json
import os
import torch
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from scripts.models.intents_classifier import RNN

def load_data(path):
    data_path = os.path.join(os.path.dirname(__file__), path)
    # Load intents
    with open(data_path, 'r') as file:
        intents = json.load(file)['intents']

    intents_dict = {}
    response_dict = {}
    for intent in intents:
        intents_dict[intent['tag']] = intent['patterns']
        response_dict[intent['tag']] = intent['responses']

    return intents_dict, response_dict

def process_intents(intents_dict):
    lemmatizer = WordNetLemmatizer()

    words = []
    intents = []
    words_tokenized = []
    ignore = ['?', '!', '.', ',']

    for intent in intents_dict:
        for pattern in intents_dict[intent]:
            w = nltk.tokenize.word_tokenize(pattern)
            words.extend(w)
            words_tokenized.append((w, intent)) # list of tuples containing list of words and tag
            if intent not in intents:
                intents.append(intent) # unique list of tags

    # find base form of word and remove ignore words
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore]
    words = sorted(list(set(words))) # remove duplicates and sort

    intents = sorted(list(set(intents)))

    return words, intents, words_tokenized
    
def predict(input_line, model, n_predictions=3):
    words, intents, _ = process_intents(load_data('../../clean_data/intents_enriched.json')[0])
    lemmatizer = WordNetLemmatizer()

    with torch.no_grad():
        output = evaluate(torch.from_numpy(bag_of_words(input_line, words, lemmatizer)).view(1, -1), model)

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            predictions.append([value, intents[category_index]])

    return predictions
    
# Just return an output given a line
def evaluate(line_tensor, model):
    hidden = model.initHidden()
    output, hidden = model(line_tensor, hidden)

    return output
    
def bag_of_words(sentence, words, lemmatizer):
    if sentence is None:
        return np.zeros(len(words))
    sentence_words = nltk.tokenize.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

class Sentence:
    def __init__(self, content):
        self.content = content
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1