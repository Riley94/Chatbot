import json
import os
import torch
import torch.nn as nn
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

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

# need to define here to load the model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        hidden = self.ReLU(hidden)
        hidden = self.dropout(hidden)
        output = self.h2o(hidden)
        output = self.ReLU(output)
        output = self.dropout(output)
        output = self.softmax(output + 1e-9)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    
def predict(input_line, model, n_predictions=3):
    print('\n> %s' % input_line)
    words, intents, _ = process_intents(load_data('../clean_data/intents_enriched.json')[0])
    lemmatizer = WordNetLemmatizer()

    with torch.no_grad():
        output = evaluate(torch.from_numpy(bag_of_words(input_line, words, lemmatizer)).view(1, -1), model)

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, intents[category_index]))
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