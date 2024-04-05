import torch
from nltk import WordNetLemmatizer
import json
import os
import nltk
import numpy as np
import random
import math
import time
import pickle

base_path = os.path.dirname(__file__)
clean_data_path = os.path.join(base_path, '../../../../clean_data')

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

def load_data(path):
    # Load intents
    with open(path, 'r') as file:
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

# Just return an output given a line
def evaluate(line_tensor, model):
    hidden = model.initHidden()
    output, hidden = model(line_tensor, hidden)

    return output

def predict(input_line, model, n_predictions=3):
    data_path = os.path.join(clean_data_path, 'processed_intents.pkl')
    
    with open(data_path, 'rb') as file:
        words, intents, _ = pickle.load(file)

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

# Pick train inputs randomly to avoid overfitting.
def randomChoice(l):
    if len(l) == 0:
        return None
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample(intents, intents_dict, words):
    lemmatizer = WordNetLemmatizer()
    category = randomChoice(intents)
    line = randomChoice(intents_dict[category])
    category_tensor = torch.tensor([intents.index(category)], dtype=torch.long)
    line_tensor = torch.from_numpy(bag_of_words(line, words, lemmatizer)).view(1, -1)
    return category, line, category_tensor, line_tensor

# Test output before training
def categoryFromOutput(intents, output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return intents[category_i], category_i

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def train(category_tensor, line_tensor, model, criterion, learning_rate=0.001):
    hidden = model.initHidden()

    model.zero_grad()

    output, hidden = model(line_tensor, hidden)

    loss = criterion(output, category_tensor)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()