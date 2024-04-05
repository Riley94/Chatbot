# Import Libraries
from nltk.stem import WordNetLemmatizer
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import ticker
import time
import numpy as np
import os
import pickle

# user defined
from helpers.intents_class_helpers import load_data, process_intents, categoryFromOutput, randomTrainingExample, train, evaluate, timeSince
from models.intents_classifier import RNN

base_path = os.path.dirname(__file__)
clean_data_path = os.path.join(base_path, '../../clean_data')
data_path = os.path.join(clean_data_path, 'intents_enriched.json')
model_path = os.path.join(clean_data_path, 'models/intents_classifier.pth')
confusion_path = os.path.join(clean_data_path, 'figures/intents_classifier_confusion.png')
loss_path = os.path.join(clean_data_path, 'figures/intents_classifier_loss.png')

intents_dict, response_dict = load_data(data_path)
n_categories = len(intents_dict)

# Lemmatize, tokenize and extract intent tags. See scripts/helpers.py for more details
words, intents, words_tokenized = process_intents(intents_dict)

# save for later use
with open(os.path.join(clean_data_path, 'processed_intents.pkl'), 'wb') as f:
    pickle.dump((words, intents, words_tokenized), f)

lemmatizer = WordNetLemmatizer()
train_x = []
train_y = []

for pair in words_tokenized:
    words_encoded = []
    pattern_words = pair[0] # list of words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        words_encoded.append(1 if w in pattern_words else 0)

    intents_encoded = [0] * len(intents)
    intents_encoded[intents.index(pair[1])] = 1 # pair[1] is the intent

    train_x.append(words_encoded)
    train_y.append(intents_encoded)

train_x = np.array(train_x)
train_y = np.array(train_y)

# Representation of the bag of words
x_tensor = torch.from_numpy(train_x).float()
y_tensor = torch.from_numpy(train_y).float()
x_tensor.shape, y_tensor.shape

# n_hidden = 128
n_hidden = 256
rnn = RNN(len(words), n_hidden, len(intents))

criterion = nn.NLLLoss()
learning_rate = 0.001

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample(intents, intents_dict, words)
    print('category =', category, '/ line =', line)

# Train the model
n_iters = 100000
print_every = 5000
plot_every = 1000

# Keep track of losses for plotting
current_loss = 0
all_losses = []

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample(intents, intents_dict, words)
    output, loss = train(category_tensor, line_tensor, rnn, criterion, learning_rate=learning_rate)
    current_loss += loss

    # Print iteration number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(intents, output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

plt.figure()
plt.plot(all_losses)

# save figure
plt.savefig(loss_path)

# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample(intents, intents_dict, words)
    output = evaluate(line_tensor, rnn)
    guess, guess_i = categoryFromOutput(intents, output)
    category_i = intents.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + intents, rotation=90)
ax.set_yticklabels([''] + intents)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.savefig(confusion_path)

# Save the model
torch.save(rnn, model_path)


