import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import os
import pickle

# user defined
from helpers.intents_class_helpers import timeSince

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def indexesFromSentence(obj, sentence):
    return [obj.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(obj, sentence, EOS_token):
    indexes = indexesFromSentence(obj, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def find_max_len(pairs):
    max_length = 0
    for pair in pairs:
        if len(pair[0].split(' ')) > max_length:
            max_length = len(pair[0].split(' '))
        if len(pair[1].split(' ')) > max_length:
            max_length = len(pair[1].split(' '))

    max_length += 1

    return max_length

def get_dataloader(batch_size, clean_data_path, EOS_token):

    with open(os.path.join(clean_data_path, 'input_corpus.pkl'), 'rb') as f:
        inputs = pickle.load(f)

    with open(os.path.join(clean_data_path, 'output_corpus.pkl'), 'rb') as f:
        outputs = pickle.load(f)

    with open(os.path.join(clean_data_path, 'pairs.pkl'), 'rb') as f:
        pairs = pickle.load(f)

    max_length = find_max_len(pairs)

    n = len(pairs)
    input_ids = np.zeros((n, max_length), dtype=np.int32)
    target_ids = np.zeros((n, max_length), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(inputs, inp)
        tgt_ids = indexesFromSentence(outputs, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return inputs, outputs, train_dataloader

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train(clean_data_path, train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    torch.save(encoder, os.path.join(clean_data_path, 'models/encoder.pth'))
    torch.save(decoder, os.path.join(clean_data_path, 'models/decoder.pth'))

    return plot_losses

def save_losses(points, clean_data_path):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(os.path.join(clean_data_path, 'figures/seq2seq_losses.png'))

def evaluate(encoder, decoder, sentence, inputs, outputs, EOS_token):
    with torch.no_grad():
        input_tensor = tensorFromSentence(inputs, sentence, EOS_token)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                break
            decoded_words.append(outputs.index2word[idx.item()])
    return decoded_words, decoder_attn

def showAttention(input_sentence, clean_data_path, output_words, attentions, figure_name="attention"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(input_sentence.split(' ')) + 2))
    ax.set_yticks(np.arange(len(output_words) + 1))

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(os.path.join(clean_data_path, f'figures/{figure_name}.png'), bbox_inches='tight')

def evaluateAndShowAttention(clean_data_path, input_sentence, encoder, decoder, input_text, output_text, figure_name="attention"):
    output_words, attentions = evaluate(encoder, decoder, input_sentence, input_text, output_text, EOS_token=1)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, clean_data_path, output_words, attentions[0, :len(output_words), :], figure_name=figure_name)