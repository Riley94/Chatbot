# # Import Libraries
import torch
import os
import spacy
from nltk.stem import WordNetLemmatizer

# user defined
from models.intents_classifier import RNN
from models.seq2seq import EncoderRNN, AttnDecoderRNN
from helpers.seq2seq_helpers import get_dataloader, train, save_losses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_generator():

    EOS_token = 1

    base_path = os.path.dirname(__file__)
    clean_data_path = os.path.join(base_path, '../../clean_data')

    hidden_size = 128
    batch_size = 32

    input_text, output_text, train_dataloader = get_dataloader(batch_size, clean_data_path, EOS_token)

    encoder = EncoderRNN(input_text.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_text.n_words).to(device)

    losses = train(clean_data_path, train_dataloader, encoder, decoder, 80, print_every=5, plot_every=5)
    save_losses(losses, clean_data_path)

if __name__ == '__main__':
    train_generator()