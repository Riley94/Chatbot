# # Import Libraries
import torch

# user defined
from models.intents_classifier import RNN
from models.seq2seq import EncoderRNN, AttnDecoderRNN
from models.model_helpers.seq2seq_helpers import get_dataloader, train, save_losses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 128
batch_size = 32

input_text, output_text, train_dataloader = get_dataloader(batch_size)

encoder = EncoderRNN(input_text.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_text.n_words).to(device)

losses = train(train_dataloader, encoder, decoder, 80, print_every=5, plot_every=5)
save_losses(losses)