# # Import Libraries
import torch
import spacy

# user defined
from models.intents_classifier import RNN
from models.seq2seq import EncoderRNN, AttnDecoderRNN
from models.model_helpers.response_helpers import evaluateRandomly, evaluateAndShowAttention, process_input, get_dataloader, train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = 128
batch_size = 32

input_text, output_text, train_dataloader = get_dataloader(batch_size)

encoder = EncoderRNN(input_text.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_text.n_words).to(device)

train(train_dataloader, encoder, decoder, 80, print_every=5, plot_every=5)

encoder.eval()
decoder.eval()

evaluateAndShowAttention(process_input('Hello. How are you?'), encoder, decoder, input_text, output_text)

evaluateAndShowAttention(process_input('Bye. Have a good day.'), encoder, decoder, input_text, output_text)

evaluateAndShowAttention(process_input('What can you do for me?'), encoder, decoder, input_text, output_text)

evaluateAndShowAttention(process_input('Find me a hospital nearby.'), encoder, decoder, input_text, output_text)