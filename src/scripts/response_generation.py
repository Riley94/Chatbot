import os
import torch
import spacy
import pickle

# user defined
from augment_data import get_intents_and_entities, augment_input_with_intent_and_entities
from helpers.seq2seq_helpers import evaluateAndShowAttention, evaluate
from augment_data import normalize_string

base_path = os.path.dirname(__file__)
clean_data_path = os.path.join(base_path, '../../clean_data')
intents_path = os.path.join(clean_data_path, 'intents_enriched.json')
intent_model_path = os.path.join(clean_data_path, 'models/intents_classifier.pth')
entity_model_path = os.path.join(clean_data_path, 'models/ner_model')

intent_model = torch.load(intent_model_path)
entity_model = spacy.load(entity_model_path)

encoder = torch.load(os.path.join(clean_data_path, 'models/encoder.pth'))
decoder = torch.load(os.path.join(clean_data_path, 'models/decoder.pth'))

with open(os.path.join(clean_data_path, 'processed_intents.pkl'), 'rb') as f:
    words, intents, _ = pickle.load(f)

def process_input(text):
    intent, entities = get_intents_and_entities(text, intent_model, entity_model)
    augmented_input = augment_input_with_intent_and_entities(normalize_string(text), intent, entities)
    return augmented_input

encoder.eval()
decoder.eval()

with open(os.path.join(clean_data_path, 'input_corpus.pkl'), 'rb') as f:
    input_text = pickle.load(f)

with open(os.path.join(clean_data_path, 'output_corpus.pkl'), 'rb') as f:
    output_text = pickle.load(f)

if __name__ == '__main__':

    while (True):
        user_input = input("User: ")
        if user_input == 'exit':
            break
        else:
            output, _ = evaluate(encoder, decoder, process_input(user_input), input_text, output_text, EOS_token=1)
            print("Bot: ", output)
            print("\n")
            print("--------------------------------------------------")
            print("\n")

    # evaluateAndShowAttention(clean_data_path, process_input('Hello. How are you?'), encoder, decoder, input_text, output_text, "attention1")

    # evaluateAndShowAttention(clean_data_path, process_input('Bye. Have a good day.'), encoder, decoder, input_text, output_text, "attention2")

    # evaluateAndShowAttention(clean_data_path, process_input('What can you do for me?'), encoder, decoder, input_text, output_text, "attention3")

    # evaluateAndShowAttention(clean_data_path, process_input('Find me a hospital nearby.'), encoder, decoder, input_text, output_text, "attention4")