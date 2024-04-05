import numpy as np
import os
import torch
import spacy
import pickle

# user defined
from helpers.intents_class_helpers import load_data, predict
from helpers.corpus import Corpus
from helpers.seq2seq_helpers import normalize_string

INTENT_PREFIX = '[INTENT]'
ENTITY_PREFIX = '[ENTITY]'

base_path = os.path.dirname(__file__)
clean_data_path = os.path.join(base_path, '../../clean_data')
intents_path = os.path.join(clean_data_path, 'intents_enriched.json')
intent_model_path = os.path.join(clean_data_path, 'models/intents_classifier.pth')
entity_model_path = os.path.join(clean_data_path, 'models/ner_model')

def get_intents_and_entities(text, intent_model, entity_model):
    intent_prob = -np.inf
    intents_pred = predict(text, intent_model)
    intent = ''

    for item in intents_pred:
        if item[0] > intent_prob:
            intent_prob = item[0]
            intent = item[1]
    ner_doc = entity_model(text)
    entities = [(ent.text, ent.label_) for ent in ner_doc.ents]
    return intent, entities

def augment_input_with_intent_and_entities(user_input, intent, entities):
    # Augment the input with the intent
    augmented_input = f"{INTENT_PREFIX}{intent} " + user_input
    
    # Augment the input with entities
    for entity, entity_type in entities:
        augmented_input += f" {ENTITY_PREFIX}{entity_type}"
    
    return augmented_input

def makePairs():
    print("Reading lines...")
    questions, responses = load_data(intents_path)

    intent_model = torch.load(intent_model_path)
    entity_model = spacy.load(entity_model_path)

    # make pairs of input and response and normalize
    pairs = []
    for tag in responses:
        for i in range(len(responses[tag])):
            _, entities = get_intents_and_entities(questions[tag][i], intent_model, entity_model)
            augmented_input = augment_input_with_intent_and_entities(normalize_string(questions[tag][i]), tag, entities)
            pairs.append([augmented_input, responses[tag][i]])
        
    inputs = Corpus('inputs')
    outputs = Corpus('responses')

    return inputs, outputs, pairs

def prepareData():
    input, output, pairs = makePairs()
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input.addSentence(pair[0])
        output.addSentence(pair[1])
    print("Counted words:")
    print(input.name, input.n_words)
    print(output.name, output.n_words)

    return input, output, pairs

def augment_data():
    input, output, pairs = prepareData()

    # Save the objects for later use
    with open(os.path.join(clean_data_path, 'input_corpus.pkl'), 'wb') as f:
        pickle.dump(input, f)
        
    with open(os.path.join(clean_data_path, 'output_corpus.pkl'), 'wb') as f:
        pickle.dump(output, f)
        
    with open(os.path.join(clean_data_path, 'pairs.pkl'), 'wb') as f:
        pickle.dump(pairs, f)

if __name__ == '__main__':
    augment_data()