import numpy as np
import os
import torch
import spacy
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# user defined
from helpers.intents_class_helpers import load_data, predict
from helpers.corpus import Corpus

def normalize_string(s):
    lemmatizer = WordNetLemmatizer()
    # Tokenize the sentence. This also implicitly removes punctuation if we do not consider them as separate tokens.
    tokens = word_tokenize(s)
    # Lemmatize and lowercase each word, excluding punctuation
    lemmatized_tokens = [lemmatizer.lemmatize(token).lower() for token in tokens if token not in string.punctuation]
    # Reconstruct the sentence from lemmatized tokens
    lemmatized_sentence = ' '.join(lemmatized_tokens)

    return lemmatized_sentence

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
    INTENT_PREFIX = '[INTENT]'
    ENTITY_PREFIX = '[ENTITY]'
    # Augment the input with the intent
    augmented_input = f"{INTENT_PREFIX}{intent} " + user_input
    
    # Augment the input with entities
    for entity, entity_type in entities:
        augmented_input += f" {ENTITY_PREFIX}{entity_type}"
    
    return augmented_input

def makePairs(questions, responses, intent_model, entity_model):
    print("Reading lines...")

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

def prepareData(inputs, outputs, pairs):
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        inputs.addSentence(pair[0])
        outputs.addSentence(pair[1])
    print("Counted words:")
    print(inputs.name, inputs.n_words)
    print(outputs.name, outputs.n_words)

    return inputs, outputs, pairs

def augment_data(clean_data_path):
    intents_path = os.path.join(clean_data_path, 'intents_enriched.json')
    intent_model_path = os.path.join(clean_data_path, 'models/intents_classifier.pth')
    entity_model_path = os.path.join(clean_data_path, 'models/ner_model')
    questions, responses = load_data(intents_path)

    intent_model = torch.load(intent_model_path)
    entity_model = spacy.load(entity_model_path)

    inputs, outputs, pairs = makePairs(questions, responses, intent_model, entity_model)
    inputs, outputs, pairs = prepareData(inputs, outputs, pairs)


    # Save the objects for later use
    with open(os.path.join(clean_data_path, 'input_corpus.pkl'), 'wb') as f:
        pickle.dump(inputs, f)
        
    with open(os.path.join(clean_data_path, 'output_corpus.pkl'), 'wb') as f:
        pickle.dump(outputs, f)
        
    with open(os.path.join(clean_data_path, 'pairs.pkl'), 'wb') as f:
        pickle.dump(pairs, f)

if __name__ == '__main__':
    base_path = os.path.dirname(__file__)
    clean_data_path = os.path.join(base_path, '../../clean_data')
    
    augment_data(clean_data_path)