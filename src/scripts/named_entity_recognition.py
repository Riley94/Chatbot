# Import Libraries
import spacy
import os
import json
import pandas as pd
import random
from spacy.training import Example
from spacy.util import minibatch, compounding

# user defined
from data_collection import get_diagnoses, get_medications, get_dosages, get_tests, get_symptoms, get_anatomies, generate_example

base_path = os.path.dirname(__file__)
clean_data_path = os.path.join(base_path, '../../clean_data')


def collect_data():
    train_data_path = os.path.join(clean_data_path, 'train_data.json')

    # Data Collection
    diagnoses = get_diagnoses()
    medications = get_medications()
    dosages = get_dosages()
    tests = get_tests()
    symptoms = get_symptoms()
    anatomies = get_anatomies()

    # Save or load the data to JSON for easy access
    if os.path.exists(train_data_path):
        # Load JSON data
        with open(train_data_path, 'r') as file:
            train_data = json.load(file)
    else:
        # Generate 1000 examples
        train_data = [generate_example(diagnoses, medications, dosages, tests, symptoms, anatomies) for _ in range(1000)]
        with open(train_data_path, 'w') as file:
            json.dump(train_data, file, indent=2)

    # Convert the JSON data to the desired format
    formatted_data = []
    for item in train_data:
        text = item['text']
        entities = []
        for entity in item['entities']:
            start = entity['start']
            end = entity['end']
            label = entity['label']
            entities.append((start, end, label.upper()))  # Convert label to uppercase as shown in the example
        formatted_data.append((text, {"entities": entities}))

    return formatted_data


def train():
    formatted_data = collect_data()
    entity_labels = set()
    for text, annotations in formatted_data:
        for entity in annotations['entities']:
            entity_labels.add(entity[2])

    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load('en_core_web_sm') # use a pre-trained model

    # Add the NER pipeline if not already present
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe('ner')
    else:
        ner = nlp.get_pipe('ner')

    # Add entity labels to the model
    for entity in entity_labels:
        ner.add_label(entity)

    optimizer = nlp.resume_training()
    for itn in range(10):  # Number of training iterations
        random.shuffle(formatted_data)
        losses = {}
        for batch in minibatch(formatted_data, size=compounding(4.0, 32.0, 1.001)):
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
        print("Losses", losses)

    # Save the model
    nlp.to_disk(os.path.join(clean_data_path, 'models/ner_model'))

    return nlp

if __name__ == '__main__':
    train()