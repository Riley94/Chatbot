import json
import os

def load_data():
    data_path = os.path.join(os.path.dirname(__file__), '../raw_data/intents.json')
    # Load intents
    with open(data_path, 'r') as file:
        intents = json.load(file)['intents']

    intents_dict = {}
    for intent in intents:
        intents_dict[intent['tag']] = intent['patterns']

    return intents_dict
