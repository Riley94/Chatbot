from helpers.data_generation_helpers import update_intents
from helpers.intents_class_helpers import load_data
import json
import os

def generate_data():
    base_path = os.path.dirname(__file__)
    intents_path = os.path.join(base_path, '../../raw_data/intents.json')
    intents_dict, response_dict = load_data(intents_path)
    intents_dict, response_dict = update_intents(intents_dict, response_dict)
    # Load the original JSON file
    with open(intents_path, 'r') as file:
        data = json.load(file)

    # Update the JSON structure with new patterns and responses except for the 'noanswer' intent
    for intent in data["intents"]:
        tag = intent["tag"]
        if tag in intents_dict:
            intent["patterns"] = intents_dict[tag]
        if tag in response_dict:
            intent["responses"] = response_dict[tag]

    data["intents"] = [intent for intent in data["intents"] if intent["tag"] != "noanswer"]

    # Write the updated JSON structure back to a file
    with open(os.path.join(base_path, '../../clean_data/intents_enriched.json'), 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == '__main__':
    generate_data()