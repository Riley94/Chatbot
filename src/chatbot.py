from functions import classify_local, get_response
import os
import json
import pickle
from tensorflow.keras.models import load_model

raw_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../raw_data/intents.json')
clean_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../clean_data/')
model = load_model(os.path.join(clean_data_path, 'model.keras'))
data = pickle.load(open(os.path.join(clean_data_path, 'data.pkl'), 'rb'))
words = data['words']
classes = data['tags']

with open(raw_data_path, 'r') as f:
    data = json.loads(open(raw_data_path).read())

while True:
    message = input("")
    ints = classify_local(message, model, words, classes)
    res = get_response(ints, data)
    print(res)
    if message == 'exit':
        break