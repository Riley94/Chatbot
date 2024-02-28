from functions import classify_local, get_response
import os
import json

raw_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../raw_data/intents.json')

with open(raw_data_path, 'r') as f:
    data = json.loads(open(raw_data_path).read())

while True:
    message = input("")
    ints = classify_local(message)
    res = get_response(ints, data)
    print(res)
    if message == 'exit':
        break