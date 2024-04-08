import os

from intents_classification import train_classifier
from named_entity_recognition import train_ner
from train_seq2seq import train_generator
from augment_data import augment_data, prepareData

if __name__ == '__main__':
    train_classifier()
    train_ner()

    choice = input("Re-Augment the data? (Y/N)")
    if choice.lower() == 'y':
        base_path = os.path.dirname(__file__)
        clean_data_path = os.path.join(base_path, '../../clean_data')
        augment_data(clean_data_path)

    train_generator()