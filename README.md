# # Chatbot
# Overview

This projects goal is to develop a chatbot using advanced Natural Language Processing (NLP) techniques, including Named Entity Recognition (NER), Intent Classification, and a Sequence-to-Sequence (Seq2Seq) model with attention for generating responses. Designed to assist in healthcare settings, the chatbot aims to understand and respond to patient queries effectively, providing an interactive experience.

# Features

    Named Entity Recognition (NER): Identifies medical entities (e.g., symptoms, medications) within user queries.
    Intent Classification: Determines the intent behind user queries, classifying them into categories such as greetings, goodbyes, etc.
    Data Generation: Enhances training data by generating sentence variations through combinations of sentence fragments or words for different intent classes.
    Response Generation: Utilizes a Seq2Seq model with attention mechanisms to generate relevant responses based on the enriched input data.

Project Structure

    src/: Contains Jupyter notebooks and Python scripts with the project's logic.
        intents_classification.ipynb
        named_entity_recognition.ipynb
        data_generation.ipynb
        response_generation.ipynb
        scripts/: Python scripts for each step in the data processing and model training pipeline.
            augment_data.py
            data_collection.py
            data_generation.py
            intents_classification.py
            named_entity_recognition.py
            response_generation.py
            train_seq2seq.py
    raw_data/: Contains scraped or provided raw data.
    clean_data/: Stores processed or extracted data, including models and figures.
        models/: Trained model files.
        figures/: Generated figures and plots.

Libraries Used

    PyTorch
    spaCy
    NLTK
    NumPy
    Matplotlib
    JSON
    Fitz
    Pandas
    itertools
    BeautifulSoup
    pickle
    Google API Client
    Google Auth OAuthlib

Getting Started

    Setup Environment: Ensure you have Python 3.x installed. It's recommended to use a virtual environment.

    bash

python -m venv chatbot-env
source chatbot-env/bin/activate  # On Windows, use `chatbot-env\Scripts\activate`

Install Dependencies:

pip install torch spacy nltk numpy matplotlib pandas beautifulsoup4 pickle google-api-python-client google-auth-oauthlib

Prepare Data: Run the data collection and preparation scripts to gather and process the raw data.

bash

python src/scripts/data_collection.py
python src/scripts/augment_data.py

Train Models: Navigate to the src/ directory and execute the Jupyter notebooks in order to train the NER, Intent Classification, and Seq2Seq models.

    jupyter notebook intents_classification.ipynb

    Launch the Chatbot: After training the models, you can interact with the chatbot through the command line or integrate it into a web application.

# ToDo
- finish response_generation notebook
  - Query and Commands
  - Fuzzy Matching (Or something like it)
  - Bot Implementation
- Improve Model (Ongoing)
  - Improve Intent Classification
  - Improve Entity Recognition
  - Improve Response Generation
  - Improve Query (time or accuracy)
  - Improve Data
    - Enrich with 3rd-Party Data
      - Web Scraping
      - Kaggle
      - Other Known Data Sources
    - Improve Randomness or Accuracy of Randomly Generated Data
