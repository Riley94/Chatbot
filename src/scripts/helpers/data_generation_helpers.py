import random
import re
import itertools
import pandas as pd
import fitz
import os

def clean_text():
    # thesaurus for ease of access to synonyms
    # load the data
    base_path = os.path.dirname(__file__)
    pdf_path = os.path.join(base_path, '../../../raw_data/thesaurus.pdf')
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")

    # Dont need intro text
    trimmed_text = text[text.find("aback"):]

    # Regex pattern to match the word, the middle part including square brackets, and the definition
    start_pattern = r"(\w+)\s\[\w+\d?\]\s(.+)"
    end_pattern = r";? ?concepts? \d{1,3}"
    lines = trimmed_text.split("\n")
    start_text = ""
    text_cleaned = []

    for i in range(len(lines)):
        start_match = re.match(start_pattern, lines[i])
        if start_match:
            word, start_text = start_match.groups()

            for j in range(i+1, len(lines)):
                end_match = re.search(end_pattern, lines[j])
                if end_match:
                    start_text += re.split(end_pattern, lines[j])[0]
                    i = j
                    break
                else:
                    start_text += lines[j]

            text_cleaned.append((word, start_text))

    return text_cleaned

# look for word in the text
def find_syns(word_to_find, context, text_cleaned):
    syns = []
    for word, definition in text_cleaned:
        if word == word_to_find and context in definition:
            possible_responses = definition.split(",")
            for response in possible_responses:
                syns.append(response.strip())

    return syns

# for correct grammar
def a_or_an(word):
    if word in ["excellent", "amazing", "outstanding", "exceptional"]:
        return "an " + word
    else:
        return "a " + word

def make_greetings(intents_dict, response_dict):
    # Get rid of synonyms that don't make sense for our purposes. Copied from above and manually edited. Prepare lists for forming greeting phrases and responses.
    greetings = ["Hello", "Hi", "Hey", "Howdy", "Greetings", "Good morning", "Good afternoon", "Good evening", "Hi there", "Hello there"]
    additional = ["how are you?", "how are you doing?", "how's it going?", "what's up?", "how do you do?", ""]
    combinations = list(itertools.product(greetings, additional))
    combinations_joined = [', '.join(combination) if combination[1] != "" else combination[0] for combination in combinations]

    responses = ["Well hello", "Hi there", "Hey yourself", "Howdy there", "Greetings", "Good morning", "Good afternoon", "Good evening", "Hi", "Hello"]
    help_inquire = "How may I help you today?" # for conciseness
    responses_additional = ["I'm doing well, thank you.", "I'm doing well, thanks for asking.", "It's going well, thank you.", "Not much, thanks for asking.",
                            "I'm doing well, thank you.", ". "]
    responses_combined = list(itertools.product(responses, responses_additional))
    responses_joined = ['. '.join(combination) if combination[1] != ". " else ''.join(combination) for combination in responses_combined]
    # Save updated intents
    intents_dict['greeting'] = combinations_joined
    response_dict['greeting'] = responses_joined

def make_goodbyes(intents_dict, response_dict):
    text_cleaned = clean_text()
    goodbye_syns = find_syns("goodbye", "farewell", text_cleaned)
    goodbye_syns.remove("leave-tak-ing")
    goodbye_syns.remove("godspeed*") # debated on keeping this
    goodbye_syns.remove("swan song*")
    goodbye_syns.remove("toodle-oo*;")
    goodbye_syns.remove("parting")

    goodbye_syns[0] = goodbye_syns[0].split(" ")[0] # remove the extra words from the first one
    # remove '*' from the end of the words
    goodbye_syns = [word.strip("*") for word in goodbye_syns]

    good_syns = ["good", "great", "excellent", "amazing", "fantastic", "superb", "terrific", "outstanding", "exceptional", "marvelous", "splendid", "wonderful"]

    goodbye_templates = []
    response_templates = []
    goodbye_syns.append("goodbye")
    goodbye_syns.append("till next time")
    goodbye_syns.append("see you later")
    goodbye_syns.append("bye")

    times = ["one", "day", "evening", "night", "rest of your day", "rest of your night", "rest of your evening", "rest of your morning"]

    goodbye_additional = ["Thanks for your help.", "Thanks again.", "Thanks for helping.", "Take care."]
    for syn in good_syns:
        for time in times:
            goodbye_additional.append(f"Have {a_or_an(syn)} {time}.")
        goodbye_additional.append(f"You've been {a_or_an(syn)} help.")

    goodbye_additional.append("")

    i = 0
    for gb_syn in goodbye_syns:
        for gb_add in goodbye_additional:
            goodbye_templates.append(f"{gb_syn}{random.choice(['!','.'])} {gb_add}")
            if i < 3: # first three are thank yous
                response_templates.append("It's my pleasure. I'm glad I could be of assistance. Goodbye!")
                i += 1
            else:
                if gb_add == "":
                    response_templates.append("Goodbye! Have a great day!") # just a goodbye
                else:
                    response_templates.append("Goodbye! Please come again!")

    intents_dict['goodbye'] = goodbye_templates
    response_dict['goodbye'] = response_templates


def make_thanks(intents_dict, response_dict):
    thanks = ['thank you very much', 'thanks a lot', 'thank you so much', 'thanks so much', 'thank you kindly', 'thank you for that.']
    thanks = thanks + intents_dict['thanks']

    second_part_nouns = ["lifesaver", "star", "gem", "legend", "hero", "saint", "miracle worker", "godsend", "savior", "treasure"]
    second_part = [" You've been a great help.", " You're a great help", " I don't know what I'd do without you."]

    for noun in second_part_nouns:
        second_part.append(f" You're a {noun}.")
        second_part.append("") # for variation

    # capitalize the first letter of each element
    thanks = [word.capitalize() for word in thanks]
    combinations = list(itertools.product(thanks, second_part))
    combinations_joined = ['. '.join(combination) if combination[1] != "" else combination[0] for combination in combinations]
    res = []
    for combination in combinations_joined:
        res.append("You're very welcome. Is there anything else I can assist you with today?")

    intents_dict['thanks'] = combinations_joined
    response_dict['thanks'] = res

def make_options(intents_dict, response_dict):
    options = []
    responses = []
    end_part = ["?", " here?", " today?", " right now?", " now?", " this evening?", " this morning?", " this afternoon?"]
    help = ["help", "assist", "support", "aid"]
    choices = ["options", "choices"]
    response_options = ['I can guide you through the adverse drug reaction list, blood pressure tracking, or find hospitals and pharmacies.',
                        'I can offer you support by giving you information on the adverse drug reaction list, blood pressure tracking, or finding hospitals and pharmacies.',
                        'I can help you with the adverse drug reaction list, blood pressure tracking, or finding hospitals and pharmacies.']

    for part in end_part:
        options.append(f"What can I do{part}")
        options.append(f"What can you do for me{part}")
        for choice in choices:
            options.append(f"What are my {choice}{part}")
        
        for help_choice in help:
            options.append(f"What can you {help_choice} me with{part}")
            options.append(f"How can you {help_choice} me{part}")
            options.append(f"What can you do to {help_choice} me{part}")

    for option in options:
        responses.append(random.choice(response_options))

    # Save the updated options
    intents_dict['options'] = options
    response_dict['options'] = responses

def make_adverse_drug_questions(intents_dict, response_dict, number_of_questions=10):
    base_path = os.path.dirname(__file__)
    medications_path = os.path.join(base_path, '../../../clean_data/medications.csv')
    inquiries = []
    # data collected using code in named_entity_recognition.ipynb
    medications = pd.read_csv(medications_path)
    choices = ['bad', 'adverse']
    interactions = ['interactions', 'effects', 'side effects', 'behavior']
    display = ['display', 'show', 'list', 'provide', 'give']
    med_syns = ['medications', 'drugs', 'pharmaceuticals', 'prescriptions', 'meds', 'medicines']
    time = ['at the same time?', 'simultaneously?', 'together?', 'with each other?', 'concurrently?']

    for i in range(number_of_questions):
        inquiries.append(f"Give me a list of {random.choice(med_syns)} causing {random.choice(choices)} {random.choice(interactions)}")
        inquiries.append(f"Which {random.choice(med_syns)} have {random.choice(choices)} {random.choice(interactions)}?")
        inquiries.append(f"{random.choice(display).capitalize()} all {random.choice(med_syns)} with {random.choice(choices)} {random.choice(interactions)}")
        inquiries.append(f"Is it safe to take {random.choice(medications['medication'].values)} and {random.choice(medications['medication'].values)} {random.choice(time)}")
        inquiries.append(f"Do {random.choice(medications['medication'].values)} and {random.choice(medications['medication'].values)} have {random.choice(choices)} {random.choice(interactions)}?")
    
    intents_dict['adverse_drug'] = inquiries
    response_dict['adverse_drug'] = ["Navigating to Adverse drug reaction module" for inquiry in inquiries]

def make_blood_pressure_questions(intents_dict, response_dict):    
    inquiries = []
    data = ['results', 'data', 'information', 'readings', 'numbers', 'stats']
    open_module = ['Open', 'Show', 'Display', 'Start', 'Run', 'Begin', 'Launch']
    module = ['module', 'manager', 'system', 'app', 'tool', 'program', 'application', 'software']
    log = ['log', 'record', 'track', 'read', 'view', 'monitor', 'check', 'watch', 'see']
    i = 0 # for use in first loop
    for elem in open_module:
        for module_syn in module:
            inquiries.append(f"{elem} the {module_syn} for blood pressure")
            inquiries.append(f"{elem} blood pressure {module_syn}")

    for data_syn in data:
        for module_syn in module:
            inquiries.append(f"Blood pressure {data_syn} {module_syn}")
        
        for elem in open_module:
            inquiries.append(f"{elem} blood pressure {data_syn}")

    for log_syn in log:
        inquiries.append(f"{log_syn} my blood pressure")
        inquiries.append(f"{log_syn} blood pressure")
        inquiries.append(f"I want to {log_syn} my blood pressure")

        for data_syn in data:
            inquiries.append(f"{log_syn} my blood pressure {data_syn}")
            inquiries.append(f"{log_syn} blood pressure {data_syn}")
            inquiries.append(f"I want to {log_syn} my blood pressure {data_syn}")

    responses = []

    for inquiry in inquiries:
        responses.append('Navigating to Blood Pressure module')

    intents_dict['blood_pressure'] = inquiries
    response_dict['blood_pressure'] = responses

def make_blood_pressure_searches(intents_dict, response_dict):
    searches = []
    search_words = ['search for', 'find', 'locate', 'view', 'show', 'display', 'pull up', 'load']
    result_words = ['results', 'data', 'information', 'readings', 'numbers', 'stats', 'history', 'logs', 'records']
    end_part = ['for patient', 'by ID', 'for patient by ID', 'for patient by name', 'for patient by name and ID', 'for patient by name or ID', 'for patient by ID or name']
    for result_word in result_words:
        for end in end_part:
            for search_word in search_words:
                searches.append(f"I want to {search_word} blood pressure {result_word} {end}")
                searches.append(f"{search_word.capitalize()} blood pressure {result_word} {end}")
            searches.append(f"Show me blood pressure {result_word} {end}")
            searches.append(f"I need {result_word} {end}")
            searches.append(f"I want {result_word} {end}")
            searches.append(f"Blood pressure {result_word} {end}")

    intents_dict['blood_pressure_search'] = searches
    responses = []
    for search in searches:
        responses.append('Please provide patient ID')

    response_dict['blood_pressure_search'] = responses


def make_blood_pressure_searches_by_patient_id(intents_dict, response_dict):
    def generate_random_patient_ids(num_ids=10, prefix="P", id_length=6):
        random_ids = []
        for _ in range(num_ids):
            id_number = ''.join([str(random.randint(0, 9)) for _ in range(id_length)])
            random_ids.append(prefix + id_number)
        return random_ids

    # Generate 100 random patient IDs
    random_patient_ids = generate_random_patient_ids(num_ids=100)

    searches = []
    search_words = ['search for', 'find', 'locate', 'view', 'show', 'display', 'pull up', 'load']
    result_words = ['results', 'data', 'information', 'readings', 'numbers', 'stats', 'history', 'logs', 'records']
    for result_word in result_words:
        for search_word in search_words:
            searches.append(f"I want to {search_word} blood pressure {result_word} for patient {random.choice(random_patient_ids)}")
            searches.append(f"{search_word.capitalize()} blood pressure {result_word} for patient {random.choice(random_patient_ids)}")
            searches.append(f"{random.choice(random_patient_ids)} blood pressure {result_word}")
            searches.append(random.choice(random_patient_ids))

    intents_dict['search_blood_pressure_by_patient_id'] = searches
    responses = []
    for search in searches:
        responses.append('Loading blood pressure data')

    response_dict['search_blood_pressure_by_patient_id'] = responses

def make_pharmacy_searches(intents_dict, response_dict):
    searches = []
    search_words = ['search for', 'find', 'locate', 'show', 'navigate to']
    location_words = ['location', 'address', 'home', 'position']
    pharmacy_words = ['pharmacy', 'drugstore', 'pharmacies', 'drugstores']
    for search_word in search_words:
        searches.append(f"I want to {search_word} a pharmacy")
        searches.append(f"{search_word.capitalize()} a pharmacy")
        for pharmacy_word in pharmacy_words:
            for location_word in location_words:
                searches.append(f"{search_word.capitalize()} {pharmacy_word} near my {location_word}")

            searches.append(f"{search_word.capitalize()} {pharmacy_word}")
            searches.append(f"{search_word} {pharmacy_word} near me")
            searches.append(f"{search_word.capitalize()} nearyby {pharmacy_word}")

    intents_dict['pharmacy_search'] = searches
    responses = []
    for search in searches:
        responses.append('Please provide pharmacy name')

    response_dict['pharmacy_search'] = responses

def make_hospital_searches(intents_dict, response_dict):
    searches = []
    search_words = ['search for', 'find', 'locate', 'view', 'show', 'display', 'pull up', 'load', 'lookup']
    hospital = ['hospital', 'hospitals']
    location_words = ['location', 'address', 'home', 'position']

    for search_word in search_words:
        searches.append(f"I want to {search_word} a hospital")
        searches.append(f"{search_word.capitalize()} a hospital")
        searches.append(f"{search_word.capitalize()} {random.choice(hospital)}")
        searches.append(f"{search_word.capitalize()} {random.choice(hospital)} near me")
        searches.append(f"{search_word.capitalize()} nearyby {random.choice(hospital)}")
        for location_word in location_words:
            searches.append(f"{search_word.capitalize()} {random.choice(hospital)} near my {location_word}")

    intents_dict['hospital_search'] = searches
    responses = []
    for search in searches:
        responses.append('Please provide hospital name, location, or type')

    response_dict['hospital_search'] = responses

def make_hospital_param_searches(intents_dict, response_dict):
    # list of hospital names
    names = ['Mayo Clinic', 'Cleveland Clinic', 'Johns Hopkins Hospital', 'Massachusetts General Hospital',
            'UCSF Medical Center', 'UCLA Medical Center', 'New York-Presbyterian Hospital', 'Stanford Health Care-Stanford Hospital',
            'Hospitals of the University of Pennsylvania-Penn Presbyterian', 'Cedars-Sinai Medical Center', 'Northwestern Memorial Hospital',
            'UPMC Presbyterian Shadyside', 'University of Michigan Hospitals-Michigan Medicine', 'Mount Sinai Hospital']
    # list of hospital addresses
    addresses = ['200 1st St SW, Rochester, MN 55905', '9500 Euclid Ave, Cleveland, OH 44195', '1800 Orleans St, Baltimore, MD 21287',
                '55 Fruit St, Boston, MA 02114', '505 Parnassus Ave, San Francisco, CA 94143', '757 Westwood Plaza, Los Angeles, CA 90095',
                '525 East 68th St, New York, NY 10065', '300 Pasteur Dr, Stanford, CA 94305', '51 N 39th St, Philadelphia, PA 19104',
                '8700 Beverly Blvd, Los Angeles, CA 90048', '251 E Huron St, Chicago, IL 60611', '200 Lothrop St, Pittsburgh, PA 15213',
                '1500 E Medical Center Dr, Ann Arbor, MI 48109', '1 Gustave L Levy Pl, New York, NY 10029']

    intents_dict['search_hospital_by_params'] = names + addresses

    response_dict['search_hospital_by_params'] = ['Loading hospital details' for _ in range(len(intents_dict['search_hospital_by_params']))]

def make_type_hospital_searches(intents_dict, response_dict):
    # list of hospital types
    types = ['general', 'community', 'teaching', 'specialty', 'clinic', 'psychiatric', 'rehabilitation', 'children', 'geriatric', 'maternity']
    searches = []

    for type in types:
        searches.append(f"Find me a {type} hospital")
        searches.append(f"{type} hospital")
        searches.append(f"Find me a {type} hospital near me")
        searches.append(f"{type} hospital near me")
        searches.append(f"Find me a {type} hospital near my location")
        searches.append(f"{type} hospital near my location")

    intents_dict['search_hospital_by_type'] = searches
    response_dict['search_hospital_by_type'] = ['Loading hospital details' for search in searches]

def update_intents(intents_dict, response_dict):
    make_greetings(intents_dict, response_dict)
    make_goodbyes(intents_dict, response_dict)
    make_thanks(intents_dict, response_dict)
    make_options(intents_dict, response_dict)
    make_adverse_drug_questions(intents_dict, response_dict)
    make_blood_pressure_questions(intents_dict, response_dict)
    make_blood_pressure_searches(intents_dict, response_dict)
    make_blood_pressure_searches_by_patient_id(intents_dict, response_dict)
    make_pharmacy_searches(intents_dict, response_dict)
    make_hospital_searches(intents_dict, response_dict)
    make_hospital_param_searches(intents_dict, response_dict)
    make_type_hospital_searches(intents_dict, response_dict)

    return intents_dict, response_dict