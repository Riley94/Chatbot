import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import random
import pickle
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload
import io
import re
import datetime

base_path = os.path.dirname(__file__)
clean_data_path = os.path.join(base_path, '../../clean_data')

# Medications
def get_medications():
    medication_path = os.path.join(clean_data_path, 'medications.csv')
    # if csv exists, load it
    if os.path.exists(medication_path):
        medications = pd.read_csv(medication_path)
    # otherwise, scrape the data
    else:
        url = "https://healthy.kaiserpermanente.org/health-wellness/drug-encyclopedia."

        medications = []

        # iterate from 'a' to 'z'
        for letter in range(97, 123):
            url = url + chr(letter)
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            for li in soup.select(".drug-column-4"):
                medications.append(li.text)
            
            url = "https://healthy.kaiserpermanente.org/health-wellness/drug-encyclopedia."

        # split by new line
        medications = [medication.split('\n') for medication in medications]
        # flatten and remove empty strings
        medications = [medication for sublist in medications for medication in sublist if medication != '']
        medications = pd.DataFrame(medications, columns=['medication'])
        medications.to_csv(medication_path, index=False)

    return medications

# Define the scope of the application
SCOPES = ['https://www.googleapis.com/auth/drive']

# Function to authenticate and create the service
def create_service():
    creds = None
    # The file token.pickle stores the user's access and refresh tokens.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    service = build('drive', 'v3', credentials=creds)
    return service

# Function to list files in a given folder ID
def list_files_in_folder(service, folder_id):
    results = service.files().list(q=f"'{folder_id}' in parents", fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])
    return items

# Function to download a file
def download_or_export_file(service, file_id, file_name, mime_type):
    try:
        # Check if the file is a Google Doc by its MIME type
        if mime_type.startswith('application/vnd.google-apps.'):
            # Define export MIME type for Google Docs (e.g., 'application/pdf' for Google Docs)
            if mime_type == 'application/vnd.google-apps.document':
                export_mime_type = 'application/pdf'
                file_name += '.pdf'  # Append appropriate file extension
            elif mime_type == 'application/vnd.google-apps.spreadsheet':
                export_mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                file_name += '.xlsx'  # Append appropriate file extension
            elif mime_type == 'application/vnd.google-apps.presentation':
                export_mime_type = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
                file_name += '.pptx'  # Append appropriate file extension
            else:
                # Default to PDF for other Google Apps documents
                export_mime_type = 'application/pdf'
                file_name += '.pdf'
            
            request = service.files().export_media(fileId=file_id, mimeType=export_mime_type)
        else:
            # For binary files, use the get_media method
            request = service.files().get_media(fileId=file_id)
        
        # Perform the download or export
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")
        
        # Write the file's contents to a local file
        with open(file_name, 'wb') as f:
            f.write(fh.getbuffer())
        print(f"File '{file_name}' downloaded successfully.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

def find_folders_by_name(service, folder_name):
    """Find folders by name and return their IDs."""
    query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}'"
    response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    return response.get('files', [])

def find_subfolder_id(service, parent_folder_id, subfolder_name):
    """Find a specific subfolder within a parent folder."""
    query = f"'{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and name='{subfolder_name}'"
    response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    files = response.get('files', [])
    if files:
        return files[0]['id']  # Return the ID of the first matching subfolder
    return None

def download_txt_files_from_folder(service, folder_id):
    """Download all .txt files from a specified folder."""
    query = f"'{folder_id}' in parents and mimeType='text/plain'"
    response = service.files().list(q=query, spaces='drive', fields='files(id, name, mimeType)').execute()
    files = response.get('files', [])
    for file in files:
        print(f"Downloading/exporting {file['name']}...")
        download_or_export_file(service, file['id'], file['name'], file['mimeType'])

# Toy diagnosis data. Might look into a better source, but this will work for now
def get_diagnoses():
    diagnoses_path = os.path.join(clean_data_path, 'diagnoses.csv')
    # again, if csv exists, load it
    if os.path.exists(diagnoses_path):
        diagnoses = pd.read_csv(diagnoses_path)
    # otherwise, download it from Google Drive
    else:
        service = create_service()  # Assume this is implemented as shown before
        top_level_folder_names = ['Base-Game', 'Mod-Diagnoses']

        for folder_name in top_level_folder_names:
            folders = find_folders_by_name(service, folder_name)
            for folder in folders:
                dept_diagnoses_folder_id = find_subfolder_id(service, folder['id'], 'Dept-Diagnoses')
                if dept_diagnoses_folder_id:
                    download_txt_files_from_folder(service, dept_diagnoses_folder_id)
        
        diagnoses = []
        # loop through all files and extract the text following '##' (diagnosis names)
        for file in os.listdir('../raw_data/diagnoses'):
            with open(f'../raw_data/diagnoses/{file}', 'r') as f:
                for line in f:
                    if '##' in line:
                        diagnoses.append(line.split('##')[1].strip())
        
        diagnoses = pd.DataFrame(diagnoses, columns=['diagnosis']).to_csv(diagnoses_path, index=False)

    return diagnoses

def get_dosages():
    # randomly generate dosage data
    dosages = []
    units = ['mg', 'g', 'mL', 'L']
    concat_every = 5
    frequency = ['twice daily', 'once daily', 'as needed']

    for hour in range(48):
        frequency.append(f'every {hour} hours')

    for i in range(300):
        dosage = str(random.choice(range(5, 1001, 5))) + f' {random.choice(units)}'
        if i % concat_every == 0:
            dosage += ' ' + random.choice(frequency)
        dosages.append(dosage)

    return dosages

# Tests
def get_tests():
    tests_path = os.path.join(clean_data_path, 'tests.csv')
    # if csv exists, load it
    if os.path.exists(tests_path):
        tests = pd.read_csv(tests_path)
    # otherwise, scrape the data
    else:
        url = "https://medlineplus.gov/lab-tests/"

        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        tests = []
        # select all uorderdered lists with class 'withident breaklist'
        for item in soup.select(".withident.breaklist"):
            tests.append(item.text)

        tests = pd.DataFrame(tests, columns=['test'])
        tests['test'] = tests['test'].str.split('\n')
        tests = tests.explode('test')
        tests = tests[tests['test'] != '']
        tests.to_csv(tests_path, index=False)

    return tests

# Symptoms
def get_symptoms():
    symptoms_path = os.path.join(clean_data_path, 'symptoms.csv')
    # if csv exists, load it
    if os.path.exists(symptoms_path):
        symptoms = pd.read_csv(symptoms_path)
    else:
        symptoms = []
        pattern = r"\+\s(.+?)\s\(\d+% of cases \| .+?\)" # matches the symptom text

        # loop through all files and extract the symptoms
        for file in os.listdir('../raw_data/diagnoses'):
            with open(f'../raw_data/diagnoses/{file}', 'r') as f:
                for line in f:
                    match = re.search(pattern, line)
                    if match:
                        symptoms.append(match.group(1))

        symptoms = pd.DataFrame(symptoms, columns=['symptom'])
        # make unique
        symptoms = symptoms.drop_duplicates()
        symptoms.to_csv(symptoms_path, index=False)
    
    return symptoms

# Dates
def get_dates():
    dates = []
    # generate dates in the format "Month Day"
    for i in range(1, 13):
        for j in range(1, 29):
            # year is arbitrary
            date = datetime.date(2021, i, j)
            full_date = f"{date.strftime('%B')} {date.strftime('%d')}"
            dates.append(full_date)

    return dates

# Times
def get_times():
    times = []
    # generate times in the format "Hour AM/PM"
    for i in range(1, 13):
        for j in ["AM", "PM"]:
            time = f"{i} {j}"
            times.append(time)
    
    return times

# Body parts
def get_anatomies():
    # Hard-coding for now. Will change
    anatomies = ["left femur", "right knee", "abdominal region", "left lung", "right lung", "left kidney", "right kidney",
                "left eye", "right eye", "left ear", "right ear", "left hand", "right hand", "left foot", "right foot",
                "left arm", "right arm", "left leg", "right leg", "left shoulder", "right shoulder", "left hip", "right hip",
                "left elbow", "right elbow", "left wrist", "right wrist", "left ankle", "right ankle", "left toe", "right toe",
                "left finger", "right finger", "left thumb", "right thumb", "left nostril", "right nostril", "left cheek", "right cheek",
                "left temple", "right temple", "left jaw", "right jaw", "left chin", "right chin", "left neck", "right neck", "left collarbone",
                "right collarbone", "left rib", "right rib", "left hip bone", "right hip bone", "left thigh", "right thigh", "left calf",
                "right calf", "left shin", "right shin", "left heel", "right heel", "left sole", "right sole", "left toe", "right toe",
                "left finger", "right finger", "left thumb", "right thumb", "left palm", "right palm", "left wrist", "right wrist", "left forearm",
                "right forearm", "left bicep", "right bicep", "left tricep", "right tricep", "left shoulder", "right shoulder", "left chest", "right chest",
                "left breast", "right breast", "left nipple", "right nipple", "left rib", "right rib", "left abdomen", "right abdomen", "left hip",
                "right hip", "left groin", "right groin", "left thigh", "right thigh", "left knee", "right knee", "left shin", "right shin", "left calf",
                "right calf", "left ankle", "right ankle", "left foot", "right foot", "left toe", "right toe", "left finger", "right finger", "left thumb",
                "right thumb", "left hand", "right hand", "left wrist", "right wrist", "left forearm", "right forearm", "left elbow", "right elbow",
                "left upper arm", "right upper arm", "left shoulder", "heart", "liver", "stomach", "intestines", "pancreas", "spleen", "bladder", "esophagus"]


    anatomies = list(set(anatomies))
    return anatomies

# # Data Processing
# Function to generate a random date
def generate_example(diagnoses, medications, dosages, tests, symptoms, anatomies):
    output = {}
    genders = ['he', 'she', 'they', 'the patient']
    diagnosis = random.choice(diagnoses['diagnosis'].values)
    medication = random.choice(medications['medication'].values)
    dosage = random.choice(dosages)
    test_name = random.choice(tests['test'].values)
    symptom = random.choice(symptoms['symptom'].values)
    body_part = random.choice(anatomies)
    gender = random.choice(genders)

    choices = [diagnosis, medication, dosage, test_name, symptom, body_part, gender]
    choice_map = ['diagnosis', 'medication', 'dosage', 'test_name', 'symptom', 'body_part', 'gender']
    entities = []

    text_elements = [
        f"{gender.capitalize()} was diagnosed with {diagnosis} last year.",
        f"{gender.capitalize()} has been prescribed {medication} {dosage}.",
        f"{test_name.capitalize()} measurements indicate {diagnosis}.",
        f"The {test_name} revealed a {diagnosis} in the {body_part}.",
        f"Patient presents with {symptom}.",
        f"Prescribe {dosage} of {medication} for pain relief.",
        f"The {test_name} shows normal {body_part} function.",
        f"{gender.capitalize()} mentioned an allergy to {medication}.",
        f"Examine the {symptom} in the patient's {body_part}.",
    ]
    text = random.choice(text_elements)
    annotated_segments = []
    for index, choice in enumerate(choices):
        start_pos = text.find(choice)
        if start_pos != -1 and not any(start <= start_pos < end for start, end in annotated_segments):
            end_pos = start_pos + len(choice)
            entities.append({"start": start_pos, "end": end_pos, "label": choice_map[index]})
            annotated_segments.append((start_pos, end_pos))
    
    output["text"] = text
    output["entities"] = entities
    return output