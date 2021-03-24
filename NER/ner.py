# for manipulating dataframes
import pandas as pd
import glob
# for natural language processing: named entity recognition
import spacy
from spacy import displacy
import scispacy
from spacy.tokens import Token
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
nlp_med = spacy.load("en_ner_bc5cdr_md")
nlp.max_length = 1500000
import gensim
from nltk.text import Text
from nltk import word_tokenize
import numpy as np
# for visualizations
import matplotlib.pyplot as plt
import ntpath

# NORP
def get_norp_entities(tokens):
    norp_list = []
    for ent in tokens.ents:
        if ent.label_ == 'NORP':
            norp_list.append(ent.text)
    return norp_list

# EVENT
def get_event_entities(tokens):
    event_list = []
    for ent in tokens.ents:
        if ent.label_ == 'EVENT':
            event_list.append(ent.text)
    return event_list

# DATE
def get_date_entities(tokens):
    date_list = []
    for ent in tokens.ents:
        if ent.label_ == 'DATE':
            date_list.append(ent.text)
    return date_list

# TIME
def get_time_entities(tokens):
    time_list = []
    for ent in tokens.ents:
        if ent.label_ == 'TIME':
            time_list.append(ent.text)
    return time_list

# ORG
def get_org_entities(tokens):
    org_list = []
    for ent in tokens.ents:
        if ent.label_ == 'ORG':
            org_list.append(ent.text)
    return org_list

# MONEY
def get_money_entities(tokens):
    money_list = []
    for ent in tokens.ents:
        if ent.label_ == 'MONEY':
            money_list.append(ent.text)
    return money_list

# GPE
def get_gpe_entities(tokens):
    gpe_list = []
    for ent in tokens.ents:
        if ent.label_ == 'GPE':
            gpe_list.append(ent.text)
    return gpe_list

# PAIN
pain_dict = {'pain', 'hurts', 'hurt', 'sick', 'injured', 'injure', 'blood', 'discomfort',
            'uncomfortable', 'suffer', 'suffered', 'agony', 'ache', 'ached', 'painkiller', 'painkillers'}
def get_pain_entities(tokens):
    pain_list = []
    for token in tokens:
        if token.text in pain_dict:
            pain_list.append(token.text)
    return pain_list

# MEDICATION
med_dict = {'paracetamol', 'morphine', 'ibuprofen', 'fentanyl', 'hydrochone', 'vicodin', 'tylenol', 'meds',
            'medication', 'nurofen', 'advil', 'drug', 'drugs', 'codeine', 'cocodamol', 'tramadol',
            'oxy', 'fentanyl', 'hydrocodone', 'oxycodone', 'tapentadol', 'aspirin', 'naloxone',
            'buprenorphine', 'methadone', 'co-codamol'}
def get_med_entities(tokens):
    med_list = []
    for token in tokens:
        if token.text in med_dict:
            med_list.append(token.text)
    return med_list

# EMOTION
emotion_dict = {'stressed', 'anxious', 'scared', 'afraid', 'angry', 'mad', 'upset', 'sad',
            'disappointed', 'depressed', 'happy', 'excited', 'grateful', 'relieved', 'furious',
            'joy', 'happiness', 'sadness', 'depression', 'anxiety', 'anger', 'relief', 'grief',
            'cry', 'laugh', 'stress'}
def get_emotion_entities(tokens):
    emotion_list = []
    for token in tokens:
        if token.text in emotion_dict:
            emotion_list.append(token.text)
    return emotion_list

# FAMILY
family_dict = {'mom', 'mum', 'mother', 'mommy', 'mummy', 'dad', 'father', 'daddy',
            'grandfather', 'gran', 'family', 'grandpa', 'granpa', 'granny', 'grandma',
            'husband', 'wife', 'sister', 'sis', 'brother', 'bro', 'aunt', 'uncle',
            'spouse', 'fiance', 'cousin', 'son', 'daughter', 'child', 'kid',
            'nephew', 'niece', 'grandson', 'granddaughter', 'brother-in-law',
            'sister-in-law', 'in-laws', 'inlaws', 'grandkids', 'children',
            'kids', 'grandchild', 'grandparents', 'grandmother', 'grandparent'}
def get_family_entities(tokens):
    family_list = []
    for token in tokens:
        if token.text in family_dict:
            family_list.append(token.text)
    return family_list

# JOBS
job_dict = {'worker', 'essential', 'frontline', 'teacher', 'nurse',
            'doctor', 'assistant', 'surgeon', 'plumber', 'engineer',
            'cabin', 'crew', 'pilot', 'stewardess', 'professor', 'work',
            'job', 'lawyer', 'postman', 'driver', 'staff', 'staffer',
            'firefighter', 'employee', 'receptionist', 'cleaner', 'barman',
            'barmaid', 'waiter', 'waitress', 'working', 'nurses', 'doctors',
            'teachers', 'assistants', 'surgeons', 'pilots', 'firefighters',
            'lawyers', 'postmen', 'drivers', 'employees', 'receptionists',
            'cleaners'}
def get_job_entities(tokens):
    job_list = []
    for token in tokens:
        if token.text in job_dict:
            job_list.append(token.text)
    return job_list

# VACCINE
vac_dict = {'vaccine', 'vaccinated', 'vaccines', 'moderna',
            'pfizer', 'shot', 'shots', 'jab', 'jabs'}
def get_vac_entities(tokens):
    vac_list = []
    for token in tokens:
        if token.text in vac_dict:
            vac_list.append(token.text)
    return vac_list

# TESTING
test_dict = {'test', 'tested', 'testing'}
def get_test_entities(tokens):
    test_list = []
    for token in tokens:
        if token.text in test_dict:
            test_list.append(token.text)
    return test_list

# DISEASE
def get_disease_entities(tokens):
    dis_list = []
    for ent in tokens.ents:
        if ent.label_ == 'DISEASE':
            dis_list.append(ent.text)
    return dis_list

# --- Get data
def get_data(data_path, path_type):
    if path_type == 'dir':
        all_files = glob.glob(data_path + "/*.csv")
        li = []
        for filename in all_files:
            frame = pd.read_csv(filename, index_col=None, header=0)
            li.append(frame)
        df = pd.concat(li, axis=0, ignore_index=True)
    else:
        df = pd.read_csv(data_path, index_col=None, header=0)
    return df

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def create_cols(df):
    df['Entity Types'] = np.empty((len(df), 0)).tolist()
    df['PAIN'] = np.empty((len(df), 0)).tolist()
    df['NORP'] = np.empty((len(df), 0)).tolist()
    df['MED'] = np.empty((len(df), 0)).tolist()
    df['DATE'] = np.empty((len(df), 0)).tolist()
    df['TIME'] = np.empty((len(df), 0)).tolist()
    df['EMOTION'] = np.empty((len(df), 0)).tolist()
    df['DISEASE'] = np.empty((len(df), 0)).tolist()
    df['FAMILY'] = np.empty((len(df), 0)).tolist()
    df['JOB'] = np.empty((len(df), 0)).tolist()
    df['MONEY'] = np.empty((len(df), 0)).tolist()
    df['TESTING'] = np.empty((len(df), 0)).tolist()
    df['GPE'] = np.empty((len(df), 0)).tolist()
    df['ORG'] = np.empty((len(df), 0)).tolist()
    df['VACCINE'] = np.empty((len(df), 0)).tolist()
    return df

def add_entry(df, tweet, entities, type, count):
    if entities != []:
        count = count + 1
        # Remove duplicates
        entities = list(set(entities))
        index = df.index[df['Tweet'] == tweet].tolist()
        df.at[index[0], 'Entity Types'].append(type)
        for item in entities:
            df.at[index[0], type].append(item)
    return df, count

def ner(dataset):
    print(dataset)
    print()
    df = pd.read_csv('/local/home/e69600mm/COMP30030/' + dataset + '_sentiment.csv', index_col=None, header=0)
    df = create_cols(df)
    count = 0
    dis_count = 0
    pain_count = 0
    med_count = 0
    norp_count = 0
    time_count = 0
    date_count = 0
    med_count = 0
    emotion_count = 0
    family_count = 0
    job_count = 0
    org_count = 0
    money_count = 0
    test_count = 0
    gpe_count = 0
    vac_count = 0

    for tweet in df.Tweet:
        count = count + 1

        words = word_tokenize(tweet)
        tokens = nlp(''.join(str(words)))
        tokens_med = nlp_med(tweet)

        entities = get_disease_entities(tokens_med)
        df, dis_count = add_entry(df, tweet, entities, 'DISEASE', dis_count)

        entities = get_pain_entities(tokens)
        df, pain_count = add_entry(df, tweet, entities, 'PAIN', pain_count)

        entities = get_gpe_entities(tokens)
        df, gpe_count = add_entry(df, tweet, entities, 'GPE', gpe_count)

        entities = get_org_entities(tokens)
        df, org_count = add_entry(df, tweet, entities, 'ORG', org_count)

        entities = get_norp_entities(tokens)
        df, norp_count = add_entry(df, tweet, entities, 'NORP', norp_count)

        entities = get_date_entities(tokens)
        df, date_count = add_entry(df, tweet, entities, 'DATE', date_count)

        entities = get_time_entities(tokens)
        df, time_count = add_entry(df, tweet, entities, 'TIME', time_count)

        entities = get_emotion_entities(tokens)
        df, emotion_count = add_entry(df, tweet, entities, 'EMOTION', emotion_count)

        entities = get_med_entities(tokens)
        df, med_count = add_entry(df, tweet, entities, 'MED', med_count)

        entities = get_family_entities(tokens)
        df, family_count = add_entry(df, tweet, entities, 'FAMILY', family_count)

        entities = get_money_entities(tokens)
        df, money_count = add_entry(df, tweet, entities, 'MONEY', money_count)

        entities = get_test_entities(tokens)
        df, test_count = add_entry(df, tweet, entities, 'TESTING', test_count)

        entities = get_job_entities(tokens)
        df, job_count = add_entry(df, tweet, entities, 'JOB', job_count)

        entities = get_vac_entities(tokens)
        df, vac_count = add_entry(df, tweet, entities, 'VACCINE', vac_count)

    print('PAIN: ' + str(pain_count/count*100))
    print('DISEASE: ' + str(dis_count/count*100))
    print('MED: ' + str(med_count/count*100))
    print('NORP: ' + str(norp_count/count*100))
    print('ORG: ' + str(org_count/count*100))
    print('DATE: ' + str(date_count/count*100))
    print('TIME: ' + str(time_count/count*100))
    print('EMOTION: ' + str(emotion_count/count*100))
    print('FAMILY: ' + str(family_count/count*100))
    print('JOB: ' + str(job_count/count*100))
    print('TESTING: ' + str(test_count/count*100))
    print('GPE: ' + str(gpe_count/count*100))
    print('MONEY: ' + str(money_count/count*100))
    print('VACCINE: ' + str(vac_count/count*100))
    print()
    df.to_csv('/local/home/e69600mm/COMP30030/' + dataset + '_sentiment.csv', columns=['Location', 'Tweet', 'label', 'polarity', 'subjectivity', 'Entity Types', 'PAIN', 'NORP', 'MED', 'ORG', 'DATE', 'TIME', 'EMOTION', 'DISEASE'])


ner('covid')
ner('opioids')
ner('opioids_and_covid')
