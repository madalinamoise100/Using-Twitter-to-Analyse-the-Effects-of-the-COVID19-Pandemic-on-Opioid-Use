# for manipulating dataframes
import pandas as pd
import glob
# for natural language processing: named entity recognition
import spacy
from spacy.tokens import Token
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
nlp.max_length = 1500000
import gensim
from nltk.text import Text
import numpy as np
# for visualizations
import matplotlib.pyplot as plt

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

pain_dict = {'pain', 'hurts', 'hurt', 'sick', 'injured', 'injure', 'blood', 'discomfort',
            'uncomfortable', 'suffer', 'suffered', 'agony', 'ache', 'ached', 'painkiller', 'painkillers'}
def get_pain_entities(tokens):
    pain_list = []
    for token in tokens:
        if token.text in pain_dict:
            pain_list.append(token.text)
    return pain_list

med_dict = {'paracetamol', 'morphine', 'ibuprofen', 'fentanyl', 'hydrochone', 'vicodin', 'tylenol', 'meds',
            'medication', 'nurofen', 'advil'}
def get_med_entities(tokens):
    med_list = []
    for token in tokens:
        if token.text in med_dict:
            med_list.append(token.text)
    return med_list

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

df_entities = pd.DataFrame()
pain_list = []
norp_list = []
org_list = []
time_list = []
date_list = []
med_list = []
emotion_list = []

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

path = r'/home/e69600mm/COMP30030/data/covid' # use your path
all_files = glob.glob(path + "/*.csv")
li = []
for filename in all_files:
    print(filename)
    df = pd.read_csv(filename, index_col=None, header=0)
    data = df.Tweet.values.tolist()
    words = list(sent_to_words(data))
    tokens = nlp(''.join(str(words)))
    pain_list.extend(get_pain_entities(tokens))
    norp_list.extend(get_norp_entities(tokens))
    org_list.extend(get_org_entities(tokens))
    time_list.extend(get_time_entities(tokens))
    date_list.extend(get_date_entities(tokens))
    med_list.extend(get_med_entities(tokens))
    emotion_list.extend(get_emotion_entities(tokens))

pain_count = Counter(pain_list).most_common(20)
df_pain = pd.DataFrame(pain_count, columns=['Entity', 'Count'])
df_pain['Entity Type'] = 'PAIN'
df_pain['Concordances'] = np.empty((len(df_pain), 0)).tolist()
dict = {}
for ent in df_pain['Entity']:
    dict[ent] = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    data = df.Tweet.values.tolist()
    words = list(sent_to_words(data))
    for tweet in words:
        textList = Text(tweet)
        for ent in dict:
            concordance_list = textList.concordance_list(ent)
            if concordance_list != []:
                for concordance in concordance_list:
                    dict[ent].append(concordance.line)
for key in dict:
    index = df_pain.index[df_pain['Entity'] == key].tolist()
    df_pain.at[index[0], 'Concordances'].append(dict[key])
df_entities = df_entities.append(df_pain)

med_count = Counter(med_list).most_common(20)
df_med = pd.DataFrame(med_count, columns=['Entity', 'Count'])
df_med['Entity Type'] = 'MED'
df_med['Concordances'] = np.empty((len(df_med), 0)).tolist()
dict = {}
for ent in df_med['Entity']:
    dict[ent] = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    data = df.Tweet.values.tolist()
    words = list(sent_to_words(data))
    for tweet in words:
        textList = Text(tweet)
        for ent in dict:
            concordance_list = textList.concordance_list(ent)
            if concordance_list != []:
                for concordance in concordance_list:
                    dict[ent].append(concordance.line)
for key in dict:
    index = df_med.index[df_med['Entity'] == key].tolist()
    df_med.at[index[0], 'Concordances'].append(dict[key])
df_entities = df_entities.append(df_med)

emotion_count = Counter(emotion_list).most_common(20)
df_emotion = pd.DataFrame(emotion_count, columns=['Entity', 'Count'])
df_emotion['Entity Type'] = 'EMOTION'
df_emotion['Concordances'] = np.empty((len(df_emotion), 0)).tolist()
dict = {}
for ent in df_emotion['Entity']:
    dict[ent] = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    data = df.Tweet.values.tolist()
    words = list(sent_to_words(data))
    for tweet in words:
        textList = Text(tweet)
        for ent in dict:
            concordance_list = textList.concordance_list(ent)
            if concordance_list != []:
                for concordance in concordance_list:
                    dict[ent].append(concordance.line)
for key in dict:
    index = df_emotion.index[df_emotion['Entity'] == key].tolist()
    df_emotion.at[index[0], 'Concordances'].append(dict[key])
df_entities = df_entities.append(df_emotion)

norp_count = Counter(norp_list).most_common(20)
df_norp = pd.DataFrame(norp_count, columns=['Entity', 'Count'])
df_norp['Entity Type'] = 'NORP'
df_norp['Concordances'] = np.empty((len(df_norp), 0)).tolist()
dict = {}
for ent in df_norp['Entity']:
    dict[ent] = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    data = df.Tweet.values.tolist()
    words = list(sent_to_words(data))
    for tweet in words:
        textList = Text(tweet)
        for ent in dict:
            concordance_list = textList.concordance_list(ent)
            if concordance_list != []:
                for concordance in concordance_list:
                    dict[ent].append(concordance.line)
for key in dict:
    index = df_norp.index[df_norp['Entity'] == key].tolist()
    df_norp.at[index[0], 'Concordances'].append(dict[key])
df_entities = df_entities.append(df_norp)

time_count = Counter(time_list).most_common(20)
df_time = pd.DataFrame(time_count, columns=['Entity', 'Count'])
df_time['Entity Type'] = 'TIME'
df_time['Concordances'] = np.empty((len(df_time), 0)).tolist()
dict = {}
for ent in df_time['Entity']:
    dict[ent] = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    data = df.Tweet.values.tolist()
    words = list(sent_to_words(data))
    for tweet in words:
        textList = Text(tweet)
        for ent in dict:
            concordance_list = textList.concordance_list(ent)
            if concordance_list != []:
                for concordance in concordance_list:
                    dict[ent].append(concordance.line)
for key in dict:
    index = df_time.index[df_time['Entity'] == key].tolist()
    df_time.at[index[0], 'Concordances'].append(dict[key])
df_entities = df_entities.append(df_time)

for item in date_list:
    item = item.replace("'", "")
    item = item.replace(']', "")
    item = item.replace(',', "")
date_count = Counter(date_list).most_common(20)
new_date_count = []
for item in date_count:
    item_list = list(item)
    item_list[0] = item_list[0].replace("'", "")
    item_list[0] = item_list[0].replace(']', "")
    item_list[0] = item_list[0].replace(',', "")
    item_list[0] = item_list[0].lstrip()
    item_list[0] = item_list[0].rstrip()
    new_item = tuple(item_list)
    if new_item not in new_date_count:
        new_date_count.append(new_item)
df_date = pd.DataFrame(new_date_count, columns=['Entity', 'Count'])
df_date['Entity Type'] = 'DATE'
df_date['Concordances'] = np.empty((len(df_date), 0)).tolist()
dict = {}
for ent in df_date['Entity']:
    dict[ent] = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    data = df.Tweet.values.tolist()
    words = list(sent_to_words(data))
    for tweet in words:
        textList = Text(tweet)
        for ent in dict:
            concordance_list = textList.concordance_list(ent)
            if concordance_list != []:
                for concordance in concordance_list:
                    dict[ent].append(concordance.line)
for key in dict:
    index = df_date.index[df_date['Entity'] == key].tolist()
    df_date.at[index[0], 'Concordances'].append(dict[key])
df_entities = df_entities.append(df_date)

org_count = Counter(org_list).most_common(20)
df_org = pd.DataFrame(org_count, columns=['Entity', 'Count'])
df_org['Entity Type'] = 'ORG'
df_org['Concordances'] = np.empty((len(df_org), 0)).tolist()
dict = {}
for ent in df_org['Entity']:
    dict[ent] = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    data = df.Tweet.values.tolist()
    words = list(sent_to_words(data))
    for tweet in words:
        textList = Text(tweet)
        for ent in dict:
            concordance_list = textList.concordance_list(ent)
            if concordance_list != []:
                for concordance in concordance_list:
                    dict[ent].append(concordance.line)
for key in dict:
    index = df_org.index[df_org['Entity'] == key].tolist()
    df_org.at[index[0], 'Concordances'].append(dict[key])
df_entities = df_entities.append(df_org)

df.reset_index(drop=True, inplace=True)

# print(df_entities.head(10))
df_entities.to_csv('covid_entities.csv')


df_entities = pd.DataFrame()
pain_list = []
norp_list = []
org_list = []
time_list = []
date_list = []
med_list = []
emotion_list = []

path = r'/home/e69600mm/COMP30030/data/opioids' # use your path
all_files = glob.glob(path + "/*.csv")
li = []
for filename in all_files:
    print(filename)
    df = pd.read_csv(filename, index_col=None, header=0)
    data = df.Tweet.values.tolist() 
    words = list(sent_to_words(data))
    tokens = nlp(''.join(str(words)))
    pain_list.extend(get_pain_entities(tokens))
    norp_list.extend(get_norp_entities(tokens))
    org_list.extend(get_org_entities(tokens))
    time_list.extend(get_time_entities(tokens))
    date_list.extend(get_date_entities(tokens))
    med_list.extend(get_med_entities(tokens))
    emotion_list.extend(get_emotion_entities(tokens))

pain_count = Counter(pain_list).most_common(20)    
df_pain = pd.DataFrame(pain_count, columns=['Entity', 'Count'])
df_pain['Entity Type'] = 'PAIN'
df_pain['Concordances'] = np.empty((len(df_pain), 0)).tolist()
dict = {}
for ent in df_pain['Entity']:
    dict[ent] = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    data = df.Tweet.values.tolist()
    words = list(sent_to_words(data))
    for tweet in words:
        textList = Text(tweet)
        for ent in dict:
            concordance_list = textList.concordance_list(ent)
            if concordance_list != []:
                for concordance in concordance_list:
                    dict[ent].append(concordance.line)
for key in dict:
    index = df_pain.index[df_pain['Entity'] == key].tolist()
    df_pain.at[index[0], 'Concordances'].append(dict[key])
df_entities = df_entities.append(df_pain)

med_count = Counter(med_list).most_common(20)
df_med = pd.DataFrame(med_count, columns=['Entity', 'Count'])
df_med['Entity Type'] = 'MED'
df_med['Concordances'] = np.empty((len(df_med), 0)).tolist()
dict = {}
for ent in df_med['Entity']:
    dict[ent] = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    data = df.Tweet.values.tolist()
    words = list(sent_to_words(data))
    for tweet in words:
        textList = Text(tweet)
        for ent in dict:
            concordance_list = textList.concordance_list(ent)
            if concordance_list != []:
                for concordance in concordance_list:
                    dict[ent].append(concordance.line)
for key in dict:
    index = df_med.index[df_med['Entity'] == key].tolist()
    df_med.at[index[0], 'Concordances'].append(dict[key])
df_entities = df_entities.append(df_med)

emotion_count = Counter(emotion_list).most_common(20)
df_emotion = pd.DataFrame(emotion_count, columns=['Entity', 'Count'])
df_emotion['Entity Type'] = 'EMOTION'
df_emotion['Concordances'] = np.empty((len(df_emotion), 0)).tolist()
dict = {}
for ent in df_emotion['Entity']:
    dict[ent] = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    data = df.Tweet.values.tolist()
    words = list(sent_to_words(data))
    for tweet in words:
        textList = Text(tweet)
        for ent in dict:
            concordance_list = textList.concordance_list(ent)
            if concordance_list != []:
                for concordance in concordance_list:
                    dict[ent].append(concordance.line)
for key in dict:
    index = df_emotion.index[df_emotion['Entity'] == key].tolist()
    df_emotion.at[index[0], 'Concordances'].append(dict[key])
df_entities = df_entities.append(df_emotion)

norp_count = Counter(norp_list).most_common(20)
df_norp = pd.DataFrame(norp_count, columns=['Entity', 'Count'])
df_norp['Entity Type'] = 'NORP'
df_norp['Concordances'] = np.empty((len(df_norp), 0)).tolist()
dict = {}
for ent in df_norp['Entity']:
    dict[ent] = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    data = df.Tweet.values.tolist()
    words = list(sent_to_words(data))
    for tweet in words:
        textList = Text(tweet)
        for ent in dict:
            concordance_list = textList.concordance_list(ent)
            if concordance_list != []:
                for concordance in concordance_list:
                    dict[ent].append(concordance.line)
for key in dict:
    index = df_norp.index[df_norp['Entity'] == key].tolist()
    df_norp.at[index[0], 'Concordances'].append(dict[key])
df_entities = df_entities.append(df_norp)   

time_count = Counter(time_list).most_common(20)
df_time = pd.DataFrame(time_count, columns=['Entity', 'Count'])
df_time['Entity Type'] = 'TIME'
df_time['Concordances'] = np.empty((len(df_time), 0)).tolist()
dict = {}
for ent in df_time['Entity']:
    dict[ent] = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    data = df.Tweet.values.tolist()
    words = list(sent_to_words(data))
    for tweet in words:
        textList = Text(tweet)
        for ent in dict:
            concordance_list = textList.concordance_list(ent)
            if concordance_list != []:
                for concordance in concordance_list:
                    dict[ent].append(concordance.line)
for key in dict:
    index = df_time.index[df_time['Entity'] == key].tolist()
    df_time.at[index[0], 'Concordances'].append(dict[key])
df_entities = df_entities.append(df_time) 

for item in date_list:
    item = item.replace("'", "")
    item = item.replace(']', "")
    item = item.replace(',', "")
date_count = Counter(date_list).most_common(20)
new_date_count = []
for item in date_count:
    item_list = list(item)
    item_list[0] = item_list[0].replace("'", "")
    item_list[0] = item_list[0].replace(']', "")
    item_list[0] = item_list[0].replace(',', "")
    item_list[0] = item_list[0].lstrip()
    item_list[0] = item_list[0].rstrip()
    new_item = tuple(item_list)
    if new_item not in new_date_count:
        new_date_count.append(new_item)
df_date = pd.DataFrame(new_date_count, columns=['Entity', 'Count'])
df_date['Entity Type'] = 'DATE'
df_date['Concordances'] = np.empty((len(df_date), 0)).tolist()
dict = {}
for ent in df_date['Entity']:
    dict[ent] = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    data = df.Tweet.values.tolist()
    words = list(sent_to_words(data))
    for tweet in words:
        textList = Text(tweet)
        for ent in dict:
            concordance_list = textList.concordance_list(ent)
            if concordance_list != []:
                for concordance in concordance_list:
                    dict[ent].append(concordance.line)
for key in dict:
    index = df_date.index[df_date['Entity'] == key].tolist()
    df_date.at[index[0], 'Concordances'].append(dict[key])
df_entities = df_entities.append(df_date)   

org_count = Counter(org_list).most_common(20)
df_org = pd.DataFrame(org_count, columns=['Entity', 'Count'])
df_org['Entity Type'] = 'ORG'
df_org['Concordances'] = np.empty((len(df_org), 0)).tolist()
dict = {}
for ent in df_org['Entity']:
    dict[ent] = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    data = df.Tweet.values.tolist()
    words = list(sent_to_words(data))
    for tweet in words:
        textList = Text(tweet)
        for ent in dict:
            concordance_list = textList.concordance_list(ent)
            if concordance_list != []:
                for concordance in concordance_list:
                    dict[ent].append(concordance.line)
for key in dict:
    index = df_org.index[df_org['Entity'] == key].tolist()
    df_org.at[index[0], 'Concordances'].append(dict[key])
df_entities = df_entities.append(df_org)

df.reset_index(drop=True, inplace=True)

# print(df_entities.head(10))
df_entities.to_csv('opioids_entities.csv')


df_entities = pd.DataFrame()
pain_list = []
norp_list = []
org_list = []
time_list = []
date_list = []
med_list = []
emotion_list = []

path = r'/home/e69600mm/COMP30030/data/opioids_and_covid' # use your path
all_files = glob.glob(path + "/*.csv")
li = []
for filename in all_files:
    print(filename)
    df = pd.read_csv(filename, index_col=None, header=0)
    data = df.Tweet.values.tolist()
    words = list(sent_to_words(data))
    tokens = nlp(''.join(str(words)))
    pain_list.extend(get_pain_entities(tokens))
    norp_list.extend(get_norp_entities(tokens))
    org_list.extend(get_org_entities(tokens))
    time_list.extend(get_time_entities(tokens))
    date_list.extend(get_date_entities(tokens))
    med_list.extend(get_med_entities(tokens))
    emotion_list.extend(get_emotion_entities(tokens))

pain_count = Counter(pain_list).most_common(20)
df_pain = pd.DataFrame(pain_count, columns=['Entity', 'Count'])
df_pain['Entity Type'] = 'PAIN'
df_pain['Concordances'] = np.empty((len(df_pain), 0)).tolist()
dict = {}
for ent in df_pain['Entity']:
    dict[ent] = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    data = df.Tweet.values.tolist()
    words = list(sent_to_words(data))
    for tweet in words:
        textList = Text(tweet)
        for ent in dict:
            concordance_list = textList.concordance_list(ent)
            if concordance_list != []:
                for concordance in concordance_list:
                    dict[ent].append(concordance.line)
for key in dict:
    index = df_pain.index[df_pain['Entity'] == key].tolist()
    df_pain.at[index[0], 'Concordances'].append(dict[key])
df_entities = df_entities.append(df_pain)

med_count = Counter(med_list).most_common(20)
df_med = pd.DataFrame(med_count, columns=['Entity', 'Count'])
df_med['Entity Type'] = 'MED'
df_med['Concordances'] = np.empty((len(df_med), 0)).tolist()
dict = {}
for ent in df_med['Entity']:
    dict[ent] = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    data = df.Tweet.values.tolist()
    words = list(sent_to_words(data))
    for tweet in words:
        textList = Text(tweet)
        for ent in dict:
            concordance_list = textList.concordance_list(ent)
            if concordance_list != []:
                for concordance in concordance_list:
                    dict[ent].append(concordance.line)
for key in dict:
    index = df_med.index[df_med['Entity'] == key].tolist()
    df_med.at[index[0], 'Concordances'].append(dict[key])
df_entities = df_entities.append(df_med)

emotion_count = Counter(emotion_list).most_common(20)
df_emotion = pd.DataFrame(emotion_count, columns=['Entity', 'Count'])
df_emotion['Entity Type'] = 'EMOTION'
df_emotion['Concordances'] = np.empty((len(df_emotion), 0)).tolist()
dict = {}
for ent in df_emotion['Entity']:
    dict[ent] = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    data = df.Tweet.values.tolist()
    words = list(sent_to_words(data))
    for tweet in words:
        textList = Text(tweet)
        for ent in dict:
            concordance_list = textList.concordance_list(ent)
            if concordance_list != []:
                for concordance in concordance_list:
                    dict[ent].append(concordance.line)
for key in dict:
    index = df_emotion.index[df_emotion['Entity'] == key].tolist()
    df_emotion.at[index[0], 'Concordances'].append(dict[key])
df_entities = df_entities.append(df_emotion)

norp_count = Counter(norp_list).most_common(20)
df_norp = pd.DataFrame(norp_count, columns=['Entity', 'Count'])
df_norp['Entity Type'] = 'NORP'
df_norp['Concordances'] = np.empty((len(df_norp), 0)).tolist()
dict = {}
for ent in df_norp['Entity']:
    dict[ent] = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    data = df.Tweet.values.tolist()
    words = list(sent_to_words(data))
    for tweet in words:
        textList = Text(tweet)
        for ent in dict:
            concordance_list = textList.concordance_list(ent)
            if concordance_list != []:
                for concordance in concordance_list:
                    dict[ent].append(concordance.line)
for key in dict:
    index = df_norp.index[df_norp['Entity'] == key].tolist()
    df_norp.at[index[0], 'Concordances'].append(dict[key])
df_entities = df_entities.append(df_norp)

time_count = Counter(time_list).most_common(20)
df_time = pd.DataFrame(time_count, columns=['Entity', 'Count'])
df_time['Entity Type'] = 'TIME'
df_time['Concordances'] = np.empty((len(df_time), 0)).tolist()
dict = {}
for ent in df_time['Entity']:
    dict[ent] = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    data = df.Tweet.values.tolist()
    words = list(sent_to_words(data))
    for tweet in words:
        textList = Text(tweet)
        for ent in dict:
            concordance_list = textList.concordance_list(ent)
            if concordance_list != []:
                for concordance in concordance_list:
                    dict[ent].append(concordance.line)
for key in dict:
    index = df_time.index[df_time['Entity'] == key].tolist()
    df_time.at[index[0], 'Concordances'].append(dict[key])
df_entities = df_entities.append(df_time)

for item in date_list:
    item = item.replace("'", "")
    item = item.replace(']', "")
    item = item.replace(',', "")
date_count = Counter(date_list).most_common(20)
new_date_count = []
for item in date_count:
    item_list = list(item)
    item_list[0] = item_list[0].replace("'", "")
    item_list[0] = item_list[0].replace(']', "")
    item_list[0] = item_list[0].replace(',', "")
    item_list[0] = item_list[0].lstrip()
    item_list[0] = item_list[0].rstrip()
    new_item = tuple(item_list)
    if new_item not in new_date_count:
        new_date_count.append(new_item)
df_date = pd.DataFrame(new_date_count, columns=['Entity', 'Count'])
df_date['Entity Type'] = 'DATE'
df_date['Concordances'] = np.empty((len(df_date), 0)).tolist()
dict = {}
for ent in df_date['Entity']:
    dict[ent] = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    data = df.Tweet.values.tolist()
    words = list(sent_to_words(data))
    for tweet in words:
        textList = Text(tweet)
        for ent in dict:
            concordance_list = textList.concordance_list(ent)
            if concordance_list != []:
                for concordance in concordance_list:
                    dict[ent].append(concordance.line)
for key in dict:
    index = df_date.index[df_date['Entity'] == key].tolist()
    df_date.at[index[0], 'Concordances'].append(dict[key])
df_entities = df_entities.append(df_date)

org_count = Counter(org_list).most_common(20)
df_org = pd.DataFrame(org_count, columns=['Entity', 'Count'])
df_org['Entity Type'] = 'ORG'
df_org['Concordances'] = np.empty((len(df_org), 0)).tolist()
dict = {}
for ent in df_org['Entity']:
    dict[ent] = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    data = df.Tweet.values.tolist()
    words = list(sent_to_words(data))
    for tweet in words:
        textList = Text(tweet)
        for ent in dict:
            concordance_list = textList.concordance_list(ent)
            if concordance_list != []:
                for concordance in concordance_list:
                    dict[ent].append(concordance.line)
for key in dict:
    index = df_org.index[df_org['Entity'] == key].tolist()
    df_org.at[index[0], 'Concordances'].append(dict[key])
df_entities = df_entities.append(df_org)

df.reset_index(drop=True, inplace=True)

# print(df_entities.head(10))
df_entities.to_csv('oc_entities.csv')
