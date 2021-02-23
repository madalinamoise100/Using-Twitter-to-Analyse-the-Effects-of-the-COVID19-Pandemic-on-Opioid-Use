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

# def get_token_sent(token):
#     token_span = token.doc[token.i:token.i+1]
#     return token_span.sent
#
# Token.set_extension('sent', getter=get_token_sent)

def get_sentence(token):
    for sent in token.doc.sents:
        if sent.start <= token.i:
            return sent

# Add a computed property, which will be accessible as token._.sent
Token.set_extension('sent', getter=get_sentence)

pain_dict = {'pain', 'hurts', 'hurt', 'sick', 'injured', 'injure', 'blood', 'discomfort',
            'uncomfortable', 'cry', 'scream', 'suffer', 'suffered', 'agony', 'ache', 'ached'}
def get_pain_entities(tokens):
    pain_list = []
    for token in tokens:
        if token.text in pain_dict:
            # print(token.text)
            # print(token._.sent)
            # print()
            pain_list.append(token.text)
    return pain_list

df_entities = pd.DataFrame()
pain_list = []
norp_list = []
org_list = []
time_list = []
date_list = []

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

path = r'/Users/maddutz/Desktop/tweet-collection/data/covid' # use your path
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

pain_count = Counter(pain_list).most_common(20)
df_pain = pd.DataFrame(pain_count, columns=['Entity', 'Count'])
df_pain['Entity Type'] = 'PAIN'
df_entities = df_entities.append(df_pain)

norp_count = Counter(norp_list).most_common(20)
df_norp = pd.DataFrame(norp_count, columns=['Entity', 'Count'])
df_norp['Entity Type'] = 'NORP'
df_entities = df_entities.append(df_norp)

time_count = Counter(time_list).most_common(20)
df_time = pd.DataFrame(time_count, columns=['Entity', 'Count'])
df_time['Entity Type'] = 'TIME'
df_entities = df_entities.append(df_time)

date_count = Counter(date_list).most_common(20)
df_date = pd.DataFrame(date_count, columns=['Entity', 'Count'])
df_date['Entity Type'] = 'DATE'
df_entities = df_entities.append(df_date)

org_count = Counter(org_list).most_common(20)
df_org = pd.DataFrame(org_count, columns=['Entity', 'Count'])
df_org['Entity Type'] = 'ORG'

df_entities = df_entities.reset_index()

# print(df_entities.head(10))
df_entities.to_csv('covid_entities.csv')
