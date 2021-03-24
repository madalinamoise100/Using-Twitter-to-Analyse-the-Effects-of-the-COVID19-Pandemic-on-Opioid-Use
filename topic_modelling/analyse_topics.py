import re
import numpy as np
import pandas as pd
import glob
from pprint import pprint
from spellchecker import SpellChecker
spell = SpellChecker()

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import logging
logging.captureWarnings(True)

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

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

# --- Clean data
def clean_data(data):
    # Convert to lower-case
    data = [sent.lower() for sent in data]
    # Remove URLs
    data = [re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', sent) for sent in data]
    # Remove emails
    data = [re.sub('\S*@\S*\s?', 'EMAIL', sent) for sent in data]
    # Remove usernames
    data = [re.sub('@[^\s]+', 'AT_USER', sent) for sent in data]
    # Remove hashtags
    data = [re.sub(r'#([^\s]+)', r'\1', sent) for sent in data]
    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]
    # Remove single quotes
    data = [re.sub("\'", "", sent) for sent in data]
    return data

# --- Tokenise
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

# --- Further preprocessing
def process_words(texts, bigram_mod, trigram_mod, generic_tokens, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    # Remove Stopwords
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    # Remove generic tokens
    texts = [[word for word in simple_preprocess(str(doc)) if word not in generic_tokens] for doc in texts]
    # Correct misspellings
    # for doc in texts:
    #     for word in simple_preprocess(str(doc)):
    #         if word not in spell:
    #             print(doc)
    #             print(word)
    #             print(spell.correction(word))
    #             print()
    # texts = [[spell.correction(word) for word in simple_preprocess(str(doc)) if word not in spell] for doc in texts]
    # Form bigrams and trigrams
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    # Lemmatise
    texts_out = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # Remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]
    return texts_out

# --- Compute coherence values
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        # model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=dictionary,
                                                num_topics=num_topics,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

# --- Analyse topics based on coherence scores
def analyse_topics(data_path, path_type, output_path, dataset, generic_tokens):
    print()
    print(dataset)
    # Get data
    df = get_data(data_path, path_type)
    # --- Clean data
    data = df.Tweet.values.tolist()
    data = clean_data(data)
    data_words = list(sent_to_words(data))
    # --- Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold => fewer phrases
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    # --- Create LDA model
    data_ready = process_words(data_words, bigram_mod, trigram_mod, generic_tokens)  # processed text data
    # Create Dictionary
    id2word = corpora.Dictionary(data_ready)
    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_ready]
    # Compute coherence values
    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_ready, start=2, limit=40, step=2)
    # Show graph
    limit=40; start=2; step=2;
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.savefig('covid_coherence_values_per_num_topics.png')
    # Print the coherence scores
    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

analyse_topics('/local/home/e69600mm/COMP30030/sentiments/covid', 'dir', '/Users/maddutz/Desktop/COMP30030/', 'covid', ['corona', 'covid', 'coronavirus'])
analyse_topics('/local/home/e69600mm/COMP30030/sentiments/opioids', 'dir', '/Users/maddutz/Desktop/COMP30030/', 'opioids', ['opioid', 'opioids'])
analyse_topics('/local/home/e69600mm/COMP30030/sentiments/opioids_and_covid', 'dir', '/Users/maddutz/Desktop/COMP30030/', 'oc', ['opioid', 'opioids', 'corona', 'covid', 'coronavirus'])
analyse_topics('/local/home/e69600mm/COMP30030/pos_covid_sentiment.csv', 'file', '/Users/maddutz/Desktop/COMP30030/', 'pos_covid', ['corona', 'covid', 'coronavirus'])
analyse_topics('/local/home/e69600mm/COMP30030/neg_covid_sentiment.csv', 'file', '/Users/maddutz/Desktop/COMP30030/', 'neg_covid', ['corona', 'covid', 'coronavirus'])
analyse_topics('/local/home/e69600mm/COMP30030/neg_opioids_sentiment.csv', 'file', '/Users/maddutz/Desktop/COMP30030/', 'neg_opioids', ['opioid', 'opioids'])
analyse_topics('/local/home/e69600mm/COMP30030/pos_opioids_sentiment.csv', 'file', '/Users/maddutz/Desktop/COMP30030/', 'pos_opioids', ['opioid', 'opioids'])
analyse_topics('/local/home/e69600mm/COMP30030/pos_oc_sentiment.csv', 'file', '/Users/maddutz/Desktop/COMP30030/', 'pos_oc', ['opioid', 'opioids', 'corona', 'covid', 'coronavirus'])
analyse_topics('/local/home/e69600mm/COMP30030/neg_oc_sentiment.csv', 'file', '/Users/maddutz/Desktop/COMP30030/', 'neg_oc', ['opioid', 'opioids', 'corona', 'covid', 'coronavirus'])

