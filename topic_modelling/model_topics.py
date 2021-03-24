import re
import numpy as np
import pandas as pd
import glob
from pprint import pprint
import seaborn as sns
import sys

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
import seaborn as sns
import matplotlib.colors as mcolors
from wordcloud import WordCloud, STOPWORDS

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

# --- Find dominant topic and its percentage contribution in each document
def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()
    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

# --- Visualise the topics
def visualise_topics(output_path, dataset, lda_model, corpus, id2word):
    print("Preparing model")
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds='mmds')
    print("Saving visualisation")
    vis_file = output_path + dataset + '_LDA_Visualization.html'
    pyLDAvis.save_html(vis, vis_file)

# --- Find dominant topic and its percentage contribution in each document
def dominant_topics(output_path, dataset, df_topic_sents_keywords):
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    dominant_topic_file = output_path + dataset + '_dominant_topics.csv'
    df_dominant_topic.to_csv(dominant_topic_file, index=False)
    print(df_dominant_topic.head(10))
    print(df_dominant_topic['Dominant_Topic'].value_counts(normalize=True) * 100)

# --- Find the most representative sentence for each topic
def representative_sentence_per_topic(output_path, dataset, df_topic_sents_keywords):
    # Display setting to show more characters in column
    pd.options.display.max_colwidth = 100
    sent_topics_sorteddf_mallet = pd.DataFrame()
    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')
    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                                 grp.sort_values(['Perc_Contribution'], ascending=False).head(1)],
                                                axis=0)
    # Reset Index
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
    # Format
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]
    # Show
    repr_file = output_path + dataset + '_representative_sentence_per_topic.csv'
    sent_topics_sorteddf_mallet.to_csv(repr_file, index=False)
    print(sent_topics_sorteddf_mallet.head(10))

# -- Word Clouds of Top N Keywords in Each Topic
def word_clouds(output_path, dataset, lda_model, num_topics, x, y):
    cols = [mcolors.XKCD_COLORS['xkcd:aqua'], mcolors.XKCD_COLORS['xkcd:azure'], mcolors.XKCD_COLORS['xkcd:green'], mcolors.XKCD_COLORS['xkcd:magenta'], mcolors.XKCD_COLORS['xkcd:orange'], mcolors.XKCD_COLORS['xkcd:orchid'], mcolors.XKCD_COLORS['xkcd:khaki'],
    mcolors.XKCD_COLORS['xkcd:red'], mcolors.XKCD_COLORS['xkcd:fuchsia'], mcolors.XKCD_COLORS['xkcd:darkgreen'], mcolors.XKCD_COLORS['xkcd:teal'], mcolors.XKCD_COLORS['xkcd:salmon']]
    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=25,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)
    topics = lda_model.show_topics(num_topics=num_topics, num_words=25, formatted=False)
    fig, axes = plt.subplots(int(x), int(y), figsize=(10,10), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    wc_file = output_path + dataset + "_word_clouds.png"
    
# --- Topic modelling
def tm(data_path, path_type, output_path, dataset, num_topics, x, y, generic_tokens):
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
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=num_topics,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               # iterations=100,
                                               per_word_topics=True)
    # Print to file
    orig_stdout = sys.stdout
    topics_file = output_path + dataset + '_topics'
    with open(topics_file, 'w') as f:
        sys.stdout = f
        pprint(lda_model.print_topics())
        # --- Evaluate model
        # Compute Perplexity
        print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_ready, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
    # Stop printing to file
    f.close()
    sys.stdout = orig_stdout
    # Visualize the topics
    visualise_topics(output_path, dataset, lda_model, corpus, id2word)
    # Find dominant topic and its percentage contribution in each document
    df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_ready)
    dominant_topics(output_path, dataset, df_topic_sents_keywords)
    # Find the most representative sentence for each topic
    representative_sentence_per_topic(output_path, dataset, df_topic_sents_keywords)
    # -- Word Clouds of Top N Keywords in Each Topic
    word_clouds(output_path, dataset, lda_model, num_topics, x, y)


model_topics('/local/home/e69600mm/COMP30030/sentiments/covid', 'dir', '/Users/maddutz/Desktop/COMP30030/', 'covid', 8, int(4), int(2), ['corona', 'covid', 'coronavirus'])
model_topics('/local/home/e69600mm/COMP30030/sentiments/opioids', 'dir', '/Users/maddutz/Desktop/COMP30030/', 'opioids', 8, int(4), int(2), ['opioid', 'opioids'])
model_topics('/local/home/e69600mm/COMP30030/sentiments/opioids_and_covid', 'dir', '/Users/maddutz/Desktop/COMP30030/', 'oc', 8, int(4), int(2), ['opioid', 'opioids', 'corona', 'covid', 'coronavirus'])
model_topics('/local/home/e69600mm/COMP30030/pos_covid_sentiment.csv', 'file', '/Users/maddutz/Desktop/COMP30030/', 'pos_covid', 8, int(4), int(2), ['corona', 'covid', 'coronavirus'])
model_topics('/local/home/e69600mm/COMP30030/neg_covid_sentiment.csv', 'file', '/Users/maddutz/Desktop/COMP30030/', 'neg_covid', 8, int(4), int(2), ['corona', 'covid', 'coronavirus'])
model_topics('/local/home/e69600mm/COMP30030/neg_opioids_sentiment.csv', 'file', '/Users/maddutz/Desktop/COMP30030/', 'neg_opioids', 8, int(4), int(2), ['opioid', 'opioids'])
model_topics('/local/home/e69600mm/COMP30030/pos_opioids_sentiment.csv', 'file', '/Users/maddutz/Desktop/COMP30030/', 'pos_opioids', 8, int(4), int(2), ['opioid', 'opioids'])
model_topics('/local/home/e69600mm/COMP30030/pos_oc_sentiment.csv', 'file', '/Users/maddutz/Desktop/COMP30030/', 'pos_oc', 8, int(4), int(2), ['opioid', 'opioids', 'corona', 'covid', 'coronavirus'])
model_topics('/local/home/e69600mm/COMP30030/neg_oc_sentiment.csv', 'file', '/Users/maddutz/Desktop/COMP30030/', 'neg_oc', 8, int(4), int(2), ['opioid', 'opioids', 'corona', 'covid', 'coronavirus'])
 
