import re
import numpy as np
import pandas as pd
import glob
from pprint import pprint
import seaborn as sns
import sys
from itertools import chain

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

# --- Get dataset
path = r'/Users/maddutz/Desktop/tweet-collection/data/covid'
all_files = glob.glob(path + "/*.csv")
li = []
for filename in all_files:
    frame = pd.read_csv(filename, index_col=None, header=0)
    li.append(frame)
df = pd.concat(li, axis=0, ignore_index=True)
# print(df.Location.unique())
# df.head()

# --- Clean data
data = df.Tweet.values.tolist()
# Remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]
# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]
# pprint(data[:1])

# --- Tokenise
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
data_words = list(sent_to_words(data))
# print(data_words[:1])

# --- Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
# print(trigram_mod[bigram_mod[data_words[0]]])

# --- Further preprocessing
def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]
    return texts_out

# --- Create LDA model
data_ready = process_words(data_words)  # processed text data
# Create Dictionary
id2word = corpora.Dictionary(data_ready)
# Create Corpus: Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_ready]
# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=26,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           # iterations=100,
                                           per_word_topics=True)

# Print to file
orig_stdout = sys.stdout
with open('/Users/maddutz/Desktop/tweet-collection/topic_modelling/covid_topics', 'w') as f:
    sys.stdout = f
    pprint(lda_model.print_topics())

    # # Assigns the topics to the documents in corpus
    # lda_corpus = lda_model[corpus]
    # # print(lda_corpus)
    #
    # # Find the threshold, let's set the threshold to be 1/#clusters,
    # # To prove that the threshold is sane, we average the sum of all probabilities:
    # scores = []
    # for doc in lda_corpus:
    #     # print(doc)
    #     # print()
    #     for topic in doc:
    #         # print('TOPIC START')
    #         # print(topic)
    #         # print('TOPIC END')
    #         # print()
    #         for topic_id, score in topic:
    #             # print(type(score))
    #             if isinstance(score, np.float32):
    #                 scores.append(score)
    # # scores = list(chain(*[[score for topic_id,score in topic] \
    #                     # for topic in [doc for doc in lda_corpus]]))
    # print('AAAAAAAAAAAA')
    # print(scores)
    # print('BBBBBBBBBBBBB')
    # threshold = sum(scores)/len(scores)
    # print(threshold)
    # print()
    #
    # cluster1 = [j for i,j in zip(lda_corpus,data) if i[0][1] > threshold]
    # cluster2 = [j for i,j in zip(lda_corpus,data) if i[1][1] > threshold]
    # cluster3 = [j for i,j in zip(lda_corpus,data) if i[2][1] > threshold]
    #
    # print(cluster1)
    # print(cluster2)
    # print(cluster3)

    # --- Evaluate model
    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is --> lower the better
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_ready, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

# Stop printing to file
f.close()
sys.stdout = orig_stdout

# --- Visualize the topics
print("Preparing model")
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds='mmds')
print("Saving visualisation")
pyLDAvis.save_html(vis, 'covid_LDA_Visualization.html')

# --- Find dominant topic and its percentage contribution in each document
def format_topics_sentences(ldamodel=None, corpus=corpus, texts=data):
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

df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_ready)
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.to_csv('/Users/maddutz/Desktop/tweet-collection/topic_modelling/covid_dominant_topics.csv', index=False)
print(df_dominant_topic.head(10))


# --- Find the most representative sentence for each topic
# Display setting to show more characters in column
pd.options.display.max_colwidth = 100

sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')
# print(sent_topics_outdf_grpd.head(10))

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                             grp.sort_values(['Perc_Contribution'], ascending=False).head(1)],
                                            axis=0)

# Reset Index
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]
# Show
sent_topics_sorteddf_mallet.to_csv('/Users/maddutz/Desktop/tweet-collection/topic_modelling/covid_representative_sentence_per_topic.csv', index=False)
print(sent_topics_sorteddf_mallet.head(10))


# # --- Frequency Distribution of Word Counts in Documents
# doc_lens = [len(d) for d in df_dominant_topic.Text]
#
# # Plot
# plt.figure(figsize=(16,7), dpi=160)
# plt.hist(doc_lens, bins = 50, color='navy')
# plt.text(40, 100, "Mean   : " + str(round(np.mean(doc_lens))))
# plt.text(40,  90, "Median : " + str(round(np.median(doc_lens))))
# plt.text(40,  80, "Stdev   : " + str(round(np.std(doc_lens))))
# plt.text(40,  70, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
# plt.text(40,  60, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))
#
# plt.gca().set(xlim=(0, 50), ylabel='Number of Documents', xlabel='Document Word Count')
# plt.tick_params(size=16)
# plt.xticks(np.linspace(0,50,9))
# plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))
# # plt.show()
# plt.savefig('covid_distribution_of_document_word_counts.png')


# # --- Distribution of Document Word Counts by Dominant Topic
cols = [mcolors.XKCD_COLORS['xkcd:aqua'], mcolors.XKCD_COLORS['xkcd:azure'], mcolors.XKCD_COLORS['xkcd:green'], mcolors.XKCD_COLORS['xkcd:magenta'], mcolors.XKCD_COLORS['xkcd:orange'], mcolors.XKCD_COLORS['xkcd:orchid'], mcolors.XKCD_COLORS['xkcd:khaki'],
        mcolors.XKCD_COLORS['xkcd:red'], mcolors.XKCD_COLORS['xkcd:fuchsia'], mcolors.XKCD_COLORS['xkcd:darkgreen'], mcolors.XKCD_COLORS['xkcd:teal'], mcolors.XKCD_COLORS['xkcd:salmon'],
        mcolors.XKCD_COLORS['xkcd:red'], mcolors.XKCD_COLORS['xkcd:fuchsia'], mcolors.XKCD_COLORS['xkcd:darkgreen'], mcolors.XKCD_COLORS['xkcd:teal'], mcolors.XKCD_COLORS['xkcd:salmon'],
        mcolors.XKCD_COLORS['xkcd:red'], mcolors.XKCD_COLORS['xkcd:fuchsia'], mcolors.XKCD_COLORS['xkcd:darkgreen'], mcolors.XKCD_COLORS['xkcd:teal'], mcolors.XKCD_COLORS['xkcd:salmon'],
        mcolors.XKCD_COLORS['xkcd:red'], mcolors.XKCD_COLORS['xkcd:fuchsia'], mcolors.XKCD_COLORS['xkcd:darkgreen'], mcolors.XKCD_COLORS['xkcd:teal'], mcolors.XKCD_COLORS['xkcd:salmon']]
#
# fig, axes = plt.subplots(4,3,figsize=(16,14), dpi=160, sharex=True, sharey=True)
#
# for i, ax in enumerate(axes.flatten()):
#     df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == i, :]
#     doc_lens = [len(d) for d in df_dominant_topic_sub.Text]
#     ax.hist(doc_lens, bins = 50, color=cols[i])
#     ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
#     sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())
#     ax.set(xlim=(0, 50), xlabel='Document Word Count')
#     ax.set_ylabel('Number of Documents', color=cols[i])
#     ax.set_title('Topic: '+str(i), fontdict=dict(size=16, color=cols[i]))
#
# fig.tight_layout()
# fig.subplots_adjust(top=0.90)
# plt.xticks(np.linspace(0,50,9))
# fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=22)
# plt.savefig('covid_distribution_of_document_word_counts_by_dominant_topic.png')


# -- Word Clouds of Top N Keywords in Each Topic
cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model.show_topics(num_topics=26, num_words=15, formatted=False)

fig, axes = plt.subplots(13, 2, figsize=(50,50), sharex=True, sharey=True)

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
plt.savefig('covid_word_clouds.png')
