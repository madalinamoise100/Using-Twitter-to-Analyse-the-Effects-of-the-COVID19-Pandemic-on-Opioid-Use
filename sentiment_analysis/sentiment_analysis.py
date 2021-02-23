# Importing TextBlob
from textblob import TextBlob
# Importing the NaiveBayesAnalyzer classifier from NLTK
from textblob.sentiments import NaiveBayesAnalyzer
# for manipulating dataframes
import pandas as pd
import glob

df_sentiment = pd.DataFrame()

path = r'/Users/maddutz/Desktop/tweet-collection/data/covid' # use your path
all_files = glob.glob(path + "/*.csv")
for filename in all_files:
    print(filename)
    df = pd.read_csv(filename, index_col=None, header=0)
    df = df.dropna()
    df = df.reset_index(drop=True)
    # Applying the NaiveBayesAnalyzer
    for tweet in df.Tweet:
        blob_object = TextBlob(tweet, analyzer=NaiveBayesAnalyzer())
        # Running sentiment analysis
        analysis = blob_object.sentiment
        df.loc[df.Tweet == tweet, 'classification'] = analysis.classification
        df.loc[df.Tweet == tweet, 'p_pos'] = analysis.p_pos
        df.loc[df.Tweet == tweet, 'p_neg'] = analysis.p_neg
        # print(df.head(10))
        # print(analysis.classification)
    df_sentiment = df_sentiment.append(df)

df_sentiment.to_csv('covid_sentiment.csv')












# from flair.models import TextClassifier
# from flair.data import Sentence
# from segtok.segmenter import split_single
# import pandas as pd
# import re
#
#
# classifier = TextClassifier.load('en-sentiment')
#
# def clean(raw):
#     """ Remove hyperlinks and markup """
#     result = re.sub("<[a][^>]*>(.+?)</[a]>", 'Link.', raw)
#     result = re.sub('&gt;', "", result)
#     result = re.sub('&#x27;', "'", result)
#     result = re.sub('&quot;', '"', result)
#     result = re.sub('&#x2F;', ' ', result)
#     result = re.sub('<p>', ' ', result)
#     result = re.sub('</i>', '', result)
#     result = re.sub('&#62;', '', result)
#     result = re.sub('<i>', ' ', result)
#     result = re.sub("\n", '', result)
#     return result
#
# def make_sentences(text):
#     """ Break apart text into a list of sentences """
#     sentences = [sent for sent in split_single(text)]
#     return sentences
#
# def predict(sentence):
#     """ Predict the sentiment of a sentence """
#     if sentence == "":
#         return 0
#     text = Sentence(sentence)
#     # stacked_embeddings.embed(text)
#     classifier.predict(text)
#     value = text.labels[0].to_dict()['value']
#     if value == 'POSITIVE':
#         result = text.to_dict()['labels'][0]['confidence']
#     else:
#         result = -(text.to_dict()['labels'][0]['confidence'])
#     return round(result, 3)
#
# def get_scores(sentences):
#     """ Call predict on every sentence of a text """
#     results = []
#
#     for i in range(0, len(sentences)):
#         results.append(predict(sentences[i]))
#     return results
#
# def get_sum(scores):
#
#     result = round(sum(scores), 3)
#     return result
#
# for filename in all_files:
#     print(filename)
#     df = pd.read_csv(filename, index_col=None, header=0)
#     df = df.dropna()
#     df = df.reset_index(drop=True)
#     df.Tweet = df.Tweet.apply(clean)
#     df['sentences'] = df.Tweet.apply(make_sentences)
#     df['sentences'] = df['sentences'].apply(get_scores)
#     df['sentences'] = df.scores.apply(get_sum)
#     print(df.head(10))
#     # data = df.Tweet.values.tolist()
#
#
# df = pd.read_json('small.json')
# df = df.dropna()
# df = df.reset_index(drop=True)
# df.text = df.text.apply(clean)
# df['sentences'] = df.text.apply(make_sentences)
# df['scores'] = df['sentences'].apply(get_scores)
# df['scores_sum'] = df.scores.apply(get_sum)
