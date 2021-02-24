# Importing TextBlob
from textblob import TextBlob
# Importing the NaiveBayesAnalyzer classifier from NLTK
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import Blobber
tb = Blobber(analyzer=NaiveBayesAnalyzer())
# for manipulating dataframes
import pandas as pd
import glob
import ntpath

df_sentiment = pd.DataFrame()

path = r'/home/e69600mm/COMP30030/data/covid' # use your path
all_files = glob.glob(path + "/*.csv")
for filename in all_files:
    print(ntpath.basename(filename))
    df = pd.read_csv(filename, index_col=None, header=0)
    df = df.dropna()
    df = df.reset_index(drop=True)
    # Applying the NaiveBayesAnalyzer
    for tweet in df.Tweet:
        blob_object = tb(tweet)
        # Running sentiment analysis
        sentiment = blob_object.sentiment
        polarity = blob_object.polarity
        df.loc[df.Tweet == tweet, 'polarity'] = polarity
        if polarity < 0:
            df.loc[df.Tweet == tweet, 'label'] = 'negative'
        elif polarity > 0:
            df.loc[df.Tweet == tweet, 'label'] = 'positive'
        else:
            df.loc[df.Tweet == tweet, 'label'] = 'neutral'
        df.loc[df.Tweet == tweet, 'sentiment'] = sentiment.classification
        df.loc[df.Tweet == tweet, 'p_pos'] = sentiment.p_pos
        df.loc[df.Tweet == tweet, 'p_neg'] = sentiment.p_neg
        df.loc[df.Tweet == tweet, 'subjectivity'] = blob_object.subjectivity
        # print(df.head(10))
        # print(analysis.classification)
    df.to_csv('/home/e69600mm/COMP30030/sentiments/'+ntpath.basename(filename))
    # df_sentiment = df_sentiment.append(df)

# df_sentiment.to_csv('covid_sentiment.csv')
