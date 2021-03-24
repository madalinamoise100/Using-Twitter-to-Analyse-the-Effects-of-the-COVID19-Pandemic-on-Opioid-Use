import pandas as pd
import numpy as np
# text blob
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import Blobber
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
tb = Blobber(analyzer=NaiveBayesAnalyzer())
# svm
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
# train Data
trainData = pd.read_csv("https://raw.githubusercontent.com/Vasistareddy/sentiment_analysis/master/data/train.csv")
# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
train_vectors = vectorizer.fit_transform(trainData['Content'])
# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
classifier_linear.fit(train_vectors, trainData['Label'])
# vader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer=SentimentIntensityAnalyzer()


df = pd.read_csv('evaluate_sa.csv', index_col=None, header=0)
df = df.reset_index(drop=True)

# sklearn SVM
test_vectors = vectorizer.transform(df['Tweet'])
df['sklearn SVM'] = classifier_linear.predict(test_vectors)

for tweet in df.Tweet:
    blob_object = tb(tweet)

    # tb pa
    polarity = blob_object.polarity
    if polarity <= -0.1:
        df.loc[df.Tweet == tweet, 'TextBlob (PA)'] = 'Neg'
    elif polarity >= 0.1:
        df.loc[df.Tweet == tweet, 'TextBlob (PA)'] = 'Pos'
    else:
        df.loc[df.Tweet == tweet, 'TextBlob (PA)'] = 'Neu'

    # tb nb
    sentiment = blob_object.sentiment
    if sentiment.classification == 'pos':
        df.loc[df.Tweet == tweet, 'TextBlob (NB)'] = 'Pos'
    elif sentiment.classification == 'neg':
        df.loc[df.Tweet == tweet, 'TextBlob (NB)'] = 'Neg'

    # vader
    score = analyzer.polarity_scores(tweet)
    if score['compound'] < 0.05 and score['compound'] > -0.05:
        df.loc[df.Tweet == tweet, 'Vader'] = 'Neu'
    elif score['compound'] >= 0.05:
        df.loc[df.Tweet == tweet, 'Vader'] = 'Pos'
    else:
        df.loc[df.Tweet == tweet, 'Vader'] = 'Neg'
    df.loc[df.Tweet == tweet, 'Vader compound score'] = score['compound']

total = 130

print('hj 1')
print(df['Human Judgement'].value_counts())
neu_count = df['Human Judgement'].value_counts()['Neu']
print('neu count: ' + str(neu_count))
print()
print('hj 2')
print(df['Human Judgement 2'].value_counts())
neu_count_2 = df['Human Judgement 2'].value_counts()['Neu']
print('neu count: ' + str(neu_count_2))
print()
print('inter rater reliability')
df['Inter-rater reliability'] = np.where(df['Human Judgement'] == df['Human Judgement 2'], 1, 0)
print('inter-rater reliability: ' + str(df['Inter-rater reliability'].value_counts()[1] / total))
print()
df['TextBlob (PA) accuracy 1'] =  np.where(df['Human Judgement'] == df['TextBlob (PA)'], 1, 0)
tb_pa_1 =  df['TextBlob (PA) accuracy 1'].value_counts()[1] / total
print('TextBlob (PA) accuracy 1: ' + str(tb_pa_1))
print()
df['TextBlob (PA) accuracy 2'] =  np.where(df['Human Judgement 2'] == df['TextBlob (PA)'], 1, 0)
tb_pa_2 = df['TextBlob (PA) accuracy 2'].value_counts()[1] / total
print('TextBlob (PA) accuracy 2: ' + str(tb_pa_2))
print()
print('TextBlob (PA) accuracy (average): ' + str((tb_pa_1 + tb_pa_2) / 2))
print()
df['TextBlob (NB) accuracy 1'] =  np.where(df['Human Judgement'] == df['TextBlob (NB)'], 1, 0)
tb_nb_1 = df['TextBlob (NB) accuracy 1'].value_counts()[1] / (total - neu_count)
print('TextBlob (NB) accuracy 1: ' + str(tb_nb_1))
print()
df['TextBlob (NB) accuracy 2'] =  np.where(df['Human Judgement 2'] == df['TextBlob (NB)'], 1, 0)
tb_nb_2 = df['TextBlob (NB) accuracy 2'].value_counts()[1] / (total - neu_count_2)
print('TextBlob (NB) accuracy 2: ' + str(tb_nb_2))
print()
print('TextBlob (NB) accuracy (average): ' + str((tb_nb_1 + tb_nb_2) / 2))
print()
df['Vader accuracy 1'] =  np.where(df['Human Judgement'] == df['Vader'], 1, 0)
vader_1 = df['Vader accuracy 1'].value_counts()[1] / total
print('Vader accuracy 1: ' + str(vader_1))
print()
df['Vader accuracy 2'] =  np.where(df['Human Judgement 2'] == df['Vader'], 1, 0)
vader_2 = df['Vader accuracy 2'].value_counts()[1] / total
print('Vader accuracy 2: ' + str(vader_2))
print()
print('Vader accuracy (average): ' + str((vader_1 + vader_2) / 2))
print()
df['sklearn accuracy 1'] =  np.where(df['Human Judgement'].str.lower() == df['sklearn SVM'], 1, 0)
sklearn_1 = df['sklearn accuracy 1'].value_counts()[1] / (total - neu_count)
print('sklearn accuracy 1: ' + str(sklearn_1))
print()
df['sklearn accuracy 2'] =  np.where(df['Human Judgement 2'].str.lower() == df['sklearn SVM'], 1, 0)
sklearn_2 = df['sklearn accuracy 2'].value_counts()[1] / (total - neu_count_2)
print('sklearn accuracy 2: ' + str(sklearn_2))
print()
print('sklearn accuracy (average): ' + str((sklearn_1 + sklearn_2) / 2))
print()

df.to_csv('evaluate_sa.csv', columns = ['Tweet', 'Human Judgement', 'Human Judgement 2', 'TextBlob (PA)', 'TextBlob (NB)', 'sklearn SVM', 'Vader'])
