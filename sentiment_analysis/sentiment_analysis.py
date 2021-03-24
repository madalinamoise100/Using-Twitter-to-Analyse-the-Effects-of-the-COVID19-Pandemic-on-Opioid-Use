# Importing TextBlob
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import Blobber
tb = Blobber(analyzer=NaiveBayesAnalyzer())
# for manipulating dataframes
import pandas as pd
import glob
import ntpath
import matplotlib.pyplot as  plt
# svm
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
# train data
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

def get_sentiment(dataset):
    print(dataset)
    df_total = pd.DataFrame()
    path = r'/local/home/e69600mm/COMP30030/data/' + dataset # use your path
    all_files = glob.glob(path + "/*.csv")
    for filename in all_files:
        print(ntpath.basename(filename))
        df = pd.read_csv(filename, index_col=None, header=0)
        df = df.dropna()
        df = df.reset_index(drop=True)
        # sa with svm
        test_vectors = vectorizer.transform(df['Tweet'])
        df['label'] = classifier_linear.predict(test_vectors)
        # subjectivity with textblob
        for tweet in df.Tweet:
            blob_object = tb(tweet)
            df.loc[df.Tweet == tweet, 'subjectivity'] = blob_object.subjectivity
        print(df['label'].value_counts(normalize=True) * 100)
        df.plot.bar(rot=0)
        df.to_csv('/local/home/e69600mm/COMP30030/sentiments/'+dataset+'/'+ntpath.basename(filename))
        df_total = df_total.append(df)
    # drop low subjectivity
    df_total = df_total[df_total.subjectivity > 0.1]
    df_total = df_total.reset_index()
    pos_file = 'pos_' + dataset + '_sentiment.csv'
    neg_file = 'neg_' + dataset + '_sentiment.csv'
    file = dataset + '_sentiment.csv'
    df_pos = df_total[df_total['label'] == 'pos']
    df_pos.to_csv(pos_file, columns=['Location', 'Tweet', 'label', 'subjectivity'])
    df_neg = df_total[df_total['label'] == 'neg']
    df_neg.to_csv(neg_file, columns=['Location', 'Tweet', 'label', 'subjectivity'])
    df_total.to_csv(file, columns=['Location', 'Tweet', 'label', 'subjectivity'])
    print(df_total['label'].value_counts(normalize=True) * 100)

get_sentiment('covid')
get_sentiment('opioids')
get_sentiment('opioids_and_covid')
