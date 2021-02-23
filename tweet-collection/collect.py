import os
import tweepy as tw
import pandas as pd
pd.set_option('display.max_columns', None)
import datetime as DT
import re

consumer_key= ''
consumer_secret= ''
access_token= ''
access_token_secret= ''

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

filters = " -filter:retweets" + " +me OR I OR mine OR my"
# + " -filter:links"

def create_opioid_search_query():
    print("Creating opioid query")
    query=""
    file = open("opioid_keywords.txt", "r")
    lines = file.readlines()
    last = lines[-1]
    for line in lines:
        if line is last:
            line = line.strip()
            query = query + line
        else:
            line = line.strip()
            query = query + line + " OR "
    query = query + file.readline()
    file.close()
    return query

def create_covid_search_query():
    print("Creating covid query")
    query=""
    file = open("covid_keywords.txt", "r")
    lines = file.readlines()
    last = lines[-1]
    for line in lines:
        if line is last:
            line = line.strip()
            query = query + line
        else:
            line = line.strip()
            query = query + line + " OR "
    file.close()
    return query

def clean_data(data):
    # Filter out empty entries
    data = data[data.Location != ""]
    # Remove mentions
    data['Tweet'] = data['Tweet'].str.replace("@[A-Za-z0-9]+([._]?[a-zA-Z0-9]+)+", "")
    # Remove newline characters
    data['Tweet'] = data['Tweet'].str.replace('\n', ' ')
    # Reset index
    data = data.reset_index(drop=True)
    return data

def get_data(query, location):
    today = DT.date.today()
    print("Today " + today.strftime("%Y-%m-%d"))
    date_since = today - DT.timedelta(days=7)
    print("Since " + date_since.strftime("%Y-%m-%d"))
    file_name = date_since.strftime("%Y-%m-%d") + "_to_" + today.strftime("%Y-%m-%d") + ".csv"
    print("File name " + file_name)
    tweets = tw.Cursor(api.search,
                q=query,
                lang="en",
                tweet_mode="extended",
                since=date_since).items(1500)
    print("Collected tweets")
    # for tweet in tweets:
    #     print(tweet.user.location)
    data = pd.DataFrame(data=[[tweet.user.location, tweet.full_text] for tweet in tweets], columns=['Location', 'Tweet'])
    print("Got data")
    data = clean_data(data)
    data.to_csv(os.path.join(os.path.split(os.path.split(os.path.dirname(__file__))[0])[0], 'data/' + location, file_name), index=False)
    return data

def get_opioids_and_covid_data():
    print("Getting opioids and covid data")
    opioid_query = create_opioid_search_query()
    covid_query = create_covid_search_query()
    search_query = opioid_query + " AND " + covid_query + filters
    # print(search_query)
    return get_data(search_query, "opioids_and_covid")

def get_opioids_data():
    print("Getting opioids data")
    opioid_query = create_opioid_search_query() + filters
    return get_data(opioid_query, "opioids")

def get_covid_data():
    print("Getting covid data")
    covid_query = create_covid_search_query() + filters
    return get_data(covid_query, "covid")

def main():
    get_opioids_data()
    get_covid_data()
    get_opioids_and_covid_data()

if __name__ == '__main__':
    main()
