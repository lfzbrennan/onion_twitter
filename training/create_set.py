### Collects all onion and not onion training data
### Partly adapted from  https://github.com/lukefeilberg/onion

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import time
import requests
import json
import csv
import datetime
import sys

warnings.filterwarnings('ignore')

dates_list = []

# Making list of dates; Each January 1st and June 1st from 2015 until January 1st 2020
for i in range(10,21):
    dates_list.append('01/01/20'+str(i))
    dates_list.append('01/06/20'+str(i))

def getTimeStamp(date_input):
    return time.mktime(datetime.datetime.strptime(date_input, "%d/%m/%Y").timetuple())

dates = [int(getTimeStamp(date)) for date in dates_list]


# send push data request
def getPushshiftData(after, before, sub):
    url = ('https://api.pushshift.io/reddit/search/submission/?size=1000&after='+
           str(after)+'&before='+str(before)+'&subreddit='+str(sub)+'&sort_type=score'+'&sort=desc')
    print(url)
    r = requests.get(url)
    data = json.loads(r.text)
    return data['data']


def getTitles(subreddit):
    titles_new = []
    titles = []

    for i in range(len(dates)-1):
        # Setting up dates
        after  = dates[i]
        before = dates[i+1]

        # Getting subreddit data between the dates after and before from r/NotTheOnion
        raw_json = getPushshiftData(after, before, subreddit)

        # Extracting just the title
        titles_new = [post['title'] for post in raw_json]

        # Appending new data on
        titles = titles + titles_new
        time.sleep(.5)

    # A few posts were extracted twice, set gets rid of duplicates
    titles = list(set(titles))
    return titles


# get titles
not_onion = getTitles('nottheonion')
onion = getTitles('theonion')

#   REMOVE THIS
with open("OnionOrNot.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        if row[1] == "1":
            onion += [row[0]]
        else:
            not_onion += [row[0]]


onion = list(set(onion))
not_onion = list(set(not_onion))

print(len(onion))
print(len(not_onion))

# convert to numpy arrays and save
not_onion = np.array(not_onion)
onion = np.array(onion)

np.save("raw_headlines/not_onion.npy", not_onion)
np.save("raw_headlines/onion.npy", onion)
