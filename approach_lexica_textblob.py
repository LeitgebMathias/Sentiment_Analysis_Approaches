from textblob import TextBlob
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords # Import the stop word list
from  text_segmentation import text_segmentation as ts

# Read the test data
data = pd.read_csv("./data/testData.tsv", header=0, delimiter="\t", quoting=3 )

num_reviews = len(data["review"])

clean_train_reviews = []
for i in range( 0, num_reviews ):
    clean_train_reviews.append( ts.review_to_words( data["review"][i] ))

richtig = 0


for i in range(0,num_reviews):
    text = clean_train_reviews[i]
    blob = TextBlob(text)
    polarity_textblob = blob.sentiment.polarity
    polarity_textblob = round((polarity_textblob + 1) * 5)

    if (i + 1) % 1000 == 0:
        print("Review %d of %d" % (i + 1, num_reviews))

    true_sentiment = int(data["id"][i].split("_")[1].replace('"', ''))

    if polarity_textblob > 5 and true_sentiment > 5:
        richtig += 1
    elif polarity_textblob <= 5 and true_sentiment <= 5:
        richtig += 1

# Dividiere Richtige durch Anzahl der Datensätze
print ("Richtig: " + str(richtig))
print ("Anzahl der Datensätze: " + str(data.shape[0]))
print("Accuracy: " + str(richtig / data.shape[0]))