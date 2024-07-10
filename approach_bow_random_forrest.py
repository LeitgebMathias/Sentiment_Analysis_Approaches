import pandas as pd
from bs4 import BeautifulSoup    
import re
import nltk
from nltk.corpus import stopwords # Import the stop word list
from  text_segmentation import text_segmentation as ts

train = pd.read_csv("./data/labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)


num_reviews = train["review"].size

print ("Cleaning and parsing the training set movie reviews...\n")

clean_train_reviews = []
for i in range( 0, num_reviews ):
    if( (i+1)%1000 == 0 ):
        print ("Clean Review %d of %d\n" % ( i+1, num_reviews ))
    clean_train_reviews.append( ts.review_to_words( train["review"][i] ))

print ("Creating the bag of words...\n")
from sklearn.feature_extraction.text import CountVectorizer

# "CountVectorizer" is scikit-learn's bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

# fit_transform() does two functions: 
# 1. Fits the model and learns the vocabulary; 
# 2. Transforms our training data into feature vectors. 

train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an array
train_data_features = train_data_features.toarray()
print (train_data_features.shape)


print ("Training the random forest...")
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the Naive Bayes Model to the training set, using the bag of words as 
# features and the sentiment labels.
forest = forest.fit( train_data_features, train["sentiment"] )

test = pd.read_csv("./data/testData.tsv", header=0, delimiter="\t", \
                   quoting=3 )


# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])
clean_test_reviews = [] 

print ("Cleaning and parsing the test set movie reviews...\n")
for i in range(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print ("Review %d of %d\n"  % (i+1, num_reviews))
    clean_review = ts.review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_Random_Forrest.csv", index=False, quoting=3 )

