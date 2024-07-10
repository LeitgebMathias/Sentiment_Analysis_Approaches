# Overview and Implementation of Sentiment Analysis Approaches

In sentiment analysis, there are two main approaches:
* Lexicon-based methods
* Machine learning-based methods

Lexicon-based methods use predefined word lists (lexicons) with sentiment values to analyze text tone. 
They are rule-based and don't need training data but can struggle with context and sarcasm. 
Machine learning methods, on the other hand, learn from large amounts of labeled data to make predictions. 
They are more accurate and adaptable but require extensive training data and computational resources.

This project implemented and tested both approaches.

For lexicon-based methods **Vader** and **Textblob** were used. 

For machine learning methods, various feature extraction techniques and models were implemented.

Tested Feature Extraction Methods:
* Bag of Words
* TF-IDF
* n-grams
* Word2Vec

Tested Models:
* Random Forest
* Support Vector Machine
* Naive Bayes

# Results
Here are the insights we draw from the results:
* **'TF-IDF'** & **'n-gram'** feature extraction slightly improve performance.
* On average, the **'Support Vector Machine'** model achieves the best results.
* On average, the **'TF-IDF'** feature extraction method achieves the best results.
* The best result is achieved with a **'Support Vector Machine'** and the **'Word2Vec'** feature extraction method.


|  | Random Forrest | Support Vector Machine [^1] | Naive Bayes | TextBlob | Vader |
|:--|:--|:--|:--| :--|:--|
| **Bag-Of-Words** | 84,37 % | 84,36 %  | 83,97 % | - | - |
| **TF-IDF** | 84,34 % | 86,27 % | 84,19 % | - | - |
| **n-grams** | 84,82 % | 84,43 % | 84,61 % | - | - |
| **Word2Vec** | 83,43 % | ***87,61 %*** | 74,71 % [^2] | - | - |
| **No Feature Extraction** | - | - | - | 77,04 % | 69,77 % |

[^1]: To Reduce Training Time, SVM only got trained with 5000 Datapoints
[^2]: Gaussian Naive Bayes used
