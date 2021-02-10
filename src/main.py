import utils
import numpy as np
import pprint
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

"""Task 0 - Split training and evaluation data"""
# Read data
X, y = utils.read_documents('../input/all_sentiment_shuffled.txt')  
count_vect = CountVectorizer(analyzer=lambda x: x)
X_vectorized = count_vect.fit_transform(X)

# Split point between training and evaluation
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_vectorized, y, np.arange(len(X)), test_size=0.2)

"""Task 1 - Plot label distribution"""
# Get label distribution
label_distribution = utils.get_label_distribution(y)
print(label_distribution)

# Plot label distribution
plt.figure()
plt.bar(label_distribution.keys(), label_distribution.values(), 0.8)
for x, y in enumerate(label_distribution.values()):
    plt.text(x, y, str(y), horizontalalignment='center')
plt.show()

"""Task 2 - Naive Bayes"""
multiNB = MultinomialNB()
multiNB.fit(X_train, y_train)
y_pred = multiNB.predict(X_test)
pprint.pprint(metrics.classification_report(y_test, y_pred, digits=4, output_dict=True))
