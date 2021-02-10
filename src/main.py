import utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

"""Task 0 - Split training and evaluation data"""
# Read data
X, y = utils.read_documents('input/all_sentiment_shuffled.txt')  

# Split point between training and evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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
