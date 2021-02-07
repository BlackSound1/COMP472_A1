import utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer


"""Task 0 - Split training and evaluation data"""

# Read data
all_docs, all_labels = utils.read_documents('input/all_sentiment_shuffled.txt')
# Split point between training and evaluation
split_point = int(0.80*len(all_docs))

# Get training and evaluation data
count_vect = CountVectorizer(tokenizer=lambda txt: txt.split())
all_docs_as_strings = list(' '.join(doc) for doc in all_docs)
all_docs_vect = count_vect.fit_transform(all_docs_as_strings)

train_docs = all_docs_vect[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs_vect[split_point:]
eval_labels = all_labels[split_point:]

"""Task 1 - Plot label distribution"""
# # Get label distribution
# label_distribution = utils.get_label_distribution(all_labels)
# print(label_distribution)

# # Plot label distribution
# plt.figure()
# plt.bar(label_distribution.keys(), label_distribution.values(), 0.8)
# for x, y in enumerate(label_distribution.values()):
#     plt.text(x, y, str(y), horizontalalignment='center')
# plt.show()

"""Task 2 - Naive Bayes"""
multiNB = MultinomialNB()
multiNB.fit(train_docs, train_labels)

y_pred = multiNB.predict(eval_docs)
print(metrics.accuracy_score(eval_labels, y_pred))
