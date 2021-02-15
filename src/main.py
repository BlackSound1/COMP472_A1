import utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

"""Task 0 - Split training and evaluation data"""
# Read data
X, y = utils.read_documents('../input/all_sentiment_shuffled.txt')

count_vect = CountVectorizer(analyzer=lambda x: x)
X_vectorized = count_vect.fit_transform(X)

# Split point between training and evaluation
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_vectorized, y, np.arange(len(X)), test_size=0.2)

"""Task 1 - Plot label distribution"""
# Get label distribution
label_distribution = utils.get_label_distribution(y_train)
print(label_distribution)

# Plot label distribution
plt.figure()
plt.bar(label_distribution.keys(), label_distribution.values(), 0.8)
for a, b in enumerate(label_distribution.values()):
    plt.text(a, b, str(b), horizontalalignment='center')
plt.title("Frequency of pos and neg")
plt.xlabel("sentiment")
plt.ylabel("frequency")    
plt.show()

"""Task 2 - Naive Bayes Classifier"""
multiNB = MultinomialNB()
multiNB.fit(X_train, y_train)
multiNB_pred = multiNB.predict(X_test)

"""Task 2 - Base-DT"""
baseDT = DecisionTreeClassifier(criterion='entropy', random_state=0)
baseDT.fit(X_train, y_train)
baseDT_pred = baseDT.predict(X_test)

"""Task 2 - Best-DT"""
# The following function Analyses the relationship between the ccp_alphas parameter and accuracy 
# for Decision Trees. It is not active when we run the code because it takes to long to complete. 
# It is present here to show where it would be used and for completeness only. 

# analyze_ccp_alpha(X_train, X_test, y_train, y_test)

bestDT = DecisionTreeClassifier(random_state=0)
bestDT = utils.get_best_model(X_train, y_train, bestDT, {"criterion": ["entropy", "gini"], "splitter": ["best", "random"]})
bestDT.fit(X_train, y_train)
bestDT_pred = bestDT.predict(X_test)

"""Task 3 - Generate output"""
labels = sorted(list(set(y)))
models = [('NaiveBayes', multiNB_pred),
          ('BaseDT', baseDT_pred),
          ('BestDT', bestDT_pred)]

utils.generate_output(indices_test, y_test, labels, models)
