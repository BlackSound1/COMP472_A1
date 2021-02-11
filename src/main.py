import utils
import numpy as np
import pprint
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

"""Task 0 - Split training and evaluation data"""
# Read data
X, y = utils.read_documents('../input/all_sentiment_shuffled.txt')

count_vect = CountVectorizer(analyzer=lambda x: x)
X_vectorized = count_vect.fit_transform(X)

# Split point between training and evaluation
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_vectorized, y, np.arange(len(X)), test_size=0.2)

test_size = len(y_test)

"""Task 1 - Plot label distribution"""
# Get label distribution
label_distribution = utils.get_label_distribution(y)
print(label_distribution)

# Plot label distribution
plt.figure()
plt.bar(label_distribution.keys(), label_distribution.values(), 0.8)
for a, b in enumerate(label_distribution.values()):
    plt.text(a, b, str(b), horizontalalignment='center')
plt.show()

"""Task 2 - Naive Bayes Classifier"""
multiNB = MultinomialNB()
multiNB.fit(X_train, y_train)
multiNB_pred = multiNB.predict(X_test)

"""Task 2 - Base-DT"""
baseDT = DecisionTreeClassifier(criterion='entropy')
baseDT.fit(X_train, y_train)
baseDT_pred = baseDT.predict(X_test)

"""Task 2 - Best-DT"""
bestDT = DecisionTreeClassifier()
bestDT.fit(X_train, y_train)
bestDT_pred = bestDT.predict(X_test)

"""Task 3 - Generate output"""
labels = sorted(list(set(y)))
score_list = ['precision', 'recall', 'f1-score']
models = [('NaiveBayes', multiNB_pred), ('BaseDT', baseDT_pred), ('BestDT', bestDT_pred)]

for (name, y_pred) in models:
    f = open(f'../output/{name}-all_sentiment_shuffled.txt', 'w')

    # Write row and predicted class
    for i in range(test_size):
        f.write(f'{indices_test[i]}, {y_pred[i]}\n')
    f.write('\n')

    # Write confusion matrix
    f.write('confusion matrix\n')
    matrix = confusion_matrix(y_test, y_pred)
    np.savetxt(f, matrix, fmt='%-5d')
    f.write('\n')

    # Calculate scores
    scores = []
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    pprint.pprint(report)

    for label in labels:
        score_row = []
        for score in score_list:
            score_row.append(report[label][score])
        scores.append(score_row)

    # Transpose score matrix
    scores = np.array([list(row) for row in zip(*scores)])

    # Write matrix
    row_format = '{:<15}' + '{:<20}' * (len(labels))
    f.write(row_format.format("", *labels) + '\n')
    for type, row in zip(score_list, scores):
        f.write(row_format.format(type, *row))
        f.write('\n')
    f.write('\n')

    # Write accuracy
    acc = str(report['accuracy'])
    f.write("accuracy: " + acc)

    f.close()
