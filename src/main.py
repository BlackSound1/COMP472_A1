import utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

"""Task 0 - Split training and evaluation data"""
# Read data
X, y = utils.read_documents('../input/all_sentiment_shuffled.txt')

# Split point between training and evaluation
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, np.arange(len(X)), test_size=0.2)

# Transform documents to matrix
vectorizer = CountVectorizer()
vectorizer.fit(X)
mat_train = vectorizer.transform(X_train)
mat_test = vectorizer.transform(X_test)

test_size = len(X_test)

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
mnb = MultinomialNB()
mnb.fit(mat_train, y_train)
mnb_pred = mnb.predict(mat_test)

"""Task 2 - Base-DT"""
bdt = DecisionTreeClassifier(criterion='entropy')
bdt.fit(mat_train, y_train)
bdt_pred = bdt.predict(mat_test)

"""Task 2 - Best-DT"""
btdt = DecisionTreeClassifier()  # Not sure what params go here...
btdt.fit(mat_train, y_train)
btdt_pred = btdt.predict(mat_test)

"""Task 3 - Generate output"""
labels = sorted(list(set(y)))
score_list = ['precision', 'recall', 'f1-measure']
models = [('NaiveBayes', mnb, mnb_pred), ('BaseDT', bdt, bdt_pred), ('BestDT', btdt, btdt_pred)]

for (name, model, y_pred) in models:
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

    # Calculate precision, recall, f1 scores
    scores = []
    for label in labels:
        pre = precision_score(y_test, y_pred, pos_label=label)
        rec = recall_score(y_test, y_pred, pos_label=label)
        f1 = f1_score(y_test, y_pred, pos_label=label)
        scores.append([pre, rec, f1])

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
    f.write("accuracy: " + str(accuracy_score(y_test, y_pred)))

    f.close()
