import utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

def main():
    """Task 0 - Split training and evaluation data"""
    # Read data
    X, y = utils.read_documents('../input/all_sentiment_shuffled.txt')

    count_vect = CountVectorizer(analyzer=lambda x: x)
    X_vectorized = count_vect.fit_transform(X)

    # Split point between training and evaluation
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_vectorized, y, np.arange(1, len(X)+1), test_size=0.2)

    """Task 1 - Plot label distribution"""
    # Get label distribution
    train_label_distribution = utils.get_label_distribution(y_train, save_plot_as='training_label_distribution')
    test_label_distribution = utils.get_label_distribution(y_test, save_plot_as='test_label_distribution')

    """Task 2 - Naive Bayes Classifier"""
    multiNB = MultinomialNB()
    multiNB.fit(X_train, y_train)
    multiNB_pred = multiNB.predict(X_test)
    multiNB_pred_prob = multiNB.predict_proba(X_test)
    utils.generate_wrong_pred("NaiveBayes", indices_test, y_test, multiNB_pred, multiNB.classes_, multiNB_pred_prob)
    utils.save_graphical_confusion_matrix(multiNB, X_test, y_test, "NB")

    """Task 2 - Base-DT"""
    baseDT = DecisionTreeClassifier(criterion='entropy', random_state=0)
    baseDT.fit(X_train, y_train)
    baseDT_pred = baseDT.predict(X_test)
    utils.save_graphical_confusion_matrix(baseDT, X_test, y_test, "DT-base")

    """Task 2 - Best-DT"""
    # The following function Analyses the relationship between the ccp_alphas parameter and accuracy 
    # for Decision Trees. It is not active when we run the code because it takes to long to complete. 
    # It is present here to show where it would be used and for completeness only. 

    # analyze_ccp_alpha(X_train, X_test, y_train, y_test)

    bestDT = DecisionTreeClassifier(random_state=0)
    bestDT = utils.get_best_model(X_train, y_train, bestDT, {"criterion": ["entropy", "gini"], "splitter": ["best", "random"]})
    bestDT.fit(X_train, y_train)
    bestDT_pred = bestDT.predict(X_test)
    utils.save_graphical_confusion_matrix(bestDT, X_test, y_test, "DT-best")

    """Task 3 - Generate output"""
    labels = sorted(set(y))
    models = [('NaiveBayes', multiNB_pred),
            ('BaseDT', baseDT_pred),
            ('BestDT', bestDT_pred)]

    utils.generate_output(indices_test, y_test, labels, models)

if __name__ == "__main__":
    main()