
from re import compile, search
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.model_selection import KFold, GridSearchCV
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier


def read_documents(path: str):
    """Reads data file.

    Args:
      path (str): Path to data file.

    Returns:
      Tuple of lists (words, labels) 
    """
    labels, docs = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            labels.append(words[1])
            docs.append(words[3:])
    return sanitize_text(docs), labels


def get_label_distribution(all_labels: list, save_plot_as: str=None) -> dict:
    """Calculates distribution of labels.

    Args:
      all_labels (list): List of all labels.

    Returns:
      Dictionary keys (labels) and values (count)

      e.g. { 'health': 3, 'books': 45 }
    """
    labels = sorted(set(all_labels))
    label_counts = {label: all_labels.count(label) for label in labels}

    if save_plot_as:
        plt.figure()
        plt.bar(label_counts.keys(), label_counts.values(), 0.8)
        for a, b in enumerate(label_counts.values()):
            plt.text(a, b, str(b), horizontalalignment='center')
        plt.title("Frequency of pos and neg")
        plt.xlabel("sentiment")
        plt.ylabel("frequency")    
        plt.savefig(f'../output/{save_plot_as}.png')

    return label_counts


def get_best_model(X, y, estimator, estimator_test_params):
    """Finds the best estimator by exhaustively searching the best parameters using cross-validation.

    Args:
      X (list): Training input.
      y (list): Target input.
      estimator (Object): Estimator.
      estimator_test_params (dict): Dictionary of parameters to test on estimator.

    Returns:
      The best estimator based on the testing parameters.
    """
    cv = KFold(n_splits=3)
    grid_search = GridSearchCV(estimator, estimator_test_params, cv=cv)
    grid_search.fit(X, y)
    return grid_search.best_estimator_


def generate_output(indices_test: list, y_test: list, labels: list, models: list):
    """Generates output file for the given models.

    Args:
      indices_test (list): List of test indices.
      y_test (list): List of test labels.
      labels (list): List of unique labels.
      models (list): List of models containing their name and label predictions.
    """
    test_size = len(y_test)
    score_list = ['precision', 'recall', 'f1-score']

    for (name, y_pred) in models:
        f = open(f'../output/{name}-all_sentiment_shuffled.txt', 'w')
        
        # Write row and predicted class
        for i in range(test_size):
            f.write(f'{indices_test[i]}, {y_pred[i]}\n')
        f.write('\n')

        # Write confusion matrix
        f.write(f'confusion matrix (row=truth, column=predicted, labels={labels})\n')
        matrix = confusion_matrix(y_test, y_pred)
        np.savetxt(f, matrix, fmt='%-5d')
        f.write('\n')

        # Calculate scores
        scores = []
        report = classification_report(y_test, y_pred, digits=4, output_dict=True)

        for score in score_list:
            score_row = []
            for label in labels:
                score_row.append(report[label][score])
            scores.append(score_row)

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


def sanitize_text(lst: list) -> list:
    """ Sanitizes input by removing numbers, special characters, and useless words

        Args:
          lst (list): The list of text to convert

        Returns:
          list: the new list without special characters or numbers

          e.g. ["hello", "1234", "#%$#"] -> ["hello"]
    """
    to_return = []
    regex = \
        compile(r'[\d!?,.()\]\[#$%^\"&*\'+=\-_\\/|]+|\b(th(e|ey|is|at|ere|eir)|an|a|it|to|and|is|for|on|of|i|my|yo(u|ur))\b')

    for sublist in lst:
        filtered_words = [word for word in sublist if not regex.search(word)]
        to_return.append(filtered_words)

    return to_return


def analyze_ccp_alpha(X_train, X_test, y_train, y_test):
    """ Analyses the relationship between the ccp_alphas parameter and accuracy for Decision Trees.
        This function was created to perform analysis on the ccp_alpha parameter to see if tuning it 
        improved the decision tree. It is not used in the normal running of the code as it takes much
        too long to complete. It is present for completeness only.
        Code derived from: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html

          Args:
            X_train (array): The training documents
            X_test (array): the testing documents
            y_train (array): the training labels
            y_test (array): the testing labels
    """
    classifier = DecisionTreeClassifier()

    path = classifier.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas

    classifiers = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        classifiers.append(clf)

    train_scores = [clf.score(X_train, y_train) for clf in classifiers]
    test_scores = [clf.score(X_test, y_test) for clf in classifiers]

    ax = plt.subplots()
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker='o', label="train",
            drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o', label="test",
            drawstyle="steps-post")
    ax.legend()
    plt.show()


def generate_wrong_pred(name, rows, y_test, y_pred, classes, probs):
    wrong_pred = []
    f = open(f'../output/{name}-all_sentiment_shuffled-error.txt', 'w')

    for row, truth, pred, prob in zip(rows, y_test, y_pred, probs):
        if truth != pred:
            wrong_pred.append((row, truth, pred, prob[list(classes).index(pred)]))
            f.write(f'{row}, {pred}, {prob[list(classes).index(pred)]}\n')

    f.close()
    return wrong_pred

def save_graphical_confusion_matrix(model, X_test, y_test, type: str):
  matrix = plot_confusion_matrix(model, X_test, y_test, labels=["pos", "neg"])

  if type == "NB":
    matrix.ax_.set_title("Naive Bayes Confusion Matrix")
    plt.savefig("../output/NB-confusion-matrix.png")
  elif type == "DT-base":
    matrix.ax_.set_title("DT-base Confusion Matrix")
    plt.savefig("../output/DT-base-confusion-matrix.png")  
  elif type == "DT-best":
    matrix.ax_.set_title("DT-best Confusion Matrix")
    plt.savefig("../output/DT-best-confusion-matrix.png")  
  