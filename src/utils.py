
from re import compile, search
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report



def read_documents(path: str) -> (list, list):
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


def get_label_distribution(all_labels: list) -> dict:
    """Calculates distribution of labels.

    Args:
      all_labels (list): List of all labels.

    Returns:
      Dictionary keys (labels) and values (count)

      e.g. { 'health': 3, 'books': 45 }
    """
    labels = sorted(list(set(all_labels)))
    label_counts = {label: all_labels.count(label) for label in labels}
    return label_counts


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
    f.write('confusion matrix\n')
    matrix = confusion_matrix(y_test, y_pred)
    np.savetxt(f, matrix, fmt='%-5d')
    f.write('\n')

    # Calculate scores
    scores = []
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)

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
