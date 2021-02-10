def read_documents(path: str) -> (list, list):
    """Reads data file.

    Args:
      path (str): Path to data file.

    Returns:
      Tuple of lists (words, labels) 
    """
    labels, docs = [], []
    with open(path, 'r') as f:
        for line in f:
            words = line.strip().split()
            labels.append(words[1])
            docs.append(' '.join(words[3:]))
    return docs, labels


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
