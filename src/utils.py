from re import compile, search


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

def list_to_string(list: list) -> str:
    """ Converts a list into a str

    Args:
      list (list): The list to convert

    Returns:
      str made from elements of the list.

      e.g. "This was a list" 
    """
    string = " "
    return (string.join(list))


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
