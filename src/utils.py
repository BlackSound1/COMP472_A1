
def read_documents(path: str) -> (list, list):
    """Reads data file.

    Args:
      path (str): Path to data file.

    Returns:
      Tuple of lists (words, labels) 
    """
    labels, docs = [], []
    with open(path, 'rb') as f:
        for line in f:
            words = line.strip().split()
            labels.append(words[1])
            docs.append(words[3:])
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

def listToString(list: list) -> str:
    """ Converts a list into a str

    Args:
      list (list): The list to convert

    Returns:
      str made from elements of the list.

      e.g. "This was a list" 
    """
    string = " "
    return (string.join(list))

def read_data(path: str) -> (list, list):
  """ Reads the data from the file, cleans it up, and gives 
      back the laberls and documents

    Args:
      path (str): The path of the file to read

    Returns:
      (list, list): 2 lists X and y, containing the documents and labels respectively

      e.g. ["A review of a product"], ["pos"] 
  """
  #all_text = []
  X, y = [], []
  with open (path, 'rt') as myfile:  
      for line in myfile:
          thisLine = line.split(" ")
          part_1 = thisLine[1]
          y.append(part_1)

          part_2 = listToString(thisLine[3:])
          X.append(part_2)

          #all_text.append(part_1 + " " + part_2) 
  return X, y        