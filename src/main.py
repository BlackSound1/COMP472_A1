import utils
import matplotlib.pyplot as plt

"""Task 0 - Split training and evaluation data"""

# Read data
all_docs, all_labels = utils.read_documents('input/all_sentiment_shuffled.txt')
# Split point between training and evaluation
split_point = int(0.80*len(all_docs))

# Get training and evaluation data
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]

"""Task 1 - Plot label distribution"""
# Get label distribution
label_distribution = utils.get_label_distribution(all_labels)
print(label_distribution)

# Plot label distribution
plt.figure()
plt.bar(label_distribution.keys(), label_distribution.values(), 0.8)
for x, y in enumerate(label_distribution.values()):
    plt.text(x, y, str(y), horizontalalignment='center')
plt.show()
