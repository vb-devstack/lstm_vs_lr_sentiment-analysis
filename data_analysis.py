







import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

print("\n=== IMDB Dataset Analysis ===")
print(f"Training samples: {len(x_train)}")
print(f"Testing samples: {len(x_test)}")

# 1. Class Distribution Analysis
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.pie([sum(y_train == 0), sum(y_train == 1)], 
        labels=['Negative', 'Positive'], 
        autopct='%1.1f%%')
plt.title('Training Set Class Distribution')

plt.subplot(1, 2, 2)
plt.pie([sum(y_test == 0), sum(y_test == 1)], 
        labels=['Negative', 'Positive'], 
        autopct='%1.1f%%')
plt.title('Testing Set Class Distribution')
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.close()

print("\nClass Distribution:")
print(f"Training set - Positive: {sum(y_train == 1)}, Negative: {sum(y_train == 0)}")
print(f"Testing set - Positive: {sum(y_test == 1)}, Negative: {sum(y_test == 0)}")

# 2. Review Length Analysis
train_lengths = [len(seq) for seq in x_train]
test_lengths = [len(seq) for seq in x_test]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(train_lengths, bins=50, alpha=0.5, label='Training')
plt.hist(test_lengths, bins=50, alpha=0.5, label='Testing')
plt.title('Distribution of Review Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
plt.boxplot([train_lengths, test_lengths], labels=['Training', 'Testing'])
plt.title('Box Plot of Review Lengths')
plt.ylabel('Number of Words')
plt.tight_layout()
plt.savefig('review_length_analysis.png')
plt.close()

print("\nReview Length Statistics:")
print(f"Training set - Mean: {np.mean(train_lengths):.2f}, Median: {np.median(train_lengths):.2f}")
print(f"Testing set - Mean: {np.mean(test_lengths):.2f}, Median: {np.median(test_lengths):.2f}")

# 3. Word Frequency Analysis
all_words = [word for seq in x_train for word in seq]
word_freq = Counter(all_words)
top_words = word_freq.most_common(20)
words, freqs = zip(*top_words)
word_names = [reverse_word_index.get(word, '?') for word in words]

plt.figure(figsize=(12, 6))
plt.bar(word_names, freqs)
plt.title('Top 20 Most Frequent Words')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('word_frequency.png')
plt.close()

# 4. Word Cloud
texts = []
for sequence in x_train[:1000]:  # Using first 1000 reviews
    text = ' '.join([reverse_word_index.get(i - 3, '?') for i in sequence])
    texts.append(text)

text = ' '.join(texts)
wordcloud = WordCloud(width=800, height=400, 
                     background_color='white',
                     max_words=100).generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of IMDB Reviews')
plt.savefig('wordcloud.png')
plt.close()

# 5. Review Length by Sentiment
pos_lengths = [len(seq) for seq, label in zip(x_train, y_train) if label == 1]
neg_lengths = [len(seq) for seq, label in zip(x_train, y_train) if label == 0]

plt.figure(figsize=(10, 6))
plt.boxplot([pos_lengths, neg_lengths], 
            labels=['Positive', 'Negative'])
plt.title('Review Length by Sentiment')
plt.ylabel('Number of Words')
plt.savefig('sentiment_length.png')
plt.close()

print("\nReview Length by Sentiment:")
print(f"Positive reviews - Mean: {np.mean(pos_lengths):.2f}, Median: {np.median(pos_lengths):.2f}")
print(f"Negative reviews - Mean: {np.mean(neg_lengths):.2f}, Median: {np.median(neg_lengths):.2f}")

print("\nAnalysis complete! Generated visualization files:")
print("- class_distribution.png")
print("- review_length_analysis.png")
print("- word_frequency.png")
print("- wordcloud.png")
print("- sentiment_length.png") 