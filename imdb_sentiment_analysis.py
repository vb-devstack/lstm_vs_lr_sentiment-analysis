import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# Load and preprocess the IMDB dataset
def load_and_preprocess_data(max_words=5000, maxlen=500):
    print("Loading IMDB dataset...")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
    
    # Pad sequences
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)
    
    return (x_train, y_train), (x_test, y_test)

# Create word cloud visualization
def create_word_cloud(x_train, word_index):
    # Create reverse word index
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    # Convert sequences back to text
    texts = []
    for sequence in x_train[:1000]:  # Using first 1000 reviews for visualization
        text = ' '.join([reverse_word_index.get(i - 3, '?') for i in sequence])
        texts.append(text)
    
    # Create and plot word cloud
    text = ' '.join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of IMDB Reviews')
    plt.savefig('wordcloud.png')
    plt.close()

# Logistic Regression Model
def train_logistic_regression(x_train, y_train, x_test, y_test):
    print("\nTraining Logistic Regression model...")
    # Convert sequences to bag of words
    vectorizer = CountVectorizer(max_features=5000)
    x_train_bow = vectorizer.fit_transform([' '.join(map(str, seq)) for seq in x_train])
    x_test_bow = vectorizer.transform([' '.join(map(str, seq)) for seq in x_test])
    
    # Train model
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(x_train_bow, y_train)
    
    # Evaluate
    y_pred = lr_model.predict(x_test_bow)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Logistic Regression Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('lr_confusion_matrix.png')
    plt.close()
    
    return accuracy, y_pred

# ANN Model
def train_ann(x_train, y_train, x_test, y_test, max_words=5000):
    print("\nTraining ANN model...")
    model = Sequential([
        Embedding(max_words, 32, input_length=500),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    # Train model
    history = model.fit(x_train, y_train,
                       batch_size=64,
                       epochs=10,
                       validation_split=0.2,
                       verbose=1)
    
    # Evaluate
    _, accuracy = model.evaluate(x_test, y_test)
    print(f"ANN Accuracy: {accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('ANN Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('ANN Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ann_training_history.png')
    plt.close()
    
    # Plot confusion matrix
    y_pred = (model.predict(x_test) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('ANN Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('ann_confusion_matrix.png')
    plt.close()
    
    return accuracy, y_pred

# LSTM Model
def train_lstm(x_train, y_train, x_test, y_test, max_words=5000):
    print("\nTraining LSTM model...")
    model = Sequential([
        Embedding(max_words, 32, input_length=500),
        LSTM(64, return_sequences=True),
        Dropout(0.5),
        LSTM(32),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    # Train model
    history = model.fit(x_train, y_train,
                       batch_size=64,
                       epochs=10,
                       validation_split=0.2,
                       verbose=1)
    
    # Evaluate
    _, accuracy = model.evaluate(x_test, y_test)
    print(f"LSTM Accuracy: {accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('LSTM Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('lstm_training_history.png')
    plt.close()
    
    # Plot confusion matrix
    y_pred = (model.predict(x_test) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('LSTM Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('lstm_confusion_matrix.png')
    plt.close()
    
    return accuracy, y_pred

def plot_model_comparison(lr_acc, ann_acc, lstm_acc):
    models = ['Logistic Regression', 'ANN', 'LSTM']
    accuracies = [lr_acc, ann_acc, lstm_acc]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies)
    
    # Add accuracy values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.title('Model Comparison - Accuracy Scores')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig('model_comparison.png')
    plt.close()

def main():
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Get word index for visualization
    word_index = imdb.get_word_index()
    
    # Create word cloud visualization
    create_word_cloud(x_train, word_index)
    
    # Train and evaluate models
    lr_acc, _ = train_logistic_regression(x_train, y_train, x_test, y_test)
    ann_acc, _ = train_ann(x_train, y_train, x_test, y_test)
    lstm_acc, _ = train_lstm(x_train, y_train, x_test, y_test)
    
    # Plot model comparison
    plot_model_comparison(lr_acc, ann_acc, lstm_acc)
    
    print("\nTraining and evaluation completed!")
    print("Visualizations have been saved as PNG files.")

if __name__ == "__main__":
    main() 