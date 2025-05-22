# Sentiment Analysis of Movie Reviews using Neural Networks

This project implements and compares three different machine learning models for sentiment analysis on the IMDB Movie Reviews dataset. The models include Logistic Regression, Artificial Neural Network (ANN), and Long Short-Term Memory (LSTM) network.

## 📋 Overview

The project performs binary sentiment classification (positive/negative) on movie reviews using three different approaches:
1. **Logistic Regression**: A traditional machine learning approach using bag-of-words representation
2. **Artificial Neural Network (ANN)**: A simple neural network with embedding and dense layers
3. **Long Short-Term Memory (LSTM)**: A recurrent neural network that captures sequential patterns in text

## 🎯 Features

- Data preprocessing and visualization
- Implementation of three different models
- Early stopping to prevent overfitting
- Model checkpointing to save best weights
- Comprehensive evaluation metrics
- Visualization of results including:
  - Word cloud of training data
  - Confusion matrices
  - Training history plots
  - Model comparison bar chart

## 📊 Model Architecture

### Logistic Regression
- Bag-of-words representation
- Maximum features: 5000
- Binary classification

### ANN Model
- Embedding Layer (32 dimensions)
- Flatten Layer
- Dense Layer (64 units, ReLU activation)
- Dropout (0.5)
- Output Layer (1 unit, Sigmoid activation)

### LSTM Model
- Embedding Layer (32 dimensions)
- LSTM Layer (64 units, return sequences)
- Dropout (0.5)
- LSTM Layer (32 units)
- Dropout (0.5)
- Output Layer (1 unit, Sigmoid activation)

## 🛠️ Requirements

Install the required packages using:
```bash
pip install -r requirements.txt
```

Required packages:
- numpy>=1.19.2
- tensorflow>=2.5.0
- scikit-learn>=0.24.2
- matplotlib>=3.4.2
- seaborn>=0.11.1
- pandas>=1.3.0
- wordcloud>=1.8.1

## 🚀 Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sentiment-analysis-imdb.git
cd sentiment-analysis-imdb
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the analysis:
```bash
python imdb_sentiment_analysis.py
```

## 📈 Output

The script generates several visualization files:
- `wordcloud.png`: Word cloud of the training data
- `lr_confusion_matrix.png`: Confusion matrix for Logistic Regression
- `ann_confusion_matrix.png`: Confusion matrix for ANN
- `ann_training_history.png`: Training history for ANN
- `lstm_confusion_matrix.png`: Confusion matrix for LSTM
- `lstm_training_history.png`: Training history for LSTM
- `model_comparison.png`: Bar chart comparing accuracy scores

The best models are saved in the `models` directory:
- `models/ann_best_model.h5`: Best ANN model weights
- `models/lstm_best_model.h5`: Best LSTM model weights

## 🎯 Training Configuration

- Dataset: IMDB Movie Reviews (50,000 reviews)
- Vocabulary size: 5,000 most frequent words
- Sequence length: 500 tokens
- Batch size: 64
- Validation split: 0.2
- Early stopping patience: 3 epochs
- Maximum epochs: 20 (with early stopping)

## 📊 Model Performance

The models are evaluated using:
- Accuracy score
- Confusion matrix
- Training/validation loss and accuracy curves

## 🔍 Implementation Details

### Data Preprocessing
- Text tokenization
- Sequence padding
- Vocabulary size limitation
- Train-test split

### Model Training
- Early stopping to prevent overfitting
- Model checkpointing
- Learning rate: 0.001
- Optimizer: Adam
- Loss function: Binary Cross-Entropy

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Vibhu - Initial work

## 🙏 Acknowledgments

- IMDB dataset provided by Keras
- TensorFlow and Keras for deep learning implementation
- Scikit-learn for traditional machine learning implementation

## 📊 Data Analysis

The project includes a comprehensive data analysis script (`data_analysis.py`) that provides insights into the IMDB dataset before model training. The analysis includes:

### Visualizations
- **Sequence Length Analysis**: Distribution of review lengths in training and testing sets
- **Word Frequency Analysis**: Top 20 most frequent words in the dataset
- **Class Distribution**: Distribution of positive and negative reviews
- **Word Cloud**: Visual representation of common words in reviews
- **Sentiment Length Distribution**: Review length distribution by sentiment

### Statistics
- Average and median review lengths
- Word frequency statistics
- Class distribution percentages
- Review length statistics by sentiment

To run the analysis:
```bash
python data_analysis.py
```

This will generate the following visualization files:
- `sequence_length_analysis.png`: Histogram and box plot of review lengths
- `word_frequency.png`: Bar chart of most frequent words
- `class_distribution.png`: Pie charts of class distribution
- `wordcloud.png`: Word cloud of common words
- `sentiment_length_distribution.png`: Violin plot of review lengths by sentiment

![lstm_training_history](https://github.com/user-attachments/assets/bc931b57-7ac0-4d6c-8520-3b1fb3568c80)
![model_comparison](https://github.com/user-attachments/assets/ac562c09-6313-4afb-abf7-958d87845071)
![wordcloud](https://github.com/user-attachments/assets/8a361b3e-3334-4609-86f3-19f2a98a4a7b)
![class_distribution](https://github.com/user-attachments/assets/2bab24cd-1e54-45cf-8fbc-53cd7b6f4671)
![review_length_analysis](https://github.com/user-attachments/assets/875ec21f-061d-46c1-89bc-24279380c7a2)
![word_frequency](https://github.com/user-attachments/assets/29eaf0d1-3452-4a1a-bf8e-9ab4b64214c5)
![ann_confusion_matrix](https://github.com/user-attachments/assets/2e340a93-791b-451c-b3c9-3888e3d304b4)
![ann_training_history](https://github.com/user-attachments/assets/8c961aaf-0226-424c-9f11-39ecd1d83053)
![lr_confusion_matrix](https://github.com/user-attachments/assets/f7de7ddc-5114-4618-afcf-21da952f2db7)
![lstm_confusion_matrix](https://github.com/user-attachments/assets/e252f027-d86d-4926-8545-5441480a0510)
