# 🐦 Twitter Sentiment Analysis

## 📌 Project Overview

This project focuses on performing sentiment analysis on Twitter data. It involves:

- Web scraping tweets 🐦
- Preprocessing the text data
- Extracting features
- Training and evaluating machine learning models (ANN and SVM)

The sentiment of tweets is classified as **positive**, **neutral**, or **negative**.

---

## 🧠 Features ✨

- **Web Scraping**: Collects tweets from Twitter using Selenium. 🕸️
- **Data Preprocessing**:
  - Removes punctuation
  - Converts text to lowercase
  - Removes stopwords
  - Applies stemming (Porter Stemmer)
  - Handles missing values
- **Sentiment Labeling**: Uses VADER to assign sentiment labels 😊😐😠
- **Feature Extraction**:
  - Bigrams (CountVectorizer)
  - TF-IDF (TfidfVectorizer)
- **Machine Learning Models**:
  - Artificial Neural Network (ANN) 🧠
  - Support Vector Machine (SVM)
- **Model Evaluation**:
  - Accuracy, precision, recall, F1-score
  - Confusion matrix 📈
- **Visualization**:
  - Word clouds for positive and negative tweets ☁️

---

## 🧪 Methodology

### Web Scraping
- Uses Selenium to log in to Twitter and scrape tweets
- Automatically scrolls to collect ~1100+ tweets
- Saves tweets to `tweets.csv`

### Preprocessing and Wrangling
- Loads data into a Pandas DataFrame
- Uses VADER to assign sentiment (0: Negative, 1: Neutral, 2: Positive)
- Drops missing labels
- Applies preprocessing:
  - Remove newlines, punctuation, stopwords
  - Convert to lowercase
  - Filter out URLs
  - Apply Porter Stemming

### Feature Extraction
- Converts cleaned text to string
- Extracts features using CountVectorizer (bigrams) and TfidfVectorizer
- Splits into 80% training and 20% test sets
- One-hot encodes labels for ANN model

### Modeling
- **ANN (Keras)**:
  - Multiple dense layers with ReLU, softmax output
  - Adam optimizer, categorical crossentropy
  - Trained for 10 epochs, batch size 32
- **SVM**:
  - Trained using SVC separately on both feature sets

---

## 📊 Results

### 🔢 ANN Model Performance

#### Bigrams:
- **Accuracy**: ~73–74%
- **Classification Report**:
  - Negative (0): Precision: 0.73, Recall: 0.52, F1-score: 0.61
  - Neutral (1): Precision: 0.68, Recall: 0.81, F1-score: 0.74
  - Positive (2): Precision: 0.87, Recall: 0.77, F1-score: 0.82
- **Confusion Matrix**:
  ```
  [[ 38   9   5]
   [ 11 113  15]
   [  2  21  74]]
  ```

#### TF-IDF:
- **Accuracy**: ~73–74%
- **Same performance metrics as Bigrams**

### 🧮 SVM Model Performance

#### Bigrams:
- **Accuracy**: 0.7365
- **Same classification report and confusion matrix as ANN**

#### TF-IDF:
- **Accuracy**: 0.7365
- **Same classification report and confusion matrix as ANN**

---

## 🖼️ Visualizations

- **Sentiment Distribution**: Pie chart of negative, neutral, positive
- **Word Clouds**:
  - Positive tweets ☁️
  - Negative tweets ☁️

---

## 💻 Installation

```bash
git clone <repository_url>
cd twitter-sentiment-analysis
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download NLTK resources manually if needed:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
```

---

## ▶️ Usage

1. **Configure Selenium**:
   - Download ChromeDriver for your browser version
   - Update `driver_path` in the notebook
   - Set your Twitter email in the `email` variable
   - Script will prompt for Twitter password

2. **Run Notebook**:
   ```bash
   jupyter notebook Twitter_Sentiment_Analysis.ipynb
   ```
   - Execute all cells in order
   - Chrome will open and scrape tweets

3. **Analyze Results**:
   - Check output metrics: accuracy, precision, recall, F1-score
   - View confusion matrices and word clouds

---

## 📄 Requirements

```txt
pandas
nltk
scikit-learn
matplotlib
seaborn
selenium
keras
tensorflow
```

