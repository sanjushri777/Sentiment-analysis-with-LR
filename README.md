# Sentiment-analysis-with-NAIVE-BAYES-ALGORITHM
## NAME: SANJUSHRI A
## REGNO:212223040187
## AIM:
To develop a sentiment analysis model that classifies text as positive or negative using Natural Language Processing (NLP) and the Naive Bayes algorithm.

## THEORY:
Sentiment analysis is the process of determining the emotional tone behind words, commonly used to identify and extract opinions from text. This helps in understanding the sentiment of a sentence, paragraph, or document.

**Naive Bayes Classifier:**
Naive Bayes is a probabilistic classifier based on Bayes' Theorem, assuming independence among predictors. It is particularly suited for text classification due to its efficiency and simplicity.

**Steps Involved:**
Text Preprocessing: Clean and prepare raw text for modeling.

Tokenization and Vectorization: Convert text into numerical format (Bag-of-Words).

Training the Model: Use labeled data to train the Naive Bayes classifier.

Testing and Evaluation: Evaluate performance using accuracy and classification metrics.

## PROCEDURE:
STEP . 1 : Import Libraries: Use libraries such as nltk, sklearn, and numpy for NLP and ML tasks.

STEP . 2 : Prepare Dataset: Create a small dataset of sentences with labeled sentiments (0 = negative, 1 = positive).

STEP . 3 : Download Stopwords: Load stopwords using NLTK to filter out common words.

STEP . 4 : Preprocess Text:
               - Remove stopwords
               - Tokenize and vectorize using CountVectorizer

STEP . 5 : Split Dataset: Divide the dataset into training and test sets (e.g., 75% training, 25% testing).

STEP . 6 : Train Model: Use MultinomialNB to train the classifier on the training set.

STEP . 7 : Test Model: Evaluate the model using the test set.

STEP . 8 : Predict Sentiments: Test the model on new, unseen sentences.

STEP . 9 : Print Results: Display accuracy, classification report, and predictions.

## PROGRAM:
```Python  # Import necessary libraries
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')

# Sample dataset
data = [
    ("I love this product, it works great!", 1),
    ("This is the best purchase I have ever made.", 1),
    ("Absolutely fantastic service and amazing quality!", 1),
    ("I am very happy with my order, will buy again.", 1),
    ("This is a horrible experience.", 0),
    ("I hate this so much, it broke on the first day.", 0),
    ("Worst product I have ever used, total waste of money.", 0),
    ("I am disappointed with this product, it didn't work as expected.", 0)
]

# Split data into texts and labels
sentences = [pair[0] for pair in data]
labels = np.array([pair[1] for pair in data])

# Split into training and testing sets
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.25, random_state=42)

# Define stopwords
stop_words = stopwords.words('english')

# Vectorization
vectorizer = CountVectorizer(stop_words=stop_words)
X_train = vectorizer.fit_transform(sentences_train)
X_test = vectorizer.transform(sentences_test)

# Train the classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Predictions
y_pred = nb_classifier.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Test on new sentences
test_sentences = ["I am happy to comment!", "This is a terrible product."]
test_X = vectorizer.transform(test_sentences)
predictions = nb_classifier.predict(test_X)

# Output predictions
for sentence, sentiment in zip(test_sentences, predictions):
    print(f"Sentence: '{sentence}' => Sentiment: {'Positive' if sentiment == 1 else 'Negative'}")
```

## OUTPUT:
![Screenshot 2025-06-04 190413](https://github.com/user-attachments/assets/a6ee4973-42f8-4f39-886c-1ff4058344a2)

## RESULT:
The sentiment analysis model was successfully implemented using NLP and the Naive Bayes algorithm, achieving ~100% accuracy on the test data. It accurately classified both existing and new sentences, demonstrating effective sentiment detection.
