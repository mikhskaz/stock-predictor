from config import NEWS_API_KEY

import requests
import csv
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DEPTH = 132
ESTIMATORS = 100 # Number of trees in the forest
STOCK = 'AAPL'
DATA_PATH = 'stock-predictor/data/sentiments.csv'

vectorizer = TfidfVectorizer()

def get_news(stock: str, api_key: str, count=10) -> list[dict]:
    """Get the news for a given stock from the News API.

    Most recently published count articles are returned.

    :param stock: The stock to get news for.
    :param api_key: The News API key.
    :param count: The number of articles to get.
    :return: The news articles.
    """
    query = f"{stock} stock OR {stock} OR {stock} earnings OR {stock} market news OR {stock} stock price OR {stock} stock news"
    url = f'https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize={count}&apiKey={api_key}'
    response = requests.get(url)
    return response.json()['articles']

def load_data(data_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the data from a CSV file and return train, validation, and test sets.

    Pre process it with a vectorizer.

    :param data_path: The path to the CSV file.
    :return: Data in numpy arrays in three sets: train, validation, and test.
    """
    global vectorizer

    X_pre =[]
    X = [] # Articles
    y = [] # Sentiments

    with open(data_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            X_pre.append(clean_article(row[1]))
            y.append(row[0])

    X = vectorizer.fit_transform(X_pre)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test

def clean_article(article: str) -> str:
    """Clean the article by removing special characters and converting to lowercase.

    :param article: The article to clean.
    :return: The cleaned article.
    """
    return ''.join(e for e in article if e.isalnum() or e.isspace()).lower()

def train_classifier(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, depth=None, estimators=None) -> RandomForestClassifier:
    """Train a classifier on the given data.

    Vary the depth of the tree to 5 levels.

    :param X_train: The training data.
    :param y_train: The training labels.
    :return: The trained classifier.
    """
    max_depths = [DEPTH] if depth is None else [depth]
    n_estimators = [ESTIMATORS] if estimators is None else [estimators]

    accuracies = []

    best_accuracy = float('-inf')
    best_clf = None
    best_estimator = None

    for depth in max_depths:
        for estimator in n_estimators:
            clf = RandomForestClassifier(n_estimators=estimator, max_depth=depth, random_state=42)
            # clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
            clf.fit(X_train, y_train)

            accuracy = clf.score(X_val, y_val)
            accuracies.append(accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_clf = clf
                best_estimator = estimator
    
    print(f'Best accuracy: {best_accuracy} at depth {best_clf.max_depth} at {best_estimator} estimators')

    if depth is None and estimators is None: # Plot the graph if needed
        plt.plot(max_depths, accuracies, label='100 estimators')
        plt.xlabel('Max Depth')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Max Depth')
        plt.show()

    return best_clf
    

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(DATA_PATH)
    # depth_list = list(range(1, 200))

    clf = train_classifier(X_train, y_train, X_val, y_val, estimators=100)

    print("Validation score:", clf.score(X_val, y_val))

    print("Test score:", clf.score(X_test, y_test))


    news = get_news(STOCK, NEWS_API_KEY, count=100)

    sentiments = ['negative', 'neutral', 'positive']
    sentiments_probabilities = []

    for article in news:
        sentiments_probabilities.append(clf.predict_proba(vectorizer.transform([clean_article(article['content'])]))[0])
    
    thoughts = np.mean(sentiments_probabilities, axis=0)
    print(f"Thoughts on {STOCK}:")
    # print(f"Positive: {thoughts[0]}")
    # print(f"Negative: {thoughts[1]}")
    # print(f"Neutral: {thoughts[2]}")
    second_most_sentiment_index = np.argsort(thoughts)[-2]
    print(f"Overall Sentiment: {sentiments[np.argmax(thoughts)]} {sentiments[second_most_sentiment_index]} at {round(((thoughts[np.argmax(thoughts)] + thoughts[second_most_sentiment_index])) * 100, 1)}% confidence")

    