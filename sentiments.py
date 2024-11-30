from config import NEWS_API_KEY

import requests
import csv
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

DEPTH = 132 # Best for sentiments.csv
DEPTH_ALTERNATE = 185 # Best for sentiments_alternate.csv
ESTIMATORS = 100 # Number of trees in the forest
STOCK = 'TSLA'
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

def load_data(data_path: str, flip=False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the data from a CSV file and return train, validation, and test sets.

    Pre process it with a vectorizer.

    :param data_path: The path to the CSV file.
    :return: Data in numpy arrays in three sets: train, validation, and test.
    """
    global vectorizer
    text = not flip
    sentiment = flip
    X_pre =[]
    X = [] # Articles
    y = [] # Sentiments

    with open(data_path, 'r', errors='replace') as file:
        reader = csv.reader(file)
        for row in reader:
            if 'https' in row[text]:
                continue
            X_pre.append(clean_article(row[text]))
            y.append(row[sentiment])

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
    return (''.join(e for e in article if e.isalnum() or e.isspace()).lower()).strip()

def train_classifier(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, depth=None, estimators=None) -> RandomForestClassifier:
    """Train a classifier on the given data.

    Vary the depth of the tree to 5 levels.

    :param X_train: The training data.
    :param y_train: The training labels.
    :return: The trained classifier.
    """
    max_depths = [DEPTH] if depth is None else depth
    n_estimators = [ESTIMATORS] if estimators is None else [estimators]

    accuracies = []

    best_accuracy = float('-inf')
    best_clf = None
    best_estimator = None

    for depth in max_depths:
        for estimator in n_estimators:
            # clf = RandomForestClassifier(n_estimators=estimator, max_depth=depth, random_state=42)
            clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
            clf.fit(X_train, y_train)

            accuracy = clf.score(X_val, y_val)
            accuracies.append(accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_clf = clf
                best_estimator = estimator
        print(f'Accuracy: {accuracy} at depth {depth}')
    
    print(f'Best accuracy: {best_accuracy} at depth {best_clf.max_depth} at {best_estimator} estimators')

    plt.plot(max_depths, accuracies)
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Max Depth')
    plt.show()

    return best_clf


def calculate_growth(stock_data_path, dates):
    """
    Calculate growth in % for each date in the sentiment dataset.

    :param stock_data_path: Path to the stock CSV file.
    :param dates: List of sentiment dates.
    :return: Growth values corresponding to the sentiment dates.
    """
    growth = {}

    with open(stock_data_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            if row[0] in dates:
                growth[row[0]] = round(float(row[2]), 2)
    return growth


def plot_sentiment(start_date, end_date, clf, count=10, stock=STOCK, api_key=NEWS_API_KEY, ax=None, sort='relevancy'):
    """Plot the sentiment and stock data.

    Provides the overall trend of the stock for the given period. YYYY-MM-DD format.
    """
    query = f"{stock} stock OR {stock} OR {stock} earnings OR {stock} market news OR {stock} stock price OR {stock} stock news"
    url = f'https://newsapi.org/v2/everything?q={query}&language=en&sortBy={sort}&pageSize={count}&apiKey={api_key}'
    url += f"&from={start_date}"
    url += f"&to={end_date}"
    response = requests.get(url)
    articles = response.json()['articles']
    articles.sort(key=lambda x: datetime.fromisoformat(x['publishedAt'].replace("Z", "")))
    dates = []

    article_dates = {}
    sentiments_probabilities = {}
    sentiments_names = ['negative', 'neutral', 'positive']
    sentiments = {}

    for article in articles:
        article_date = article['publishedAt'][:10]

        if article_date not in dates:
            dates.append(article_date)

        if article_date not in article_dates:
            article_dates[article_date] = []
            sentiments_probabilities[article_date] = []
            sentiments[article_date] = []

        article_dates[article_date].append(article['title'])
        sentiments_probabilities[article_date].append(
            clf.predict_proba(vectorizer.transform([clean_article(article['title'])]))[0]
        )
        sentiments[article_date].append(clf.predict(vectorizer.transform([clean_article(article['title'])])[0]))

    dates = sorted(dates)
    growth_dict = calculate_growth(f'stock-predictor/cleaned_{stock}.csv', dates)

    previous = None
    for date in dates:
        if date not in growth_dict:
            growth_dict[date] = previous
        else:
            previous = growth_dict[date]

    plot_dates = [date for date in dates if date in growth_dict]
    growth = [growth_dict[date] for date in plot_dates]

    # For 3 sentiment values
    # sentiments_rgb = {
    #     date: np.array([
    #         np.mean(sentiments_probabilities[date], axis=0)[0] * 255,
    #         np.mean(sentiments_probabilities[date], axis=0)[1] * 255,
    #         np.mean(sentiments_probabilities[date], axis=0)[2] * 255
    #     ]).astype(int)
    #     for date in plot_dates
    # }

    # Binary sentiment decision
    # sentiments = {
    #     date: sentiments_names[np.argmax(np.mean(sentiments_probabilities[date], axis=0))]
    #     for date in sentiments_probabilities
    # }
    # for date in sentiments:
    #     avg_probs = np.mean(sentiments_probabilities[date], axis=0)
        
    #     if sentiments[date] == 'neutral':
    #         sentiments[date] = 'positive' if avg_probs[2] > avg_probs[0] else 'negative'
    # sentiments_rgb = {
    #     date: np.array([
    #         255 if sentiments[date] == 'negative' else 0,
    #         255 if sentiments[date] == 'positive' else 0,
    #         0
    #     ]).astype(int)
    #     for date in plot_dates
    # }

    sentiments_rgb = {
        date: np.array([
            255 if sum(sentiments[date])/count <= 0 else 0,
            255 if sum(sentiments[date], axis=0)/count > 0 else 0,
            0
        ]).astype(int)
        for date in plot_dates
    }
    

    assert len(plot_dates) == len(growth) == len(sentiments_rgb)

    if ax is None:
        ax = plt.gca()

    for i, date in enumerate(plot_dates):
        plt.scatter(date, growth[i], color=sentiments_rgb[date] / 255.0, s=100, alpha=0.7)

    plt.plot(plot_dates, growth, linewidth=1, label=stock)

    return plot_dates


def plot_sentiment_alternate(start_date, end_date, clf, count=10, stock=STOCK, api_key=NEWS_API_KEY, ax=None, sort='relevancy'):
    """Plot the sentiment and stock data.

    Provides the overall trend of the stock for the given period. YYYY-MM-DD format.
    """
    query = f"{stock} stock OR {stock} OR {stock} earnings OR {stock} market news OR {stock} stock price OR {stock} stock news"
    url = f'https://newsapi.org/v2/everything?q={query}&language=en&sortBy={sort}&pageSize={count}&apiKey={api_key}'
    url += f"&from={start_date}"
    url += f"&to={end_date}"
    response = requests.get(url)
    articles = response.json()['articles']
    articles.sort(key=lambda x: datetime.fromisoformat(x['publishedAt'].replace("Z", "")))
    dates = []

    article_dates = {}
    sentiments = {}
    sentiments_rgb = {}
    thoughts = {}

    for article in articles:
        article_date = article['publishedAt'][:10]

        if article_date not in dates:
            dates.append(article_date)

        if article_date not in article_dates:
            article_dates[article_date] = []
            sentiments[article_date] = []

        article_dates[article_date].append(article['title'])
        sentiments[article_date].append(clf.predict(vectorizer.transform([clean_article(article['title'])])[0])[0])

    dates = sorted(dates)
    growth_dict = calculate_growth(f'stock-predictor/cleaned_{stock}.csv', dates)

    previous = None
    for date in dates:
        if date not in growth_dict:
            growth_dict[date] = previous
        else:
            previous = growth_dict[date]

    plot_dates = [date for date in dates if date in growth_dict]
    growth = [growth_dict[date] for date in plot_dates]
    
    for date in plot_dates:
        positive_count = sentiments[date].count('1')  # Assuming positive is '1'
        negative_count = sentiments[date].count('-1')  # Assuming negative is '-1'
        total_count = positive_count + negative_count  # Total sentiments for normalization

        # Avoid division by zero
        if total_count == 0:
            total_count = 1

        # Calculate RGB values based on relative dominance
        thoughts[date] = [positive_count, negative_count]
        sentiments_rgb[date] = np.array([
            255 * (negative_count / total_count),  # Red (negative)
            255 * (positive_count / total_count),  # Green (positive)
            0  # Blue remains 0 for simplicity
        ]).astype(int)

    assert len(plot_dates) == len(growth) == len(sentiments_rgb)

    if ax is None:
        ax = plt.gca()

    for i, date in enumerate(plot_dates):
        plt.scatter(date, growth[i], color=sentiments_rgb[date] / 255.0, s=100, alpha=0.7)
        plt.text(date, growth[i], f'{thoughts[date][0]}:{thoughts[date][1]}', fontsize=8)

    plt.plot(plot_dates, growth, linewidth=1, label=stock)
    plt.xticks(rotation=45, fontsize=6)

    return plot_dates


if __name__ == '__main__':
    data_path_alternate = 'stock-predictor/data/sentiments_alternate.csv'
    # X_train, y_train, X_val, y_val, X_test, y_test = load_data(DATA_PATH)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_path_alternate, flip=True)
    depth_list = list(range(1, 201))

    # clf = train_classifier(X_train, y_train, X_val, y_val, estimators=100)
    clf = train_classifier(X_train, y_train, X_val, y_val, estimators=100, depth=depth_list)

    print("Validation score:", clf.score(X_val, y_val))

    print("Test score:", clf.score(X_test, y_test))


    # news = get_news(STOCK, NEWS_API_KEY, count=100)

    # sentiments = ['negative', 'neutral', 'positive']
    # sentiments_probabilities = []
    # sentiment = []

    # for article in news:
    #     # sentiments_probabilities.append(clf.predict_proba(vectorizer.transform([clean_article(article['title'])]))[0])
    #     # print(sentiments_probabilities)
    #     sentiment.append(clf.predict(vectorizer.transform([clean_article(article['title'])])[0])[0])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_sentiment_alternate('2024-10-30', '2024-11-27', clf, count=100, ax=ax)

    ax.set_title('Stock Sentiment Analysis')
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price')
    ax.legend()

    plt.show()
    