import requests

API_KEY = 'your_finnhub_api_key'
symbol = 'BSCW'
url = f"https://finnhub.io/api/v1/news-sentiment?symbol={BPT}&token={ct2huqhr01qiurr3u6rgct2huqhr01qiurr3u6s0}"

response = requests.get(url)
sentiment_data = response.json()
print(sentiment_data)
