>>> from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
>>> sentence = "The book was good."
>>> analyzer = SentimentIntensityAnalyzer()
>>> analyzer.polarity_scores(sentence)
{'pos': 0.492, 'compound': 0.4404, 'neu': 0.508, 'neg': 0.0}