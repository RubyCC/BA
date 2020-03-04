>>> from nltk.stem import WordNetLemmatizer
>>> wnl = WordNetLemmatizer()
>>> tokens = ['better', 'corpora', 'books']
>>> lemmata = [wnl.lemmatize(t) for t in tokens]
['good', 'corpus', 'book']