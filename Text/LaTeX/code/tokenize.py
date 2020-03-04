>>> from nltk.tokenize import sent_tokenize, word_tokenize
>>> text = 'But it was alright, everything was alright, the struggle was finished. He had won the victory over himself. He loved Big Brother.'
>>> sentences = sent_tokenize(text)
['But it was alright, everything was alright, the struggle was finished.', 'He had won the victory over himself.', 'He loved Big Brother.']
>>> tokens = [word_tokenize(s) for s in sentences]
[['But', 'it', 'was', 'alright', ',', 'everything', 'was', 'alright', ',', 'the', 'struggle', 'was', 'finished', '.'], ['He', 'had', 'won', 'the', 'victory', 'over', 'himself', '.'], ['He', 'loved', 'Big', 'Brother', '.']]