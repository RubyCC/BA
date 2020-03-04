>>> from nltk.tag import pos_tag
>>> tokens = ['It', 'was', 'a', 'bright', 'cold', 'day', 'in', 'April', ',', 'and', 'the', 'clocks', 'were', 'striking', 'thirteen', '.']
>>> pos_tag(tokens)
[('It', 'PRP'), ('was', 'VBD'), ('a', 'DT'), ('bright', 'JJ'), ('cold', 'JJ'), ('day', 'NN'), ('in', 'IN'), ('April', 'NNP'), (',', ','), ('and', 'CC'), ('the', 'DT'), ('clocks', 'NNS'), ('were', 'VBD'), ('striking', 'VBG'), ('thirteen', 'NN'), ('.', '.')]