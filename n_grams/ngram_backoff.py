import numpy as np

from ngram_vanilla import NGramVanilla


class NGramBackoff(NGramVanilla):
    def __init__(self, n, vsize):
        self.n = n
        self.sub_models = [NGramVanilla(k, vsize) for k in range(1, n + 1)]

    def estimate(self, sequences):
        for sub_model in self.sub_models:
            sub_model.estimate(sequences)

    def ngram_prob(self, ngram):
        """Return the smoothed probability with backoff.
        
        That is, if the n-gram count of size self.n is defined, return that.
        Otherwise, check the n-gram of size self.n - 1, self.n - 2, etc. until you find one that is defined.
        """
        ngram_updated = ngram
        for i in range(len(self.sub_models)-1, 0, -1):
            prefix = ngram_updated[:-1]
            token = ngram_updated[-1]  
            if(self.sub_models[i].count[tuple(prefix)][token] > 0): 
                return self.sub_models[i].ngram_prob(ngram_updated)
            ngram_updated = ngram_updated[1:]
            
        return self.sub_models[0].ngram_prob(ngram_updated)
