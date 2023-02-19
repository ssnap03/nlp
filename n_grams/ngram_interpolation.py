import numpy as np

from ngram_vanilla import NGramVanilla, NEG_INFINITY


class NGramInterpolation(NGramVanilla):

    def __init__(self, lambdas, vsize):
        self.lambdas = lambdas
        self.sub_models = [NGramVanilla(n, vsize) for n in range(1, len(lambdas) + 1)]
    
    def estimate(self, sequences):
        for sub_model in self.sub_models:
            sub_model.estimate(sequences)

    def sequence_logp(self, sequence):
        n = len(self.lambdas)
        padded_sequence = ['<bos>']*(n - 1) + sequence + ['<eos>']
        total_logp = 0
        for i in range(len(padded_sequence) - n + 1):
            ngram = tuple(padded_sequence[i:i+n])
            logp = np.log2(self.ngram_prob(ngram))
            total_logp += max(NEG_INFINITY, logp)
        return total_logp

    def ngram_prob(self, ngram):
        """Return the smoothed probability of an n-gram with interpolation smoothing."""
        prob = 0
        for i in range(0, len(self.lambdas)):
            prob += self.lambdas[i]*self.sub_models[i].ngram_prob(ngram[-self.sub_models[i].n:])
        return prob
