import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        return 1


class NeuralNet(nn.Module):

    def __init__(self, embeddings: nn.Embedding, hidden_dim: int):
        # TODO: Your code here!
        super(NeuralNet, self).__init__()
        self.embeddings = embeddings
        self.embeddings.weight.requires_grad = False
        self.hidden_dim = hidden_dim
        self.hidden_layer = nn.Linear(50, self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, 2)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input) -> torch.Tensor:
        """Takes a list of words and returns a tensor of class predictions.
        
        input: torch tensor of a sentence with token indices
        input.shape: (len(sentence))
        
        Steps:
          1. Embed each word using self.embeddings.
          2. Take the average embedding of each word.
          3. Pass the average embedding through a neural net with one hidden layer (with ReLU nonlinearity).
          4. Return a tensor representing a log-probability distribution over classes.
        """
        out = self.embeddings(input)                  # out.shape: [len(sentence), 50]
        out = torch.mean(out,dim=0).reshape(1,-1)     # out.shape: [1,50]
        out = self.hidden_layer(out)                  # out.shape: [1,100]
        out = F.relu(out)                             # out.shape: [1,100]
        out = self.linear(out)                        # out.shape: [1,2]                              
        out = self.logSoftmax(out)                    # out.shape: [1,2]
        return out


class NeuralSentimentClassifier(SentimentClassifier):
    def __init__(self, hidden_dim: int, word_embeddings: WordEmbeddings):
        # TODO: Your code here!
        self.hidden_dim = hidden_dim
        self.embeddings = word_embeddings.get_initialized_embedding_layer()
        self.word_embeddings = word_embeddings
        self.classifier = NeuralNet(self.embeddings,self.hidden_dim)
        
    # def train_(self, ex_words: List[str]) -> torch.Tensor:
    #     inp = torch.tensor([self.embeddings.word_indexer.index_of(word) for word in ex_words])
    #     return self.classifier.forward(inp)

    def predict(self, ex_words: List[str]) -> int:
        # TODO: Your code here!
        inp = [self.word_embeddings.word_indexer.index_of(word) for word in ex_words]
        inp = torch.tensor([1 if ind==-1 else ind for ind in inp])
        out = self.classifier.forward(inp)
        return out.argmax(dim=1).item()


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args
    :param train_exs: training examples
    :param dev_exs: development set
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    n_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    
    nsc = NeuralSentimentClassifier(args.hidden_size, word_embeddings)
    net = nsc.classifier
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    train_sentences = []
    train_labels = []
    for example in train_exs:
        train_sentences.append(example.words)
        train_labels.append(example.label)
    
    n_train_samples = len(train_labels)
    
    for epoch in range(n_epochs):
        train_indices = np.random.permutation(n_train_samples)
        for i in range(0,n_train_samples,batch_size):
            if i+batch_size > n_train_samples: break
            curr_indices = train_indices[i:i+batch_size]
            net.zero_grad()
            predicted_tensors = []
            target_labels = []
            for index in curr_indices:
                input_sentence = train_sentences[index]
                input_sentence = [word_embeddings.word_indexer.index_of(word) for word in input_sentence]
                input_sentence =torch.tensor([1 if ind==-1 else ind for ind in input_sentence])
                #target_tensor = torch.tensor([train_labels[index]])
                target_labels.append(train_labels[index])
                out_tensor = net(input_sentence)
                predicted_tensors.append(out_tensor)
            loss = criterion(torch.cat(predicted_tensors),torch.tensor(target_labels))
            loss.backward()
            optimizer.step()
    return nsc

