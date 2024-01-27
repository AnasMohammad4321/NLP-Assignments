
import os.path
import sys
import random
from operator import itemgetter
from collections import defaultdict

# Constants
UNK = "UNK"     # Unknown word token
start = "<s>"   # Start-of-sentence token
end = "</s>"    # End-of-sentence-token


def readFileToCorpus(f):
    """ Reads in the text file f which contains one sentence per line.
    Args:
        f (str): File path to the corpus file.

    Returns:
        list: A list of sentences, where each sentence is represented as a list of words.
    """
    if os.path.isfile(f):
        with open(f, "r") as file:
            i = 0
            corpus = []

            print("Reading file ", f)

            for line in file:
                i += 1
                sentence = line.split()
                corpus.append(sentence)

                if i % 1000 == 0:
                    sys.stderr.write("Reading sentence " + str(i) + "\n")

        return corpus
    else:
        print("Error: Corpus file", f, "does not exist")
        sys.exit()


def preprocess(corpus):
    """
    Preprocesses the input corpus by replacing rare words with UNK, and bookending sentences with start and end tokens.

    Args:
        corpus (list): A list of sentences, where each sentence is represented as a list of words.

    Returns:
        list: Preprocessed corpus with rare words replaced by UNK and sentences bookended with start and end tokens.
    """
    freqDict = defaultdict(int)
    for sen in corpus:
        for word in sen:
            freqDict[word] += 1

    for sen in corpus:
        for i in range(len(sen)):
            word = sen[i]
            if freqDict[word] < 2:
                sen[i] = UNK

    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)

    return corpus


def preprocessTest(vocab, corpus):
    """
    Preprocesses a test corpus by replacing words that were unseen in the training with UNK, and bookending sentences with start and end tokens.

    Args:
        vocab (set): A set containing the vocabulary of the training corpus.
        corpus (list): A list of sentences in the test corpus, where each sentence is represented as a list of words.

    Returns:
        list: Preprocessed test corpus with unseen words replaced by UNK and sentences bookended with start and end tokens.
    """
    for sen in corpus:
        for i in range(len(sen)):
            word = sen[i]
            if word not in vocab:
                sen[i] = UNK

    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)

    return corpus

# --------------------------------------------------------------
# Language models and data structures
# --------------------------------------------------------------

# Parent class for the three language models you need to implement


class LanguageModel:
    # Initialize and train the model (ie, estimate the model's underlying probability
    # distribution from the training corpus)
    def __init__(self, corpus):
        print("""Your task is to implement four kinds of n-gram language models:
      a) an (unsmoothed) unigram model (UnigramModel)
      b) a unigram model smoothed using Laplace smoothing (SmoothedUnigramModel)
      c) an unsmoothed bigram model (BigramModel)
      d) a bigram model smoothed using linear interpolation smoothing (SmoothedBigramModelInt)
      """)
    # enddef

    # Generate a sentence by drawing words according to the
    # model's probability distribution
    # Note: think about how to set the length of the sentence
    # in a principled way
    def generateSentence(self):
        print("Implement the generateSentence method in each subclass")
        return "mary had a little lamb ."
    # emddef

    # Given a sentence (sen), return the probability of
    # that sentence under the model
    def getSentenceProbability(self, sen):
        print("Implement the getSentenceProbability method in each subclass")
        return 0.0
    # enddef

    # Given a corpus, calculate and return its perplexity
    # (normalized inverse log probability)
    def getCorpusPerplexity(self, corpus):
        print("Implement the getCorpusPerplexity method")
        return 0.0
    # enddef

    # Given a file (filename) and the number of sentences, generate a list
    # of sentences and write each to file along with its model probability.
    # Note: you shouldn't need to change this method
    def generateSentencesToFile(self, numberOfSentences, filename):
        filePointer = open(filename, 'w+')
        for i in range(0, numberOfSentences):
            sen = self.generateSentence()
            prob = self.getSentenceProbability(sen)

            stringGenerated = str(prob) + " " + " ".join(sen)
            print(stringGenerated, end="\n", file=filePointer)

        # endfor
    # enddef
# endclass

# Unigram language model


class UnigramModel(LanguageModel):
    def __init__(self, corpus):
        print("Subtask: implement the unsmoothed unigram language model")
    # endddef
# endclass

# Smoothed unigram language model (use laplace for smoothing)


class SmoothedUnigramModel(LanguageModel):
    def __init__(self, corpus):
        print("Subtask: implement the smoothed unigram language model")
    # endddef
# endclass

# Unsmoothed bigram language model


class BigramModel(LanguageModel):
    def __init__(self, corpus):
        print("Subtask: implement the unsmoothed bigram language model")
    # endddef
# endclass


# Smoothed bigram language model (use linear interpolation for smoothing, set lambda1 = lambda2 = 0.5)
class SmoothedBigramModelKN(LanguageModel):
    def __init__(self, corpus):
        print("Subtask: implement the smoothed bigram language model with kneser-ney smoothing")
    # endddef
# endclass


# Sample class for a unsmoothed unigram probability distribution
# Note:
#       Feel free to use/re-use/modify this class as necessary for your
#       own code (e.g. converting to log probabilities after training).
#       This class is intended to help you get started
#       with your implementation of the language models above.
class UnigramDist:
    def __init__(self, corpus):
        self.counts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)
    # endddef

    # Add observed counts from corpus to the distribution
    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.counts[word] += 1.0
                self.total += 1.0
            # endfor
        # endfor
    # enddef

    # Returns the probability of word in the distribution
    def prob(self, word):
        return self.counts[word]/self.total
    # enddef

    # Generate a single random word according to the distribution
    def draw(self):
        rand = random.random()
        for word in self.counts.keys():
            rand -= self.prob(word)
            if rand <= 0.0:
                return word
            # endif
        # endfor
    # enddef
# endclass


# -------------------------------------------
# The main routine
# -------------------------------------------
if __name__ == "__main__":
    # read your corpora
    trainCorpus = readFileToCorpus('train.txt')
    # print(trainCorpus)
    trainCorpus = preprocess(trainCorpus)
    # print(trainCorpus)
    posTestCorpus = readFileToCorpus('pos_test.txt')
    negTestCorpus = readFileToCorpus('neg_test.txt')

    vocab = set()
    # Please write the code to create the vocab over here before the function preprocessTest
    for sentence in trainCorpus:
        for word in sentence:
            vocab.add(word)
    print("""Task 0: create a vocabulary(collection of word types) for the train corpus""")
    print("Vocabulary Size:", len(vocab))

    posTestCorpus = preprocessTest(vocab, posTestCorpus)
    negTestCorpus = preprocessTest(vocab, negTestCorpus)

    # # Run sample unigram dist code
    # unigramDist = UnigramDist(trainCorpus)
    # print("Sample UnigramDist output:")
    # print("Probability of \"picture\": ", unigramDist.prob("picture"))
    # print("\"Random\" draw: ", unigramDist.draw())
