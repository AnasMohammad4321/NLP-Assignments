"""
    Name: Mohammad Anas
    Roll Number: 20L-1289
    Section: BDS-8A
    Part 2
    Assignment 1
"""

import math
from collections import Counter
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
        #     print("""Your task is to implement four kinds of n-gram language models:
        #   a) an (unsmoothed) unigram model (UnigramModel)
        #   b) a unigram model smoothed using Laplace smoothing (SmoothedUnigramModel)
        #   c) an unsmoothed bigram model (BigramModel)
        #   d) a bigram model smoothed using linear interpolation smoothing (SmoothedBigramModelInt)
        #   """)
        self.corpus = corpus
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
        super().__init__(corpus)
        self.counts = defaultdict(float)
        self.total = 0.0
        self.vocab_size = 0  # To keep track of the vocabulary size
        self.train(corpus)

    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                self.counts[word] += 1.0
        self.total = sum(self.counts.values())
        # Update vocabulary size based on unique words
        self.vocab_size = len(self.counts)

    def draw(self):
        # Improved draw method to ensure it doesn't go infinite loop
        rand = random.random() * self.total
        cumulative = 0.0
        for word, count in self.counts.items():
            cumulative += count
            if cumulative >= rand:
                return word
        return end  # Fallback to ensure method completes

    def probability(self, word):
        # Use total calculated in train to avoid recalculating sum each time
        return self.counts[word] / self.total if word in self.counts else 0

    def generateSentence(self):
        sentence = [start]
        while True:
            next_word = self.draw()
            # Ensure sentences don't become too long
            if next_word == end or len(sentence) >= 20:
                sentence.append(end)
                break
            else:
                sentence.append(next_word)
        if sentence[0] == start or sentence[-1] == end:
            del sentence[-1]  # Remove end token if it's not needed
            del sentence[0]  # Remove start token if it's not needed
        return sentence

    def getSentenceProbability(self, sen):
        log_probability = 0.0
        for word in sen[1:]:  # Skip <s> since it doesn't contribute to probability in unigram model
            word_prob = self.probability(word)
            if word_prob > 0:
                log_probability += math.log(word_prob)
            else:
                # Return -inf if any word in the sentence has 0 probability
                return float('-inf')
        return log_probability

    def getCorpusPerplexity(self, corpus):
        total_log_probability = 0.0
        total_words = 0
        for sentence in corpus:
            total_words += len(sentence) - 1  # Exclude <s> from word count
            sentence_log_probability = self.getSentenceProbability(sentence)
            if sentence_log_probability == float('-inf'):
                # Return inf if any sentence has -inf log probability
                return float('inf')
            total_log_probability += sentence_log_probability
        return math.exp(-total_log_probability / total_words)


# Smoothed unigram language model (use laplace for smoothing)
class SmoothedUnigramModel(LanguageModel):
    def __init__(self, corpus):
        super().__init__(corpus)
        self.counts = defaultdict(float)
        self.total = 0.0
        self.vocab_size = 0
        self.train(corpus)

    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                self.counts[word] += 1.0
                self.total += 1.0
        self.vocab_size = len(self.counts)

    def probability(self, word):
        return (self.counts[word] + 1.0) / (self.total + self.vocab_size)

    def generateSentence(self):
        sentence = [start]
        while True:
            next_word = self.draw()
            sentence.append(next_word)
            if next_word == end or len(sentence) > 20:
                break
        if sentence[0] == start or sentence[-1] == end:
            del sentence[-1]  # Remove end token if it's not needed
            del sentence[0]  # Remove start token if it's not needed
        return sentence

    def draw(self):
        rand = random.random()
        cumulative_prob = 0.0
        for word in self.counts.keys():
            cumulative_prob += self.probability(word)
            if rand < cumulative_prob:
                return word

    def getSentenceProbability(self, sen):
        log_probability = 0.0
        for word in sen:
            log_probability += math.log(self.probability(word))
        return log_probability

    def getCorpusPerplexity(self, corpus):
        total_log_probability = 0.0
        total_words = 0
        for sentence in corpus:
            total_words += len(sentence)
            total_log_probability += self.getSentenceProbability(sentence)
        perplexity = math.exp(-total_log_probability / total_words)
        return perplexity


# Unsmoothed bigram language model
class BigramModel(LanguageModel):
    def __init__(self, corpus):
        super().__init__(corpus)
        self.bigrams = defaultdict(lambda: defaultdict(float))
        self.unigram_counts = defaultdict(float)
        self.train(corpus)

    def train(self, corpus):
        for sen in corpus:
            previous_word = None
            for word in sen:
                if previous_word is not None:
                    self.bigrams[previous_word][word] += 1.0
                self.unigram_counts[word] += 1.0
                previous_word = word

    def probability(self, previous_word, word):
        return self.bigrams[previous_word][word] / self.unigram_counts[previous_word] if self.unigram_counts[previous_word] > 0 else 0.0

    def generateSentence(self):
        sentence = [start]
        while True:
            next_word = self.draw(sentence[-1])
            sentence.append(next_word)
            if next_word == end or len(sentence) > 20:
                break
        if sentence[0] == start or sentence[-1] == end:
            del sentence[-1]  # Remove end token if it's not needed
            del sentence[0]  # Remove start token if it's not needed
        return sentence

    def draw(self, previous_word):
        words = list(self.bigrams[previous_word].keys())
        probabilities = [self.probability(previous_word, w) for w in words]
        probabilities_sum = sum(probabilities)
        rand = random.random() * probabilities_sum
        for i, word in enumerate(words):
            rand -= probabilities[i]
            if rand <= 0:
                return word
        return end  # Fallback in case of rounding errors

    def getSentenceProbability(self, sen):
        log_probability = 0.0
        previous_word = None
        for word in sen:
            if previous_word is not None:
                prob = self.probability(previous_word, word)
                if prob > 0:
                    log_probability += math.log(prob)
                else:
                    return float('-inf')
            previous_word = word
        return log_probability

    def getCorpusPerplexity(self, corpus):
        total_log_probability = 0.0
        total_bigrams = 0
        for sentence in corpus:
            total_bigrams += len(sentence) - 1  # Start with the second word
            total_log_probability += self.getSentenceProbability(sentence)
        perplexity = math.exp(-total_log_probability / total_bigrams)
        return perplexity


# Smoothed bigram language model (use linear interpolation for smoothing, set lambda1 = lambda2 = 0.5)
class SmoothedBigramModelKN(LanguageModel):
    def __init__(self, corpus, lambda1=0.5, lambda2=0.5):
        super().__init__(corpus)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.bigram_model = BigramModel(corpus)
        self.unigram_model = SmoothedUnigramModel(corpus)

    def probability(self, previous_word, word):
        return self.lambda1 * self.unigram_model.probability(word) + self.lambda2 * self.bigram_model.probability(previous_word, word)

    def generateSentence(self):
        return self.bigram_model.generateSentence()  # Reuse bigram generation logic

    def getSentenceProbability(self, sen):
        log_probability = 0.0
        previous_word = None
        for word in sen:
            if previous_word is not None:
                prob = self.probability(previous_word, word)
                log_probability += math.log(prob)
            previous_word = word
        return log_probability

    def getCorpusPerplexity(self, corpus):
        # Delegate to bigram model
        return self.bigram_model.getCorpusPerplexity(corpus)


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
    # Read your corpora
    trainCorpus = readFileToCorpus('train.txt')
    trainCorpus = preprocess(trainCorpus)

    posTestCorpus = readFileToCorpus('pos_test.txt')
    negTestCorpus = readFileToCorpus('neg_test.txt')

    # Create the vocabulary
    vocab = set()
    for sentence in trainCorpus:
        for word in sentence:
            vocab.add(word)
    print("Task 0: create a vocabulary (collection of word types) for the train corpus")
    print("Vocabulary Size:", len(vocab))

    # Preprocess test corpora
    posTestCorpus = preprocessTest(vocab, posTestCorpus)
    negTestCorpus = preprocessTest(vocab, negTestCorpus)

    # UnigramModel
    unigramModel = UnigramModel(trainCorpus)
    print("\nUnigram Model:")
    generated_sentence = unigramModel.generateSentence()
    print("Generated Sentence:", ' '.join(generated_sentence))
    sentence_probability = unigramModel.getSentenceProbability(
        generated_sentence)
    print("Sentence Log Probability:", sentence_probability)
    corpus_perplexity = unigramModel.getCorpusPerplexity(trainCorpus)
    print("Corpus Perplexity:", corpus_perplexity)

    # SmoothedUnigramModel
    smoothedUnigramModel = SmoothedUnigramModel(trainCorpus)
    print("\nSmoothed Unigram Model:")
    generated_sentence = smoothedUnigramModel.generateSentence()
    print("Generated Sentence:", ' '.join(generated_sentence))
    sentence_probability = smoothedUnigramModel.getSentenceProbability(
        generated_sentence)
    print("Sentence Log Probability:", sentence_probability)
    corpus_perplexity = smoothedUnigramModel.getCorpusPerplexity(trainCorpus)
    print("Corpus Perplexity:", corpus_perplexity)

    # BigramModel
    bigramModel = BigramModel(trainCorpus)
    print("\nBigram Model:")
    generated_sentence = bigramModel.generateSentence()
    print("Generated Sentence:", ' '.join(generated_sentence))
    sentence_probability = bigramModel.getSentenceProbability(
        generated_sentence)
    print("Sentence Log Probability:", sentence_probability)
    corpus_perplexity = bigramModel.getCorpusPerplexity(trainCorpus)
    print("Corpus Perplexity:", corpus_perplexity)

    # SmoothedBigramModelKN (Linear Interpolation)
    smoothedBigramModelKN = SmoothedBigramModelKN(trainCorpus)
    print("\nSmoothed Bigram Model (KN):")
    generated_sentence = smoothedBigramModelKN.generateSentence()
    print("Generated Sentence:", ' '.join(generated_sentence))
    sentence_probability = smoothedBigramModelKN.getSentenceProbability(
        generated_sentence)
    print("Sentence Log Probability:", sentence_probability)
    corpus_perplexity = smoothedBigramModelKN.getCorpusPerplexity(trainCorpus)
    print("Corpus Perplexity:", corpus_perplexity)
