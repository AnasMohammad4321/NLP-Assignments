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


class LanguageModel:
    def __init__(self, corpus):
        self.corpus = corpus

    def generateSentence(self):
        print("Implement the generateSentence method in each subclass")
        return "mary had a little lamb ."

    def getSentenceProbability(self, sen):
        print("Implement the getSentenceProbability method in each subclass")
        return 0.0

    def getCorpusPerplexity(self, corpus):
        print("Implement the getCorpusPerplexity method")
        return 0.0

    def generateSentencesToFile(self, numberOfSentences, filename):
        filePointer = open(filename, 'w+')
        for i in range(0, numberOfSentences):
            sen = self.generateSentence()
            prob = self.getSentenceProbability(sen)

            stringGenerated = str(prob) + " " + " ".join(sen)
            print(stringGenerated, end="\n", file=filePointer)


class UnigramModel(LanguageModel):
    """A unigram language model.

    Args:
        corpus (list of list of str): List of sentences, each represented as a list of words.
    """

    def __init__(self, corpus):
        """Initialize the UnigramModel.

        Args:
            corpus (list of list of str): List of sentences, each represented as a list of words.
        """
        super().__init__(corpus)
        self.counts = defaultdict(float)
        self.total = 0.0
        self.vocab_size = 0
        self.train(corpus)

    def train(self, corpus):
        """Train the unigram model.

        Args:
            corpus (list of list of str): List of sentences, each represented as a list of words.
        """
        for sen in corpus:
            for word in sen:
                self.counts[word] += 1.0
        self.total = sum(self.counts.values())
        self.vocab_size = len(self.counts)

    def draw(self):
        """Draw a word from the unigram distribution.

        Returns:
            str: The drawn word.
        """
        rand = random.random() * self.total
        cumulative = 0.0
        for word, count in self.counts.items():
            cumulative += count
            if cumulative >= rand:
                return word
        return end  # Fallback to ensure method completes

    def probability(self, word):
        """Calculate the probability of a word in the unigram model.

        Args:
            word (str): The word for which to calculate the probability.

        Returns:
            float: The probability of the word in the unigram model.
        """
        # Use total calculated in train to avoid recalculating sum each time
        return self.counts[word] / self.total if word in self.counts else 0

    def generateSentence(self):
        """Generate a sentence using the unigram model.

        Returns:
            list of str: The generated sentence.
        """
        sentence = [start]
        current_word = start

        while True:
            next_word = self.draw()
            if next_word == end or len(sentence) >= 20:
                break
            elif next_word != start:
                sentence.append(next_word)
                current_word = next_word
        if sentence[-1] != end:
            sentence.append(end)
        return sentence

    def getSentenceProbability(self, sen):
        """Calculate the log probability of a sentence in the unigram model.

        Args:
            sen (list of str): The sentence for which to calculate the log probability.

        Returns:
            float: The log probability of the sentence.
        """
        log_probability = 0.0
        for word in sen[1:]:  # Skip <s> since it doesn't contribute to probability in unigram model
            word_prob = self.probability(word)
            if word_prob > 0:
                log_probability += math.log(word_prob)
            else:
                return float('-inf')
        return log_probability

    def getCorpusPerplexity(self, corpus):
        """Calculate the perplexity of a corpus in the unigram model.

        Args:
            corpus (list of list of str): The corpus for which to calculate perplexity.

        Returns:
            float: The perplexity of the corpus.
        """
        total_log_probability = 0.0
        total_words = 0
        for sentence in corpus:
            total_words += len(sentence) - 1  # Exclude <s> from word count
            sentence_log_probability = self.getSentenceProbability(sentence)
            if sentence_log_probability == float('-inf'):
                return float('inf')
            total_log_probability += sentence_log_probability
        return math.exp(-total_log_probability / total_words)

    def generateSentencesToFile(self, numberOfSentences, filename):
        """Generate sentences using the unigram model and write them to a file.

        Args:
            numberOfSentences (int): The number of sentences to generate.
            filename (str): The name of the file to write the sentences to.
        """
        with open(filename, 'w', encoding='utf-8') as file:
            for _ in range(numberOfSentences):
                sentence = self.generateSentence()
                file.write(' '.join(sentence) + '\n')


class SmoothedUnigramModel(LanguageModel):
    """A smoothed unigram language model using Laplace (add-one) smoothing.

    Args:
        corpus (list of list of str): List of sentences, each represented as a list of words.
    """

    def __init__(self, corpus):
        """Initialize the SmoothedUnigramModel.

        Args:
            corpus (list of list of str): List of sentences, each represented as a list of words.
        """
        super().__init__(corpus)
        self.counts = defaultdict(float)
        self.total = 0.0
        self.vocab_size = 0
        self.train(corpus)

    def train(self, corpus):
        """Train the smoothed unigram model.

        Args:
            corpus (list of list of str): List of sentences, each represented as a list of words.
        """
        for sen in corpus:
            for word in sen:
                self.counts[word] += 1.0
                self.total += 1.0
        self.vocab_size = len(self.counts)

    def probability(self, word):
        """Calculate the Laplace-smoothed probability of a word.

        Args:
            word (str): The word for which to calculate the probability.

        Returns:
            float: The Laplace-smoothed probability of the word.
        """
        return (self.counts[word] + 1.0) / (self.total + self.vocab_size)

    def generateSentence(self):
        """Generate a sentence using the smoothed unigram model.

        Returns:
            list of str: The generated sentence.
        """
        sentence = [start]
        while True:
            next_word = self.draw()
            if next_word != start:  # Skip the start token
                sentence.append(next_word)
            if next_word == end or len(sentence) > 20:
                break
        if sentence[-1] != end:
            sentence.append(end)
        return sentence

    def draw(self):
        """Draw the next word based on the smoothed unigram probabilities.

        Returns:
            str: The next word in the sequence.
        """
        rand = random.random()
        cumulative_prob = 0.0
        for word in self.counts.keys():
            cumulative_prob += self.probability(word)
            if rand < cumulative_prob:
                return word

    def getSentenceProbability(self, sen):
        """Calculate the log probability of a sentence.

        Args:
            sen (list of str): The sentence for which to calculate the log probability.

        Returns:
            float: The log probability of the sentence.
        """
        log_probability = 0.0
        for word in sen:
            log_probability += math.log(self.probability(word))
        return log_probability

    def getCorpusPerplexity(self, corpus):
        """Calculate the perplexity of a corpus.

        Args:
            corpus (list of list of str): The corpus for which to calculate perplexity.

        Returns:
            float: The perplexity of the corpus.
        """
        total_log_probability = 0.0
        total_words = 0
        for sentence in corpus:
            total_words += len(sentence)
            total_log_probability += self.getSentenceProbability(sentence)
        perplexity = math.exp(-total_log_probability / total_words)
        return perplexity

    def generateSentencesToFile(self, numberOfSentences, filename):
        """Generate sentences using the smoothed unigram model and write them to a file.

        Args:
            numberOfSentences (int): The number of sentences to generate.
            filename (str): The name of the file to write the sentences to.
        """
        with open(filename, 'w', encoding='utf-8') as file:
            for _ in range(numberOfSentences):
                sentence = self.generateSentence()
                file.write(' '.join(sentence) + '\n')


class BigramModel(LanguageModel):
    """A bigram language model.

    Args:
        corpus (list of list of str): List of sentences, each represented as a list of words.
    """

    def __init__(self, corpus):
        """Initialize the BigramModel.

        Args:
            corpus (list of list of str): List of sentences, each represented as a list of words.
        """
        super().__init__(corpus)
        self.bigrams = defaultdict(lambda: defaultdict(float))
        self.unigram_counts = defaultdict(float)
        self.train(corpus)

    def train(self, corpus):
        """Train the bigram model.

        Args:
            corpus (list of list of str): List of sentences, each represented as a list of words.
        """
        for sen in corpus:
            previous_word = None
            for word in sen:
                if previous_word is not None:
                    self.bigrams[previous_word][word] += 1.0
                self.unigram_counts[word] += 1.0
                previous_word = word

    def probability(self, previous_word, word):
        """Calculate the conditional probability of a word given the previous word.

        Args:
            previous_word (str): The previous word in the bigram.
            word (str): The word for which to calculate the conditional probability.

        Returns:
            float: The conditional probability of the word given the previous word.
        """
        return self.bigrams[previous_word][word] / self.unigram_counts[previous_word] if self.unigram_counts[previous_word] > 0 else 0.0

    def generateSentence(self):
        """Generate a sentence using the bigram model.

        Returns:
            list of str: The generated sentence.
        """
        sentence = [start]
        while True:
            next_word = self.draw(sentence[-1])
            if next_word != start:
                sentence.append(next_word)
            if next_word == end or len(sentence) > 20:
                break
        if sentence[-1] != end:
            sentence.append(end)
        return sentence

    def draw(self, previous_word):
        """Draw the next word based on the conditional probabilities.

        Args:
            previous_word (str): The previous word in the bigram.

        Returns:
            str: The next word in the sequence.
        """
        words = list(self.bigrams[previous_word].keys())
        probabilities = [self.probability(previous_word, w) for w in words]
        probabilities_sum = sum(probabilities)
        rand = random.random() * probabilities_sum
        for i, word in enumerate(words):
            rand -= probabilities[i]
            if rand <= 0:
                return word
        return end

    def getSentenceProbability(self, sen):
        """Calculate the log probability of a sentence.

        Args:
            sen (list of str): The sentence for which to calculate the log probability.

        Returns:
            float: The log probability of the sentence.
        """
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
        """Calculate the perplexity of a corpus.

        Args:
            corpus (list of list of str): The corpus for which to calculate perplexity.

        Returns:
            float: The perplexity of the corpus.
        """
        total_log_probability = 0.0
        total_bigrams = 0
        for sentence in corpus:
            total_bigrams += len(sentence) - 1
            total_log_probability += self.getSentenceProbability(sentence)
        perplexity = math.exp(-total_log_probability / total_bigrams)
        return perplexity

    def generateSentencesToFile(self, numberOfSentences, filename):
        """Generate sentences using the bigram model and write them to a file.

        Args:
            numberOfSentences (int): The number of sentences to generate.
            filename (str): The name of the file to write the sentences to.
        """
        with open(filename, 'w', encoding='utf-8') as file:
            for _ in range(numberOfSentences):
                sentence = self.generateSentence()
                file.write(' '.join(sentence) + '\n')


class SmoothedBigramModelKN(LanguageModel):
    """A smoothed bigram language model using Kneser-Ney smoothing.

    Args:
        corpus (list of list of str): List of sentences, each represented as a list of words.
        lambda1 (float, optional): Weight parameter for unigram model. Defaults to 0.5.
        lambda2 (float, optional): Weight parameter for bigram model. Defaults to 0.5.
    """

    def __init__(self, corpus, lambda1=0.5, lambda2=0.5):
        """Initialize the SmoothedBigramModelKN.

        Args:
            corpus (list of list of str): List of sentences, each represented as a list of words.
            lambda1 (float, optional): Weight parameter for unigram model. Defaults to 0.5.
            lambda2 (float, optional): Weight parameter for bigram model. Defaults to 0.5.
        """
        super().__init__(corpus)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.bigram_model = BigramModel(corpus)
        self.unigram_model = SmoothedUnigramModel(corpus)

    def probability(self, previous_word, word):
        """Calculate the probability of a word given the previous word.

        Args:
            previous_word (str): The previous word in the bigram.
            word (str): The word for which to calculate the probability.

        Returns:
            float: The probability of the word given the previous word.
        """
        return self.lambda1 * self.unigram_model.probability(word) + self.lambda2 * self.bigram_model.probability(previous_word, word)

    def generateSentence(self):
        """Generate a sentence using the bigram model.

        Returns:
            list of str: The generated sentence.
        """
        return self.bigram_model.generateSentence()

    def getSentenceProbability(self, sen):
        """Calculate the log probability of a sentence.

        Args:
            sen (list of str): The sentence for which to calculate the log probability.

        Returns:
            float: The log probability of the sentence.
        """
        log_probability = 0.0
        previous_word = None
        for word in sen:
            if previous_word is not None:
                prob = self.probability(previous_word, word)
                log_probability += math.log(prob)
            previous_word = word
        return log_probability

    def getCorpusPerplexity(self, corpus):
        """Calculate the perplexity of a corpus.

        Args:
            corpus (list of list of str): The corpus for which to calculate perplexity.

        Returns:
            float: The perplexity of the corpus.
        """
        return self.bigram_model.getCorpusPerplexity(corpus)

    def generateSentencesToFile(self, numberOfSentences, filename):
        """Generate sentences using the bigram model and write them to a file.

        Args:
            numberOfSentences (int): The number of sentences to generate.
            filename (str): The name of the file to write the sentences to.
        """
        with open(filename, 'w', encoding='utf-8') as file:
            for _ in range(numberOfSentences):
                sentence = self.generateSentence()
                file.write(' '.join(sentence) + '\n')


if __name__ == "__main__":
    trainCorpus = readFileToCorpus('train.txt')
    trainCorpus = preprocess(trainCorpus)

    posTestCorpus = readFileToCorpus('pos_test.txt')
    negTestCorpus = readFileToCorpus('neg_test.txt')

    vocab = set()
    for sentence in trainCorpus:
        for word in sentence:
            vocab.add(word)
    print("Task 0: create a vocabulary (collection of word types) for the train corpus")
    print("Vocabulary Size:", len(vocab))

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
    unigramModel.generateSentencesToFile(20, 'output/unigram_output.txt')

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
    smoothedUnigramModel.generateSentencesToFile(
        20, 'output/smooth_unigram_output.txt')

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
    bigramModel.generateSentencesToFile(20, 'output/bigram_output.txt')

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
    smoothedBigramModelKN.generateSentencesToFile(
        20, 'output/smooth_bigram_kn_output.txt')

    # Positive Test Corpus
    print("\nPositive Test Corpus:")
    print("Unigram Model Perplexity:",
          unigramModel.getCorpusPerplexity(posTestCorpus))
    print("Smoothed Unigram Model Perplexity:",
          smoothedUnigramModel.getCorpusPerplexity(posTestCorpus))
    print("Bigram Model Perplexity:",
          bigramModel.getCorpusPerplexity(posTestCorpus))
    print("Smoothed Bigram Model (KN) Perplexity:",
          smoothedBigramModelKN.getCorpusPerplexity(posTestCorpus))

    # Negative Test Corpus
    print("\nNegative Test Corpus:")
    print("Unigram Model Perplexity:",
          unigramModel.getCorpusPerplexity(negTestCorpus))
    print("Smoothed Unigram Model Perplexity:",
          smoothedUnigramModel.getCorpusPerplexity(negTestCorpus))
    print("Bigram Model Perplexity:",
          bigramModel.getCorpusPerplexity(negTestCorpus))
    print("Smoothed Bigram Model (KN) Perplexity:",
          smoothedBigramModelKN.getCorpusPerplexity(negTestCorpus))


""" 
Evaluate the models on the test corpora. Do you see a difference between the two
test domains?
    
The Unigram and Smoothed Unigram models show lower perplexity for 
the negative test corpus compared to the positive, 
indicating a slightly better fit for negative corpus language. 
The Bigram and Smoothed Bigram models exhibit infinite perplexity on both corpora, 
reflecting their struggle with unseen word pairs and inability to differentiate between test domains.
"""


"""
1. When generating sentences with the unigram model, what controls the length of the generated sentences? How does this differ from the sentences produced by the bigram models?
The length of sentences generated by the unigram model is controlled by arbitrary constraints like a maximum sentence length or the selection of an end token. Bigram models, however, generate sentences based on the probabilities of word pairs, potentially leading to more naturally ending sentences influenced by preceding words.

2. Consider the probability of the generated sentences according to your models. Do your models assign drastically different probabilities to the different sets of sentences? Why do you think that is?
Yes, the probabilities assigned by the models vary significantly. Unigram models rely on individual word frequencies, while bigram models consider word pairs, leading to differences based on the structure and coherence of sentences as informed by the training data.

3. Generate additional sentences using your bigram and smoothed bigram models. In your opinion, which model produces better / more realistic sentences?
The Smoothed Bigram Model (KN) likely produces more realistic sentences, as smoothing techniques help it handle unseen word pairs more effectively than the unsmoothed Bigram Model, allowing for a broader range of coherent sentence constructions.

4. For each of the four models, which test corpus has a higher perplexity? Why? Make sure to include the perplexity values in the answer.

Unigram Model: Positive Test Corpus has higher perplexity (653.865 vs. 636.801), likely due to slight differences in language use or vocabulary.
Smoothed Unigram Model: Also, the Positive Test Corpus has higher perplexity (581.570 vs. 564.374), for similar reasons as the Unigram Model.
Bigram Model and Smoothed Bigram Model (KN): Both show infinite perplexity for both corpora, indicating a significant issue with handling unseen bigrams, making them unable to differentiate between the corpora effectively.
"""
