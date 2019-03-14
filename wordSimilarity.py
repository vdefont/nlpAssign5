import math
import sys

# TODO:
# 1. Fix PMI to handle 0 case
# 2. Encapsulate makeStopList() and processSentences() into class called WordSimilarity

# Maps (wordA, wordB) -> val
# Ignores the ordering of the pair
class PairCounter:

    map = {}

    def __init__(self, words = []):
        # Create empty map
        for word in words:
            self.map[word] = {}

    def ensureKey(self, word):
        if word not in self.map:
            self.map[word] = {}

    # Returns sorted pair: (small, large)
    def sortPair(self, eltA, eltB):
        if eltA < eltB:
            return eltA, eltB
        return eltB, eltA

    def increment(self, wordA, wordB):
        wordA, wordB = self.sortPair(wordA, wordB)
        self.ensureKey(wordA)
        # Ensure key
        if wordB not in self.map[wordA]:
            self.map[wordA][wordB] = 0
        # Increment
        self.map[wordA][wordB] += 1

    def put(self, wordA, wordB, val):
        wordA, wordB = self.sortPair(wordA, wordB)
        self.ensureKey(wordA)
        self.map[wordA][wordB] = val

    # Returns 0 if not found
    def get(self, wordA, wordB):
        wordA, wordB = self.sortPair(wordA, wordB)
        self.ensureKey(wordA)
        if wordB not in self.map[wordA]:
            return 0
        return self.map[wordA][wordB]

stoplist = {}

def makeStoplist(stoplistFile):
    file = open(stoplistFile, 'r')
    for line in file:

        # Remove newline character
        line = line[:-1]

        stoplist[line] = 1

# Returns:
# - pairCount (PairCounter: pair -> int)
# - docFreq (map: word -> int)
# - invDocFreq (map: word -> float)
# - pmi (PairCounter: pair -> float)
def processSentences(fileName):
    # Build pairCount, docFreq
    pairCount = PairCounter()
    docFreq = {}
    # Extra things to keep track of
    numDocs = 0
    numWords = 0 # Sum of all occurrences of all words
    wordCounts = {}
    file = open(fileName, 'r')
    for line in file:
        numDocs += 1

        # Remove newline character
        line = line[:-1]

        # Analyze words
        words = line.split(" ")
        wordsSeen = {}
        for i in range(len(words)):

            word = words[i]

            # Ensure not in stoplist
            if word in stoplist:
                continue

            numWords += 1

            # Add to word counts
            if word not in wordCounts:
                wordCounts[word] = 0
            wordCounts[word] += 1

            # Mark word as seen
            wordsSeen[word] = 1

            # Update pairCount
            if i + 1 < len(words):
                pairCount.increment(word, words[i+1])
            if i + 2 < len(words):
                pairCount.increment(word, words[i+2])

        # Update docFreq
        for word in wordsSeen:
            if word not in docFreq:
                docFreq[word] = 0
            docFreq[word] += 1

    # Calculate inverse document frequency
    invDocFreq = {}
    for word in docFreq:
        invDocFreq[word] = math.log(numDocs / docFreq[word])
    print(invDocFreq)

    # Calculate Pointwise Mutual Information
    pmi = PairCounter()
    words = list(wordCounts.keys())
    for i in range(len(words)):
        for j in range(i, len(words)):
            wordA = words[i]
            wordB = words[j]
            curPmi = 1.0 * pairCount.get(wordA, wordB) * numWords
            curPmi /= wordCounts[wordA] * wordCounts[wordB]
            print(curPmi)
            curPmi = math.log(curPmi)
            pmi.put(wordA, wordB, curPmi)
    print(pmi.map)

args = sys.argv
stoplistFile = args[1]
sentencesFile = args[2] # Training data
inputFile = args[3] # Queries about word similarity

makeStoplist(stoplistFile)
processSentences(sentencesFile)
