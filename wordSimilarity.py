import math
import sys
import heapq

# TODO:
# 1. Fix PMI to handle 0 case
# 2. Encapsulate makeStopList() and processSentences() into class called WordSimilarity

class PairMap:

    def __init__(self):
        self.map = {}

    def __ensureKey(self, word):
        if word not in self.map:
            self.map[word] = {}

    def put(self, wordA, wordB, val):
        self.__ensureKey(wordA)
        self.map[wordA][wordB] = val

    # Returns 0 if not found
    def get(self, wordA, wordB):
        self.__ensureKey(wordA)
        if wordB not in self.map[wordA]:
            return 0
        return self.map[wordA][wordB]


# Maps (wordA, wordB) -> val
# Ignores the ordering of the pair
class PairCounter(PairMap):

    def __init__(self, words = []):
        # Create empty map
        self.map = {}
        for word in words:
            self.map[word] = {}

    # Returns sorted pair: (small, large)
    def __sortPair(self, eltA, eltB):
        if eltA < eltB:
            return eltA, eltB
        return eltB, eltA

    def put(self, wordA, wordB, val):
        wordA, wordB = self.__sortPair(wordA, wordB)
        PairMap.put(self, wordA, wordB, val)

    def get(self, wordA, wordB):
        wordA, wordB = self.__sortPair(wordA, wordB)
        return PairMap.get(self, wordA, wordB)

    def increment(self, wordA, wordB):
        cur = self.get(wordA, wordB)
        self.put(wordA, wordB, cur + 1)

# Returns stoplist as set {wordA: 1, wordB: 1, ...}
def makeStoplist(stoplistFile):

    stoplist = {}

    file = open(stoplistFile, 'r')
    for line in file:

        # Remove newline character
        line = line[:-1]

        stoplist[line.lower()] = 1
    file.close()

    return stoplist

# 1. Remove words with non-alphabetic chars
# 2. Lowercase all
# 3. Remove words from stoplist
def processWords(words, stoplist):
    newWords = []
    for word in words:
        if word.isalpha():
            word = word.lower()
            if word not in stoplist:
                newWords.append(word)
    return newWords

# Returns:
# - weightings dict: str -> PairMap. Keys are "TF", "TFIDF", "PMI"
# - words list
def processData(fileName, stoplist):

    # Build pairCount, docFreq
    pairCount = PairCounter()
    docFreq = {}
    # Extra things to keep track of
    numDocs = 0
    numWords = 0 # Sum of all occurrences of all words
    wordCounts = {}

    # Process each document
    file = open(fileName, 'r')
    for line in file:
        numDocs += 1

        # Remove newline character
        line = line[:-1]

        # Analyze words
        words = line.split(" ")
        words = processWords(words, stoplist)
        numWords += len(words)
        wordsSeen = {}
        for i in range(len(words)):

            word = words[i]

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
    file.close()

    # Print statistics
    print("Unique Words: " + str(len(wordCounts)))
    print("Word Occurrences: " + str(numWords))
    print("Sentences: " + str(numDocs))

    # Calculate inverse document frequency
    # wordA is the main word, so we weight by doc freq of second word
    invDocFreq = PairMap()
    for wordA in docFreq:
        for wordB in docFreq:
            tf = pairCount.get(wordA, wordB)
            idf = math.log(numDocs / docFreq[wordB]) / math.log(10.0)
            tfIdf = tf * idf
            invDocFreq.put(wordA, wordB, tfIdf)

    # Calculate Pointwise Mutual Information
    pmi = PairCounter()
    words = list(wordCounts.keys())
    for i in range(len(words)):
        for j in range(i, len(words)):
            wordA = words[i]
            wordB = words[j]
            curPmi = 1.0 * pairCount.get(wordA, wordB) * numWords
            curPmi /= wordCounts[wordA] * wordCounts[wordB]
            if curPmi != 0: # Only take the log if it is not 0
                curPmi = math.log(curPmi)
            pmi.put(wordA, wordB, curPmi)

    weightings = {}
    weightings["TF"] = pairCount
    weightings["TFIDF"] = invDocFreq
    weightings["PMI"] = pmi
    wordList = wordCounts.keys()
    return weightings, wordList

def makeWordVector(word, words, weightDict):
    vector = []
    for wordB in words:
        vector.append(weightDict.get(word, wordB))
    return vector

# TODO: add more than euclidian
def getDist(vecA, vecB, similiarityMeasure):
    sum = 0.0
    for i in range(len(vecA)):
        sum += (vecA[i] + vecB[i]) ** 2
    return sum ** 0.5

def getMostSimilarWords(word, words, weightDict, similiarityMeasure):

    queue = []

    vecA = makeWordVector(word, words, weightDict)
    for candidate in words:
        if candidate != word:
            vecB = makeWordVector(candidate, words, weightDict)
            dist = getDist(vecA, vecB, similiarityMeasure)
            heapq.heappush(queue, (dist, candidate))

    similarWords = {}
    top = min(10, len(queue)) # Top 10 items from queue
    for i in range(top):
        similarity, word = heapq.heappop(queue)
        similarWords[word] = similarity

    return similarWords

def processInputs(inputFile, weightings, wordsList):

    file = open(inputFile, 'r')
    for line in file:

        # Remove newline character
        line = line[:-1]

        words = line.split(" ")
        word = words[0]
        weighting = words[1]
        similiarityMeasure = words[2]

        # Return dict: word -> score'
        weightDict = weightings[weighting]
        similarWords = getMostSimilarWords(word, wordsList, weightDict, similiarityMeasure)
        print("\n" + " ".join(["SIM:", word, weighting, similiarityMeasure]))
        for similarWord in similarWords:
            print(" ".join([similarWord, str(similarWords[similarWord])]))

    file.close()

args = sys.argv
stoplistFile = args[1]
sentencesFile = args[2] # Training data
inputFile = args[3] # Queries about word similarity

stoplist = makeStoplist(stoplistFile)
weightings, words = processData(sentencesFile, stoplist)
processInputs(inputFile, weightings, words)
