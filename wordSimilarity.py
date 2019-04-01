import math
import sys
import heapq


# 2D hashmap, with automatic key-checking
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
            return 0.0
        return self.map[wordA][wordB]

    def getPairedElts(self, wordA):
        self.__ensureKey(wordA)
        return self.map[wordA]

# Maps (wordA, wordB) -> val
# Ensures that pair has same value no matter order: get(a,b) = get(b,a)
class SymPairMap(PairMap):

    def __init__(self, words = []):
        # Create empty map
        self.map = {}
        for word in words:
            self.map[word] = {}

    def put(self, wordA, wordB, val):
        PairMap.put(self, wordA, wordB, val)
        PairMap.put(self, wordB, wordA, val)

    def increment(self, wordA, wordB):
        cur = self.get(wordA, wordB)
        # If the words are the same, increment twice
        if wordA == wordB:
            next = cur + 2.0
        else:
            next = cur + 1.0
        self.put(wordA, wordB, next)
        self.put(wordB, wordA, next)

# Returns the base 10 log of a number
def log10(x):
    return math.log(x) / math.log(10)


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
# Returns: filtered set of words
def processWords(words, stoplist):
    newWords = []
    for word in words:
        word = word.lower()
        if word.isalpha():
            if word not in stoplist:
                newWords.append(word)
    return newWords

# Returns:
# - weightings (dict: str -> PairMap. Keys are "TF", "TFIDF", "PMI")
# - word counts (dict)
def processData(fileName, stoplist):

    # Build pairCount, docFreq
    pairCount = SymPairMap()
    docFreq = {}
    # Extra things to keep track of
    numDocs = 0.0
    numWords = 0.0 # Sum of all occurrences of all words
    wordCounts = {}

    # Process each document
    file = open(fileName, 'r')
    for line in file:
        numDocs += 1.0

        # Remove newline character
        line = line[:-1]

        # Analyze words
        words = line.split()
        words = processWords(words, stoplist)
        numWords += len(words)
        wordsSeen = {}
        for i in range(len(words)):

            word = words[i]

            # Add to word counts
            if word not in wordCounts:
                wordCounts[word] = 0.0
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
                docFreq[word] = 0.0
            docFreq[word] += 1.0
    file.close()

    # Print statistics
    print("Unique Words: " + str(len(wordCounts)))
    print("Word Occurrences: " + str(int(numWords)))
    print("Sentences: " + str(int(numDocs)))

    # Calculate inverse document frequencies
    idfDict = {}
    for word in wordCounts:
        idf = log10(numDocs / docFreq[word])
        idfDict[word] = idf

    invDocFreq = PairMap()
    # Calculate weight for all pairs
    for wordA in wordCounts:
        wordBList = pairCount.getPairedElts(wordA)
        for wordB in wordBList:
            tf = wordBList[wordB]
            idf = idfDict[wordB] # wordA is main word, so weight by doc freq of wordB
            tfIdf = tf * idf
            invDocFreq.put(wordA, wordB, tfIdf)

    # Calculate Pointwise Mutual Information
    pmi = SymPairMap()
    words = list(wordCounts.keys())
    for wordA in wordCounts:
        wordBList = pairCount.getPairedElts(wordA)
        for wordB in wordBList:
            if wordB >= wordA: # Don't do doube the work
                curPmi = 1.0 * wordBList[wordB] * numWords
                curPmi /= wordCounts[wordA] * wordCounts[wordB]
                curPmi = log10(curPmi)
                pmi.put(wordA, wordB, curPmi)

    weightings = {}
    weightings["TF"] = pairCount
    weightings["TFIDF"] = invDocFreq
    weightings["PMI"] = pmi
    return weightings, wordCounts


# Euclidean length of a vector
def getLength(vec):
    sum = 0.0
    for v in vec:
        sum += v ** 2
    return sum ** 0.5

# Normalizes a vector. Mutates, no return
def normalize(vec):
    l = getLength(vec.values())
    if l > 0.0:
        for word in vec:
            vec[word] = 1.0 * vec[word] / l

# Computes distance between two vectors
# similarityMeasure is one of: "L1", "EUCLIDEAN", "COSINE"
# smaller values means more similar
def getSimilarity(vecA, vecB, similarityMeasure):

    if similarityMeasure == "COSINE":
        sim = 0.0
        for k in vecA.keys():
            if k in vecB.keys():
                sim += vecA[k] * vecB[k]
        return -1 * sim

    absDiff = []
    for k in vecA.keys():
        if k in vecB.keys(): # Both
            absDiff.append(abs(vecA[k] - vecB[k]))
        else: # A only
            absDiff.append(abs(vecA[k]))
    for k in vecB.keys():
        if k not in vecA.keys(): # B only
            absDiff.append(abs(vecB[k]))

    if similarityMeasure == "L1":
        return sum(absDiff)
    # EUCLIDEAN
    return getLength(absDiff)

# Generate the bag-of-words vector corresponding to a given word
# weightDict: pre-computer weights of each pair of words
def makeWordVector(word, wordCounts, weightDict):
    vector = {}
    wordBList = weightDict.getPairedElts(word)
    for wordB in wordBList:
        val = wordBList[wordB]
        vector[wordB] = val
    return vector

# Given: a word, all words, pairwise weight function, and distance measure
# Returns: top 10 most similar words, as a list of (word, similarity) tuples
def getMostSimilarWords(word, wordCounts, weightDict, similiarityMeasure):

    queue = []

    vecA = makeWordVector(word, wordCounts, weightDict)
    normalize(vecA)
    for candidate in wordCounts:
        # Candidate must not be same word, and must have occurred at least 3 times
        if candidate != word and wordCounts[candidate] >= 3:
            vecB = makeWordVector(candidate, wordCounts, weightDict)
            normalize(vecB)
            sim = getSimilarity(vecA, vecB, similiarityMeasure)
            heapq.heappush(queue, (sim, candidate))

    similarWords = []
    top = min(10, len(queue)) # Top 10 items from queue
    for i in range(top):
        sim, word = heapq.heappop(queue)
        similarWords.append((word, abs(sim)))

    return similarWords


# Read each line from input file. Prints most similar words for each line
def processInputs(inputFile, weightings, wordCounts):

    file = open(inputFile, 'r')
    for line in file:

        # Remove newline character
        line = line[:-1]

        words = line.split()
        word = words[0]
        weighting = words[1]
        similiarityMeasure = words[2]

        # Return dict: word -> score'
        weightDict = weightings[weighting]
        similarWords = getMostSimilarWords(word, wordCounts, weightDict, similiarityMeasure)
        print("\n" + " ".join(["SIM:", word, weighting, similiarityMeasure]))
        for similarWord, similarity in similarWords:
            print(similarWord + " " + str(similarity))

    file.close()


# Main routine: process stoplist, data, and handle input queries
def main():
    args = sys.argv
    stoplistFile = args[1]
    sentencesFile = args[2] # Training data
    inputFile = args[3] # Queries about word similarity

    stoplist = makeStoplist(stoplistFile)
    weightings, wordCounts = processData(sentencesFile, stoplist)
    processInputs(inputFile, weightings, wordCounts)



if __name__ == "__main__":
    main()
