import math
import sys
import heapq

# TODO:
# 1. Debug cosine distance on smaller handmade example 

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
# Returns: filtered set of words
def processWords(words, stoplist):
    newWords = []
    for word in words:
        if word.isalpha():
            word = word.lower()
            if word not in stoplist:
                newWords.append(word)
    return newWords

# Returns:
# - weightings (dict: str -> PairMap. Keys are "TF", "TFIDF", "PMI")
# - word counts (dict)
def processData(fileName, stoplist, inputWords):

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
    for wordA in inputWords:
        for wordB in docFreq:
            tf = pairCount.get(wordA, wordB)
            idf = math.log(numDocs / docFreq[wordB]) / math.log(10.0)
            tfIdf = tf * idf
            invDocFreq.put(wordA, wordB, tfIdf)

    # Calculate Pointwise Mutual Information
    pmi = PairCounter()
    words = list(wordCounts.keys())
    for wordA in inputWords:
        for wordB in words:
            curPmi = 1.0 * pairCount.get(wordA, wordB) * numWords
            curPmi /= wordCounts[wordA] * wordCounts[wordB]
            if curPmi != 0: # Only take the log if it is not 0
                curPmi = math.log(curPmi)
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
    l = getLength(vec)
    if l > 0.0:
        for i in range(len(vec)):
            vec[i] = vec[i] / l

# Computes distance between two vectors
# similarityMeasure is one of: "L1", "EUCLIDEAN", "COSINE"
def getDist(vecA, vecB, similarityMeasure):
    normalize(vecA)
    normalize(vecB)
    if similarityMeasure == "L1":
        dist = 0.0
        for i in range(len(vecA)):
            dist += abs(vecA[i] - vecB[i])
    elif similarityMeasure == "EUCLIDEAN":
        diff = []
        for i in range(len(vecA)):
            diff.append(vecA[i] - vecB[i])
        dist = getLength(diff)
    else: # COSINE
        dist = 0.0
        for i in range(len(vecA)):
            dist += vecA[i] * vecB[i]
    return dist

# Generate the bag-of-words vector corresponding to a given word
# weightDict: pre-computer weights of each pair of words
def makeWordVector(word, wordCounts, weightDict):
    vector = []
    for wordB in wordCounts:
        val = weightDict.get(word, wordB)
        vector.append(val)
    return vector

# Given: a word, all words, pairwise weight function, and distance measure
# Returns: top 10 most similar words, as a list of (word, similarity) tuples
def getMostSimilarWords(word, wordCounts, weightDict, similiarityMeasure):

    queue = []

    vecA = makeWordVector(word, wordCounts, weightDict)
    for candidate in ["copper"]: # TODO loop thru wordCounts
        # Candidate must not be same word, and must have occurred at least 3 times
        if candidate != word and wordCounts[candidate] >= 3:
            vecB = makeWordVector(candidate, wordCounts, weightDict)
            dist = getDist(vecA, vecB, similiarityMeasure)
            heapq.heappush(queue, (-1 * dist, candidate))

    similarWords = []
    top = min(10, len(queue)) # Top 10 items from queue
    for i in range(top):
        similarity, word = heapq.heappop(queue)
        similarWords.append((word, -1 * similarity))

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


# Return dict of words that user will ask for
def getInputWords(inputFile):

    file = open(inputFile, 'r')
    inputWords = {}
    for line in file:

        # Remove newline character
        line = line[:-1]

        words = line.split()
        word = words[0]
        inputWords[word] = 1

    file.close()
    return inputWords


# Main routine: process stoplist, data, and handle input queries
def main():
    args = sys.argv
    stoplistFile = args[1]
    sentencesFile = args[2] # Training data
    inputFile = args[3] # Queries about word similarity

    stoplist = makeStoplist(stoplistFile)
    inputWords = getInputWords(inputFile)
    weightings, wordCounts = processData(sentencesFile, stoplist, inputWords)
    processInputs(inputFile, weightings, wordCounts)

if __name__ == "__main__":
    main()
