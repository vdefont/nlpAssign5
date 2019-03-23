import numpy as np
import math

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

a = 1.0 * np.array([6,5,3,7,2])
b = 1.0 * np.array([5,0,2,0,1])
c = 1.0 * np.array([3,2,2,0,3])

a /= np.array([100,30,40,40,40])
b /= np.array([30,9,12,12,12])

a *= 25
b *= 25

for i in range(len(a)):
    if a[i] != 0:
        a[i] = math.log(a[i]) / math.log(10)
    if b[i] != 0:
        b[i] = math.log(b[i]) / math.log(10)

print(a)
print(b)

dist = getDist(a, b, "COSINE")
print(dist)
