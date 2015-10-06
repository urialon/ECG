from sklearn.neighbors import LSHForest
from Common import offspringsInGeneration
import Common
import random
from Sets import Sets
import multiprocessing

def GetPrediction(y, distances, indices):
    histo = dict()
    sumOfDistances = dict()
    for indexAndDistance in zip(indices, distances):
        if histo.has_key(y[indexAndDistance[0]]):
            histo[y[indexAndDistance[0]]] += 1
            sumOfDistances[y[indexAndDistance[0]]] += indexAndDistance[1]
        else:
            histo[y[indexAndDistance[0]]] = 1
            sumOfDistances[y[indexAndDistance[0]]] = indexAndDistance[1]

    best = histo.keys()[0]

    for key in histo.keys():
        if (key != best):
            if (histo[key] > histo[best]):
                best = key
            elif (histo[key] == histo[best]) & (sumOfDistances[key] < sumOfDistances[best]):
                best = key

    return best

trainingSetSize = 15


def FactorizeVectors(vectorized, factorVector):
    enumerable = range(len(vectorized[0]))
    result = [[(vector[i] - Common.meanVector[i])* factorVector[i] for i in enumerable] for vector in vectorized]
    return result

def score(factors):
    verifyCount = 3
    X, y = Sets.trainingSet
    test_set, databases = Sets.testSet
    X = FactorizeVectors(X, factors)
    test_set = FactorizeVectors(test_set, factors)
    correctionAverage = 0
    for i in range(verifyCount):
        best_predictions = 0
        clf = LSHForest(n_estimators = 10, n_candidates = 10)
        clf.fit(X)

        correct = 0
        total = 0

        for j in range(len(test_set)):
            total += 1
            actual = databases[j]
            distances, indices = clf.kneighbors(test_set[j], n_neighbors=5)
            predicted = GetPrediction(y, distances[0], indices[0])
            if (actual == predicted):
                correct += 1

        if (correct > best_predictions):
            best_predictions = correct
        correctionAverage += best_predictions
    correctionAverage = float(correctionAverage)/verifyCount
    return correctionAverage

def mutate(vector, mutationFactor):
    for i in range(len(vector)):
        shouldMutate = random.choice([0, 1])
        if (shouldMutate == 1):
            mutation = random.uniform(mutationFactor*(-1), mutationFactor)
            if (vector[i] < 0.1) | (vector[i] > 100000):
                vector[i] = vector[i] - (abs(vector[i] - 0.1)/(vector[i] - 0.1)) * vector[i] * abs(mutation)
            else:
                vector[i] *= 1+mutation
    return vector

def singleProcess(vector):
    results = score(vector)
    return results

def RunOneGeneration(startingVector, mutationFactor):
    max_score = 0
    leader = 0
    newVectors = [startingVector] + [mutate(list(startingVector),  mutationFactor) for i in range(offspringsInGeneration -1)]

    pool = multiprocessing.Pool(processes=10)

    scores = pool.map(singleProcess, newVectors)

    pool.close()
    pool.join()

    for i in range(offspringsInGeneration):
        if scores[i] >= max_score:
            max_score = scores[i]
            leader = i
    print('Max Score: ' + max_score.__str__() + ' vector: ' + newVectors[leader].__str__())
    return newVectors[leader], max_score

def Evolution(startingVector, startingMutationFactor):
    lastBestVector = list(startingVector)
    currentMutationFactor = startingMutationFactor
    lastBestScore = 0
    noChangeCounter = 0
    secondBestScore, secondBestVector = 0, list()
    while (True):
        currentVector, currentScore = RunOneGeneration(lastBestVector, currentMutationFactor)

        if (currentScore >= secondBestScore):
            secondBestScore, secondBestVector = currentScore, currentVector
        if currentScore <= lastBestScore:
            noChangeCounter += 1
        else:
            currentMutationFactor = startingMutationFactor
            noChangeCounter = 0
        if (noChangeCounter == 10):
            if (currentMutationFactor < 0.5):
                currentMutationFactor += 0.1
                print('Increasing mutation factor to: ' + currentMutationFactor.__str__())
            noChangeCounter = 0
            lastBestScore, lastBestVector = secondBestScore, secondBestVector
            print('Changing to second best score: '+ secondBestScore.__str__() + ' second best vector: ' + secondBestVector.__str__())
            secondBestScore = 0
            secondBestVector = list()
        if currentScore >= lastBestScore:
            lastBestScore = currentScore
            lastBestVector = currentVector
            print('Taking the current score: ' + lastBestScore.__str__() + ' and vector: ' + lastBestVector.__str__())

startingVector = [268.2219308702712, 1175.543198488771, 1044.9510869045512, 1618.0474956879277, 2861.877445243491, 364.03686992286356, 708.0772292778498, 9323.2014983376, 953.151626413617, 1251.1187443314202]
startingMutationFactor = Common.defaultMutationFactor


if __name__=='__main__':
    Evolution(startingVector, startingMutationFactor)
    #print(score(startingVector))
