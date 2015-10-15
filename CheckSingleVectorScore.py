from sklearn.neighbors import LSHForest
import Common
from Sets import Sets

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

def score(factors):
    verifyCount = 50
    neighbors = 5
    estimators = 10
    candidates = 10
    X, y = Sets.trainingSet
    test_set, databases = Sets.testSet
    X = Common.FactorizeVectors(X, factors)
    test_set = Common.FactorizeVectors(test_set, factors)
    vectorAndDatabaseList = zip(test_set, databases)
    best_neighbor, best_candidates, best_estimator, best_predictions = 0, 0, 0, 0
    correctionAverage = 0
    for i in range(verifyCount):
        best_predictions = 0
        clf = LSHForest(n_neighbors=5, n_estimators = 10, n_candidates = 10)
        clf.fit(X)

        correct = 0
        total = 0

        for vectorAndDb in vectorAndDatabaseList:
            total += 1
            actual = vectorAndDb[1]
            #predicted = clf.predict(vectorAndDb[0])[0]
            distances, indices = clf.kneighbors(vectorAndDb[0], n_neighbors=neighbors)
            predicted = GetPrediction(y, distances[0], indices[0])
            if (actual == predicted):
                correct += 1
            #print('Actual: ' + actual +', predicted: ' + predicted)

        if (correct > best_predictions):
            best_predictions = correct
            best_neighbor, best_candidates, best_estimator = neighbors, candidates, estimators
        correctionAverage += best_predictions
    correctionAverage = float(correctionAverage)/verifyCount
    return correctionAverage, best_neighbor, best_candidates, best_estimator

startingVector = [10.45935922250565026, 13211.087958630917, 2551.4811742410716, 921.5007903683774, 398.02065999138813, 112.54104110334723, 133.8470938825284, 650.0264814360967, 162.64549063052917, 160.09765706586313]
startingMutationFactor = Common.defaultMutationFactor

print(score(startingVector))
