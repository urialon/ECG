import Sets
from Common import GetTrainingSet, GetTestSet, VectorizeFeatureVector, Constants


def CalculateMeanVector(vectors):
    sumVector = [0 for i in vectors[0]]
    total = 0
    for singleVector in vectors:
        for i in range(len(singleVector)):
            sumVector[i] += singleVector[i]
        total += 1
    return [float(component)/total for component in sumVector]

trainingSet = GetTrainingSet()
testSet = GetTestSet()

print('Training Set:')
vectorizedTrainingSet = [VectorizeFeatureVector(vector) for vector in trainingSet]
print((vectorizedTrainingSet, [vector[Constants.Database] for vector in trainingSet]))

print('Test Set:')
print(([VectorizeFeatureVector(vector) for vector in testSet], [vector[Constants.Database] for vector in testSet]))

print('Training Set mean vector:')
print(CalculateMeanVector(Sets.Sets.trainingSet[0]))