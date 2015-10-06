from collections import namedtuple
import operator
from numpy import std, mean
from pymongo import MongoClient, ASCENDING
from pyeeg import ap_entropy


class Constants:
    Database = "Database"
    RecordNumber = "RecordNumber"
    AverageHeartbeat = "AverageHeartbeat"
    IrregularBeatsPercent = "IrregularBeatsPercent"
    AverageBeatChange = "AverageBeatChange"
    Irregularity = "Irregularity"
    QS = "QS"
    QtoR = "QtoR"
    StoR = "StoR"
    QSTD = "QSTD"
    RSTD = "RSTD"
    SSTD = "SSTD"

    ID = "ID"
    Gender = "Gender"
    Smoking = "Smoking"
    familyHistory = "familyHistory"
    Sport = "Sport"
    Age = "Age"
    SystolicBP = "SystolicBP"
    DiastolicBP = "DiastolicBP"
    BMI = "BMI"
    Diagnosis = "Diagnosis"

    Record = "Record"
    PreviousResultsDb = 'PreviousResults'
    FeatureVectors = 'FeatureVectors'
    People = 'People'
    TrainingSetDbName = 'TrainingSet'
    TrainingSetCollectionName = 'TrainingSetCollection'
    LocalHost = "localhost"
    PersonToRecordCollection = "PersonToRecord"

    Time = "Time"
    Value = "Value"
    Label = "Label"
    LabelNone = "None"
    LabelR = "R"
    LabelQ = "Q"
    LabelS = "S"

    MongoPort = 27017

FeatureVector = namedtuple("FeatureVector", " ".join((Constants.Database, Constants.RecordNumber, Constants.AverageHeartbeat, Constants.IrregularBeatsPercent, Constants.AverageBeatChange, Constants.Irregularity, Constants.QS, Constants.QtoR, Constants.StoR, Constants.QSTD, Constants.RSTD, Constants.SSTD)))

Person = namedtuple("Person", " ".join((Constants.ID, Constants.Gender, Constants.Smoking, Constants.familyHistory, Constants.Sport, Constants.Age, Constants.SystolicBP, Constants.DiastolicBP, Constants.BMI, Constants.Diagnosis)))

SamplingRates = {'nsrdb': 128, 'mitdb': 360, 'afdb': 250, 'svdb': 128, 'cudb': 250}
defaultFactor = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

offspringsInGeneration = 10
defaultMutationFactor = 0.1
meanVector = [84.09906486418608, 0.26678132286443434, 0.20721282754655673, 0.970974320025125, 91.95639357742009, 0.3777119473541706, 2.6660855423453014, 285.2128648853569, 328.56882100633914, 520.6623932902744]

def VectorizeFeatureVector(featureVector):
    return [
        featureVector[Constants.AverageHeartbeat],
        featureVector[Constants.IrregularBeatsPercent],
        featureVector[Constants.AverageBeatChange],
        featureVector[Constants.Irregularity],
        featureVector[Constants.QS],
        featureVector[Constants.QtoR],
        featureVector[Constants.StoR],
        featureVector[Constants.QSTD],
        featureVector[Constants.RSTD],
        featureVector[Constants.SSTD]
    ]

def FactorizeVectors(vectorized, factorVector):
    result = [[(vector[i] - meanVector[i]) * factorVector[i] for i in range(len(vector))] for vector in vectorized]
    return result


def GetTrainingSet():
    client = MongoClient(Constants.LocalHost, Constants.MongoPort)
    featureVectorsDb = client.get_database(Constants.FeatureVectors)
    X = list()
    for singleCollection in SamplingRates.keys():
        collection = featureVectorsDb.get_collection(singleCollection)
        limitSize = 15
        if singleCollection == 'nsrdb':
            limitSize = 10
        for item in collection.find().limit(limitSize):
            X.append(item)
    return X


def GetTestSet():
    client = MongoClient(Constants.LocalHost, Constants.MongoPort)
    featureVectorsDb = client.get_database(Constants.FeatureVectors)
    X = list()
    for singleCollection in SamplingRates.keys():
        collection = featureVectorsDb.get_collection(singleCollection)
        skipSize = 15
        if singleCollection == 'nsrdb':
            skipSize = 10
        for item in collection.find().skip(skipSize):
            X.append(item)
    return X


def PutTrainingSet(trainingSet):
    client = MongoClient(Constants.LocalHost, Constants.MongoPort)
    trainingSetDb = client.get_database(Constants.TrainingSetDbName)
    trainingSetDb.drop_collection(Constants.TrainingSetCollectionName)
    trainingSetCollection = trainingSetDb.get_collection(Constants.TrainingSetCollectionName)
    trainingSetCollection.insert_many(trainingSet)
    print(trainingSet)


def GetPeopleTrainingSet():
    client = MongoClient(Constants.LocalHost, Constants.MongoPort)
    trainingSetDb = client.get_database(Constants.TrainingSetDbName)
    trainingSetCollection = trainingSetDb.get_collection(Constants.TrainingSetCollectionName)
    peopleCollection = client.get_database(Constants.People).get_collection(Constants.People)
    personToRecordCollection = client.get_database(Constants.People).get_collection(Constants.PersonToRecordCollection)

    result = list()
    for featureVector in trainingSetCollection.find():
        database = featureVector[Constants.Database]
        record = featureVector[Constants.RecordNumber]
        id = personToRecordCollection.find_one({Constants.Record: record, Constants.Database: database})[Constants.ID]
        person = peopleCollection.find_one({Constants.ID: id})
        result.append(person)
    return result


def PutPeopleTrainingSet(trainingSet):
    client = MongoClient(Constants.LocalHost, Constants.MongoPort)
    trainingSetDb = client.get_database(Constants.TrainingSetDbName)
    trainingSetDb.drop_collection(Constants.People)
    trainingSetCollection = trainingSetDb.get_collection(Constants.People)
    trainingSetCollection.insert_many(trainingSet)
    # print(trainingSet)


def CreateFeatureVector(collection, dbName, takeFirstMinutes):
    limitSamples = SamplingRates[dbName] * 60 * takeFirstMinutes
    lastBeat = -1
    lastQ = -1
    amplitudeSum = {Constants.LabelQ: 0, Constants.LabelR: 0, Constants.LabelS: 0}
    labelsCounters = {Constants.LabelQ: 0, Constants.LabelR: 0, Constants.LabelS: 0}
    amplitudesLists = {Constants.LabelQ: list(), Constants.LabelR: list(), Constants.LabelS: list()}
    heartbeats = list()
    valuesHistogram = dict()
    sumQS = 0
    countQS = 0

    for entry in collection.find().sort(Constants.Time, ASCENDING).limit(limitSamples):
        label = entry[Constants.Label]
        time = entry[Constants.Time]
        value = entry[Constants.Value]

        if label == Constants.LabelNone:
            continue

        amplitudeSum[label] += value
        labelsCounters[label] += 1
        amplitudesLists[label].append(value)

        if value in valuesHistogram:
            valuesHistogram[value] += 1
        else:
            valuesHistogram[value] = 1

        if label == Constants.LabelR:
            if lastBeat > 0:
                heartbeats.append(time - lastBeat)
            lastBeat = time

        elif label == Constants.LabelQ:
            lastQ = time

        elif label == Constants.LabelS:
            if lastQ > 0:
                sumQS += time - lastQ
                countQS += 1
                lastQ = -1

    averageQAmplitude = amplitudeSum[Constants.LabelQ] / float(labelsCounters[Constants.LabelQ])
    averageRAmplitude = amplitudeSum[Constants.LabelR] / float(labelsCounters[Constants.LabelR])
    averageSAmplitude = amplitudeSum[Constants.LabelS] / float(labelsCounters[Constants.LabelS])
    # Calculate the baseline by the most common value
    baseline = max(valuesHistogram.iteritems(), key=operator.itemgetter(1))[0]

    # Convert heartbeats length from a number of samples to actual time
    normalizedHeartbeats = [ float(i) / SamplingRates[dbName] for i in heartbeats ]
    heartbeatSTD = std(normalizedHeartbeats)

    beatChanges = [abs(float(x) - normalizedHeartbeats[i-1])/normalizedHeartbeats[i-1] for i, x in enumerate(normalizedHeartbeats)][1:]
    beatChangesCount = len([change for change in beatChanges if float(change) >= 0.1])
    totalBeatsCount = len(beatChanges)

    result = FeatureVector(
        Database = dbName,
        RecordNumber = collection.name,
        AverageHeartbeat = 60 / mean(normalizedHeartbeats),
        IrregularBeatsPercent = float(beatChangesCount) / totalBeatsCount,
        AverageBeatChange = mean(beatChanges),
        Irregularity = ap_entropy(normalizedHeartbeats, 2, heartbeatSTD*0.2),
        QS = (sumQS / float(countQS)) * 1000 / SamplingRates[dbName],
        QtoR = abs(averageQAmplitude - baseline) / abs(averageRAmplitude - baseline),
        StoR = abs(averageSAmplitude - baseline) / abs(averageRAmplitude - baseline),
        QSTD = std(amplitudesLists[Constants.LabelQ]),
        RSTD = std(amplitudesLists[Constants.LabelR]),
        SSTD = std(amplitudesLists[Constants.LabelS])
    )
    return result


def GetAverageValues():
    client = MongoClient(Constants.LocalHost, Constants.MongoPort)
    db = client.get_database(Constants.FeatureVectors)
    sumVector = [0] * 10
    total = 0
    for singleCollection in SamplingRates.keys():
        collection = db.get_collection(singleCollection)
        for item in collection.find():
            vector = VectorizeFeatureVector(item)
            total += 1
            for i in range(10):
                sumVector[i] += vector[i]
    print('sums: ' + sumVector.__str__())
    for i in range(10):
        sumVector[i] /= float(total)
    return sumVector